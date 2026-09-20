"""Isolated Modal tournament for the 27B TF32 optimization candidate.

This file intentionally does not import ``modal_app``: importing the production app would
also register its functions and makes accidental checkpoint/volume writes easier.  Both
legs run sequentially on the same H200 x8 container.  The existing cache volume is mounted
only to reuse already-downloaded Hugging Face artifacts; this experiment never commits it,
never supplies ``--ckpt-dir``, and writes transient reports under ``/tmp``.

Run without changing production behavior::

    modal run modal_mfu_app.py --order random
    modal run modal_mfu_app.py --order AB
    modal run modal_mfu_app.py --order BA
"""

from __future__ import annotations

import json
import os
import secrets as random_secrets
import subprocess
import time
import uuid

import modal


app = modal.App("dllm-qwen38-mfu-tf32")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6",
        "transformers>=5.17",
        "datasets",
        "safetensors",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "flash-linear-attention",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir("src/dllm_qwen38", remote_path="/root/dllm_qwen38")
)

# No create_if_missing and, critically, no commit(): the experiment treats this as an
# existing read-through model cache, not as an output or checkpoint volume.
cache_vol = modal.Volume.from_name("dllm-qwen38-cache")
hf_secrets = [modal.Secret.from_name("hf-token")]

WARMUP_STEPS = 10
MEASURED_STEPS = 30
TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS
H200_BF16_DENSE_FLOPS = 989e12
MODEL_PARAMS = 27e9
EXECUTED_POSITIONS_PER_UNIQUE_TOKEN = 4.0  # complementary views x clean/noised streams


def _resolved_order(order: str) -> str:
    value = order.strip().upper()
    if value == "RANDOM":
        return random_secrets.choice(("AB", "BA"))
    if value not in {"AB", "BA"}:
        raise ValueError("order must be random, AB, or BA")
    return value


def _proxy_mfu(seq_tokens_per_s: float, world: int = 8) -> float:
    """6N useful-training proxy divided by dense BF16 Tensor Core peak."""
    useful_flops_s = (
        6.0 * MODEL_PARAMS * EXECUTED_POSITIONS_PER_UNIQUE_TOKEN * seq_tokens_per_s
    )
    return useful_flops_s / (world * H200_BF16_DENSE_FLOPS)


def _comparison(baseline: dict, candidate: dict) -> dict:
    base_tps = float(baseline.get("measured_seq_tokens_per_s") or 0.0)
    cand_tps = float(candidate.get("measured_seq_tokens_per_s") or 0.0)
    speedup = cand_tps / base_tps if base_tps > 0.0 else None
    base_losses = baseline.get("losses") or []
    cand_losses = candidate.get("losses") or []
    paired = min(len(base_losses), len(cand_losses))
    loss_diffs = [abs(float(cand_losses[i]) - float(base_losses[i])) for i in range(paired)]
    return {
        "baseline_measured_seq_tokens_per_s": base_tps,
        "candidate_measured_seq_tokens_per_s": cand_tps,
        "candidate_speedup": speedup,
        "candidate_cost_ratio": (1.0 / speedup) if speedup else None,
        "baseline_proxy_mfu": _proxy_mfu(base_tps),
        "candidate_proxy_mfu": _proxy_mfu(cand_tps),
        "proxy_mfu_definition": "6*N*4*unique_sequence_tokens_per_s/(8*989e12), N=27e9",
        "paired_loss_steps": paired,
        "loss_max_abs_diff": max(loss_diffs) if loss_diffs else None,
        "loss_mean_abs_diff": sum(loss_diffs) / paired if paired else None,
        "both_exited_zero": baseline.get("exit") == 0 and candidate.get("exit") == 0,
        "both_weight_spread_zero": baseline.get("weight_spread") == 0.0
        and candidate.get("weight_spread") == 0.0,
    }


def _run_leg(
    *,
    root: str,
    name: str,
    tf32: bool,
    model: str,
    data: str,
    split: str,
    batch: int,
    seq_len: int,
    block: int,
    lr: float,
    seed: int,
) -> dict:
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=False)
    argv = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29518",
        "--module",
        "dllm_qwen38.fsdp_train",
        "--model",
        model,
        "--data",
        data,
        "--split",
        split,
        "--steps",
        str(TOTAL_STEPS),
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--batch",
        str(batch),
        "--seq-len",
        str(seq_len),
        "--block",
        str(block),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--gdn",
        "torch",
        "--grad-checkpoint",
        "--out",
        out,
    ]
    if tf32:
        argv.append("--tf32")

    started = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True)
    report_path = os.path.join(out, "report.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as handle:
            report = json.load(handle)
    report.update(
        {
            "leg": name,
            "tf32_requested": tf32,
            "exit": proc.returncode,
            "wall_total_s": round(time.time() - started, 3),
            "transient_run_dir": out,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    )
    return report


@app.function(
    image=image,
    gpu="H200:8",
    timeout=4 * 3600,
    volumes={"/cache": cache_vol},
    secrets=hf_secrets,
)
def paired_tf32_remote(
    order: str,
    model: str,
    data: str,
    split: str,
    batch: int,
    seq_len: int,
    block: int,
    lr: float,
    seed: int,
) -> dict:
    resolved = _resolved_order(order)
    experiment_id = f"mfu-tf32-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = os.path.join("/tmp", experiment_id)
    os.makedirs(root, exist_ok=False)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,pci.bus_id",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    configs = {
        "A": {"name": "baseline", "tf32": False},
        "B": {"name": "tf32", "tf32": True},
    }
    reports = {}
    for key in resolved:
        config = configs[key]
        report = _run_leg(
            root=root,
            name=config["name"],
            tf32=config["tf32"],
            model=model,
            data=data,
            split=split,
            batch=batch,
            seq_len=seq_len,
            block=block,
            lr=lr,
            seed=seed,
        )
        reports[config["name"]] = report

    summary = _comparison(reports["baseline"], reports["tf32"])
    result = {
        "experiment_id": experiment_id,
        "requested_order": order,
        "resolved_order": resolved,
        "gpu_inventory": gpu.replace("\n", " | "),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "shape": {
            "model": model,
            "data": data,
            "split": split,
            "batch": batch,
            "seq_len": seq_len,
            "block": block,
            "lr": lr,
            "seed": seed,
            "world": 8,
            "gradient_checkpointing": True,
            "gdn": "torch",
        },
        "reports": reports,
        "summary": summary,
        "isolation": {
            "checkpoint_dir_supplied": False,
            "volume_committed": False,
            "outputs_persistent": False,
            "transient_root": root,
        },
    }
    print("MFU-PAIR\t" + json.dumps(summary, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def run(
    order: str = "random",
    model: str = "Qwen/Qwen3.8-27B",
    data: str = "nvidia/Llama-Nemotron-Post-Training-Dataset",
    split: str = "chat",
    batch: int = 8,
    seq_len: int = 512,
    block: int = 32,
    lr: float = 1e-5,
    seed: int = 0,
):
    result = paired_tf32_remote.remote(
        order, model, data, split, batch, seq_len, block, lr, seed
    )
    print(json.dumps(result, indent=1))
