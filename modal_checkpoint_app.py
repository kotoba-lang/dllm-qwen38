"""Isolated H200 x8 tournament: full activation checkpointing vs disabled.

TF32 is explicitly off in both legs.  The two fresh 27B runs execute sequentially in the
same container and differ only in ``--grad-checkpoint``.  The shared volume is used as an
uncommitted Hugging Face cache; all reports are transient under ``/tmp`` and no checkpoint
directory is ever passed to the trainer.
"""

from __future__ import annotations

import json
import os
import secrets as random_secrets
import subprocess
import time
import uuid

import modal


app = modal.App("dllm-qwen38-mfu-checkpoint")

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

cache_vol = modal.Volume.from_name("dllm-qwen38-cache")
hf_secrets = [modal.Secret.from_name("hf-token")]

WARMUP_STEPS = 10
MEASURED_STEPS = 30
TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS


def _resolved_order(order: str) -> str:
    value = order.strip().upper()
    if value == "RANDOM":
        return random_secrets.choice(("AB", "BA"))
    if value not in {"AB", "BA"}:
        raise ValueError("order must be random, AB, or BA")
    return value


def _run_leg(
    *,
    root: str,
    name: str,
    grad_checkpoint: bool,
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
        "--master_port=29519",
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
        "--out",
        out,
    ]
    if grad_checkpoint:
        argv.append("--grad-checkpoint")
    # No --tf32: fsdp_train explicitly configures the baseline as highest/allow_tf32=False.
    # No --ckpt-dir, --ckpt-every, or --resume: experiment output cannot touch /cache/ckpts.

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
            "grad_checkpoint_requested": grad_checkpoint,
            "tf32_requested": False,
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
def paired_checkpoint_remote(
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
    from dllm_qwen38.mfu_metrics import checkpoint_comparison

    resolved = _resolved_order(order)
    experiment_id = f"mfu-checkpoint-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
        "A": {"name": "full_checkpoint", "grad_checkpoint": True},
        "B": {"name": "no_checkpoint", "grad_checkpoint": False},
    }
    reports = {}
    for key in resolved:
        config = configs[key]
        reports[config["name"]] = _run_leg(
            root=root,
            name=config["name"],
            grad_checkpoint=config["grad_checkpoint"],
            model=model,
            data=data,
            split=split,
            batch=batch,
            seq_len=seq_len,
            block=block,
            lr=lr,
            seed=seed,
        )

    summary = checkpoint_comparison(
        reports["full_checkpoint"], reports["no_checkpoint"]
    )
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
            "gdn": "torch",
            "tf32": False,
            "only_independent_variable": "gradient_checkpointing",
        },
        "reports": reports,
        "summary": summary,
        "gate_policy": {
            "fail_closed": True,
            "correctness": "both exit zero; TF32 off; modes match; 30 measured steps; finite; rank spread zero; max paired loss diff <= 0.01",
            "safety": "candidate peak allocated memory <= 136 GiB",
            "performance": "candidate speedup >= 1.10",
            "promotion": "correctness AND safety AND performance",
        },
        "isolation": {
            "checkpoint_dir_supplied": False,
            "volume_committed": False,
            "outputs_persistent": False,
            "transient_root": root,
        },
    }
    print("MFU-CHECKPOINT-PAIR\t" + json.dumps(summary, sort_keys=True), flush=True)
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
    result = paired_checkpoint_remote.remote(
        order, model, data, split, batch, seq_len, block, lr, seed
    )
    print(json.dumps(result, indent=1))
