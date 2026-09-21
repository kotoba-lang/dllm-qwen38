"""Isolated H200 x8 tournament: PyTorch vs compiled causal-conv1d.

The workload is intentionally fixed to the accepted 27B baseline shape.  Both fresh runs use
TF32-off, full per-layer activation checkpointing, the torch GDN core, identical data order and
seed, and execute sequentially in one container.  Reports live only under ``/tmp``; the shared
volume is an uncommitted Hugging Face cache and no training checkpoint argument is accepted.
"""

from __future__ import annotations

import json
import os
import secrets as random_secrets
import subprocess
import time
import uuid

import modal


app = modal.App("dllm-qwen38-mfu-causal-conv")

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
        "ninja",
        "packaging",
        "wheel",
    )
    # causal-conv1d's upstream build imports the already-installed torch package.  The second
    # layer and --no-build-isolation are therefore intentional.  A missing binary extension is
    # still rejected at runtime; a Python package import alone never qualifies the candidate.
    .run_commands("python -m pip install causal-conv1d==1.7.0 --no-build-isolation")
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

MODEL = "Qwen/Qwen3.8-27B"
DATA = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SPLIT = "chat"
GLOBAL_BATCH = 8
SEQ_LEN = 512
BLOCK = 32
LR = 1e-5
SEED = 0
WORLD = 8
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


def _run_leg(*, root: str, name: str, causal_conv: str) -> dict:
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=False)
    argv = [
        "torchrun",
        f"--nproc_per_node={WORLD}",
        "--master_port=29521",
        "--module",
        "dllm_qwen38.fsdp_train",
        "--model",
        MODEL,
        "--data",
        DATA,
        "--split",
        SPLIT,
        "--steps",
        str(TOTAL_STEPS),
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--batch",
        str(GLOBAL_BATCH),
        "--seq-len",
        str(SEQ_LEN),
        "--block",
        str(BLOCK),
        "--lr",
        str(LR),
        "--seed",
        str(SEED),
        "--gdn",
        "torch",
        "--causal-conv",
        causal_conv,
        "--grad-checkpoint",
        "--out",
        out,
    ]
    if causal_conv == "optimized":
        argv.append("--causal-conv-probe")
    # No --tf32: fsdp_train explicitly selects highest/allow_tf32=False.
    # No --ckpt-dir, --ckpt-every, --resume, or target-step: /cache/ckpts is unreachable.
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
            "causal_conv_requested": causal_conv,
            "grad_checkpoint_requested": True,
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
def paired_causal_conv_remote(order: str) -> dict:
    from dllm_qwen38.mfu_metrics import causal_conv_comparison

    resolved = _resolved_order(order)
    experiment_id = f"mfu-causal-conv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
        "A": {"name": "reference", "causal_conv": "reference"},
        "B": {"name": "optimized", "causal_conv": "optimized"},
    }
    reports = {}
    for key in resolved:
        config = configs[key]
        reports[config["name"]] = _run_leg(root=root, **config)

    summary = causal_conv_comparison(reports["reference"], reports["optimized"])
    result = {
        "experiment_id": experiment_id,
        "requested_order": order,
        "resolved_order": resolved,
        "gpu_inventory": gpu.replace("\n", " | "),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "shape": {
            "model": MODEL,
            "data": DATA,
            "split": SPLIT,
            "batch": GLOBAL_BATCH,
            "seq_len": SEQ_LEN,
            "block": BLOCK,
            "lr": LR,
            "seed": SEED,
            "world": WORLD,
            "gdn": "torch",
            "tf32": False,
            "gradient_checkpointing": True,
            "only_independent_variable": "causal_conv_backend",
        },
        "reports": reports,
        "summary": summary,
        "gate_policy": {
            "fail_closed": True,
            "correctness": "both exit zero; requested backends active; compiled extension/version present; kernel output diff <=0.02, gradient cosine >=0.999 and norm ratio 0.99-1.01; 30 paired finite losses with max diff <=0.01; rank spread zero",
            "safety": "candidate peak <=136 GiB and <=reference+2 GiB",
            "performance": "optimized/reference measured throughput >=1.08",
            "promotion": "correctness AND safety AND performance",
        },
        "isolation": {
            "checkpoint_dir_supplied": False,
            "volume_committed": False,
            "outputs_persistent": False,
            "transient_root": root,
        },
    }
    print("MFU-CAUSAL-CONV-PAIR\t" + json.dumps(summary, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def run(order: str = "random"):
    result = paired_causal_conv_remote.remote(order)
    print(json.dumps(result, indent=1))
