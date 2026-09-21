"""Isolated H200 x8 tournament: sibling FSDP2 roots vs one called true root.

Both fresh 27B legs keep the accepted workload fixed: global batch 8, sequence 512,
block 32, TF32 disabled, torch GDN and reference causal-conv, and full per-layer
non-reentrant activation checkpointing. The only independent variable is the FSDP2
execution topology. Reports live under ``/tmp``; the mounted volume is an uncommitted
read-through model cache and no production checkpoint argument is accepted.
"""

from __future__ import annotations

import json
import os
import secrets as random_secrets
import subprocess
import time
import uuid

import modal


app = modal.App("dllm-qwen38-mfu-fsdp-root")

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

MODEL = "Qwen/Qwen3.8-27B"
DATA = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SPLIT = "chat"
GLOBAL_BATCH = 8
SEQ_LEN = 512
BLOCK = 32
LR = 1e-5
SEED = 0
WORLD = 8
LAYERS = 64
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


def _run_leg(*, root: str, name: str, topology: str) -> dict:
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=False)
    argv = [
        "torchrun",
        f"--nproc_per_node={WORLD}",
        "--master_port=29522",
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
        "reference",
        "--fsdp-topology",
        topology,
        "--checkpoint-schema-probe",
        "--grad-checkpoint",
        "--out",
        out,
    ]
    # No --tf32: fsdp_train explicitly selects highest/allow_tf32=False.
    # No checkpoint/resume/target arguments and no Volume commit: production state is
    # unreachable from both tournament legs.
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
            "fsdp_topology_requested": topology,
            "causal_conv_requested": "reference",
            "grad_checkpoint_requested": True,
            "tf32_requested": False,
            "exit": proc.returncode,
            "wall_total_s": round(time.time() - started, 3),
            "transient_run_dir": out,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-6000:],
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
def paired_fsdp_root_remote(order: str) -> dict:
    from dllm_qwen38.mfu_metrics import fsdp_root_comparison

    resolved = _resolved_order(order)
    experiment_id = f"mfu-fsdp-root-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
        "A": {"name": "sibling_baseline", "topology": "sibling"},
        "B": {"name": "true_root", "topology": "root"},
    }
    reports = {}
    for key in resolved:
        config = configs[key]
        reports[config["name"]] = _run_leg(root=root, **config)

    summary = fsdp_root_comparison(
        reports["sibling_baseline"], reports["true_root"], expected_layers=LAYERS
    )
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
            "layers": LAYERS,
            "gdn": "torch",
            "causal_conv": "reference",
            "tf32": False,
            "gradient_checkpointing": "full_non_reentrant_per_layer",
            "only_independent_variable": "fsdp_execution_topology",
        },
        "reports": reports,
        "summary": summary,
        "gate_policy": {
            "fail_closed": True,
            "structure": "candidate has one called root/shared state+comm context, 64 unique child groups in post-forward order, identical evidence on all ranks",
            "correctness": "both exit zero; exact workload modes; 30 finite paired losses with max diff <=0.01; first global grad-norm ratio 0.99-1.01; rank spread zero; DCP model schema digest identical",
            "activation": "both report 64 non-reentrant checkpoint regions per forward",
            "safety": "candidate peak <=136 GiB and <=baseline+5 GiB",
            "performance": "candidate/baseline measured throughput >=1.08",
            "promotion": "structure AND correctness AND safety AND performance",
        },
        "isolation": {
            "checkpoint_dir_supplied": False,
            "volume_committed": False,
            "outputs_persistent": False,
            "transient_root": root,
        },
    }
    print("MFU-FSDP-ROOT-PAIR\t" + json.dumps(summary, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def run(order: str = "random"):
    result = paired_fsdp_root_remote.remote(order)
    print(json.dumps(result, indent=1))
