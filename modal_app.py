"""Modal entry points (owner directive 2026-09-12: "train は modal を使って").

    modal run modal_app.py::equivalence --model Qwen/Qwen3.8-27B            # ADR §3a on the 27B itself
    modal run modal_app.py::equivalence --model Qwen/Qwen3.8-27B --layers 16 --dtype float32
    modal run modal_app.py::train_run --model Qwen/Qwen3.5-0.8B --steps 200  # Phase 2 loop smoke, tok/s

Everything the container produces lands in the volume `dllm-qwen38-cache` (`/cache/hf` = HF
cache, `/cache/runs/<name>/` = reports + checkpoints) and is echoed back as the return value.
Single GPU only; the 27B training run needs FSDP and is not here yet.
"""

from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("dllm-qwen38")

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
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("src/dllm_qwen38", remote_path="/root/dllm_qwen38")
)

vol = modal.Volume.from_name("dllm-qwen38-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]


def _run_dir(name: str) -> str:
    d = f"/cache/runs/{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(d, exist_ok=True)
    return d


@app.function(image=image, gpu="H100", timeout=4 * 3600, volumes={"/cache": vol}, secrets=secrets)
def equivalence_remote(model: str, layers: int | None, block: int, nblocks: int, dtype: str, tol: float | None, oracle_kernels: str) -> dict:
    import subprocess
    import sys

    if oracle_kernels == "reference":
        # make `import fla` fail before transformers resolves its hub-kernel fallbacks, so the
        # oracle runs HF's torch reference path instead of the Triton kernel
        sys.modules["fla"] = None
    from dllm_qwen38 import equivalence as eq

    d = _run_dir("equivalence")
    argv = ["--model", model, "--block", str(block), "--nblocks", str(nblocks), "--dtype", dtype, "--device", "cuda", "--json", f"{d}/report.json"]
    if layers:
        argv += ["--layers", str(layers)]
    if tol is not None:
        argv += ["--tol", str(tol)]
    t0 = time.time()
    code = eq.main(argv)
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    rep.update({"exit": code, "wall_total_s": round(time.time() - t0, 1), "run_dir": d, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()})
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


@app.function(image=image, gpu="H100", timeout=8 * 3600, volumes={"/cache": vol}, secrets=secrets)
def train_remote(model: str, data: str, split: str, steps: int, batch: int, seq_len: int, block: int, lr: float) -> dict:
    import subprocess

    from dllm_qwen38 import train as tr

    d = _run_dir("train")
    argv = ["--model", model, "--data", data, "--split", split, "--steps", str(steps), "--batch", str(batch), "--seq-len", str(seq_len), "--block", str(block), "--lr", str(lr), "--device", "cuda", "--out", d]
    t0 = time.time()
    code = tr.main(argv)
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    rep.pop("losses", None)
    rep.update({"exit": code, "wall_total_s": round(time.time() - t0, 1), "run_dir": d, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()})
    vol.commit()
    return rep


@app.local_entrypoint()
def equivalence(model: str = "Qwen/Qwen3.8-27B", layers: int = 0, block: int = 32, nblocks: int = 4, dtype: str = "bfloat16", tol: float = -1.0, oracle_kernels: str = "reference"):
    rep = equivalence_remote.remote(model, layers or None, block, nblocks, dtype, None if tol < 0 else tol, oracle_kernels)
    print(json.dumps(rep, indent=1))


@app.local_entrypoint()
def train_run(model: str = "Qwen/Qwen3.5-0.8B", data: str = "nvidia/Llama-Nemotron-Post-Training-Dataset", split: str = "chat", steps: int = 200, batch: int = 4, seq_len: int = 512, block: int = 32, lr: float = 1e-5):
    rep = train_remote.remote(model, data, split, steps, batch, seq_len, block, lr)
    print(json.dumps(rep, indent=1))
