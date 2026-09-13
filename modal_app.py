"""Modal entry points (owner directive 2026-09-12: "train は modal を使って").

    modal run modal_app.py::equivalence --model Qwen/Qwen3.8-27B            # ADR §3a on the 27B itself
    modal run modal_app.py::equivalence --model Qwen/Qwen3.8-27B --layers 16 --dtype float32
    modal run modal_app.py::train_run --model Qwen/Qwen3.5-0.8B --steps 200  # Phase 2 loop smoke, tok/s
    modal run modal_app.py::eval_run --mode ar                              # GSM8K subset, AR baseline
    modal run modal_app.py::eval_run --mode dllm --ckpt dllm-27b-1pct       # block-diffusion decode

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
    # PYTHONPATH=/root because fsdp_train.py is launched with `--module`, which needs the
    # package dir's parent on the path (set here, on the base image: Modal rejects build
    # steps layered after add_local_*)
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": "/root"})
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


@app.function(image=image, gpu="H100", timeout=4 * 3600, volumes={"/cache": vol}, secrets=secrets)
def lever_bench_remote(model: str, data: str, split: str, steps: int, batch: int, seq_len: int, block: int, lr: float) -> dict:
    """Throughput levers, each measured as its own leg on fresh-loaded weights.

    First the kernel-diff probe (fla Triton vs torch reference on real weights, bf16, with
    per-block states — the ADR's open fla-vs-reference item), printed immediately so a later
    crash cannot lose it. Then one lever at a time, 50-step training legs on fresh-loaded
    weights each: baseline (eager attn + reference GDN core) → +sdpa (attention lever) → +fla
    core (GDN core lever, the one that measured 4× down on the first run) → +per-layer
    gradient checkpointing → +double batch under the freed memory → checkpointing also on the
    fla core. (The checkpoint legs were first measured 2026-09-12 as non-training: the
    checkpointed function closed over the layer loop variable, so recompute re-ran the last
    layer for every checkpoint — grads only on 13/108 params, loss flat. partial() fixes it;
    non-reentrant mode re-measures them.) Each leg reloads the model so leg N does not
    inherit leg N−1's allocator state.
    """
    import subprocess

    import torch
    from transformers import AutoTokenizer

    from dllm_qwen38 import gdn, train as tr
    from dllm_qwen38.equivalence import load_checkpoint
    from dllm_qwen38.objective import mask_token_id

    dtype = torch.bfloat16
    rep = {"model": model, "data": data, "split": split, "steps_per_leg": steps, "seq_len": seq_len, "block": block, "lr": lr}

    text_model, lm_head, _ = load_checkpoint(model, dtype, "cuda", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(model)
    mask_id = mask_token_id(text_model, tok)

    i_lin = next(i for i, t in enumerate(text_model.config.layer_types) if t == "linear_attention")
    rep["kernel_diff"] = gdn.kernel_diff(text_model.layers[i_lin].linear_attn, T=block, B=2, device="cuda")
    print(f"KERNEL_DIFF\t{json.dumps(rep['kernel_diff'])}")  # printed now, not only in the final json — a later leg crash must not lose it
    torch.cuda.empty_cache()

    # One lever at a time: baseline→sdpa-torch isolates the attention backend, sdpa-torch→fla-sdpa
    # isolates the GDN core. The ckpt legs ride the torch-core stack (the fastest so far measured);
    # fla-sdpa-ckpt is kept so the ckpt lever is also seen on the other core.
    legs = [
        {"name": "baseline", "attn": "eager", "gdn": "torch", "ckpt": False, "batch": batch},
        {"name": "sdpa-torch", "attn": "sdpa", "gdn": "torch", "ckpt": False, "batch": batch},
        {"name": "fla-sdpa", "attn": "sdpa", "gdn": None, "ckpt": False, "batch": batch},
        {"name": "sdpa-torch-ckpt", "attn": "sdpa", "gdn": "torch", "ckpt": True, "batch": batch},
        {"name": "sdpa-torch-ckpt-b2x", "attn": "sdpa", "gdn": "torch", "ckpt": True, "batch": batch * 2},
        {"name": "fla-sdpa-ckpt", "attn": "sdpa", "gdn": None, "ckpt": True, "batch": batch},
    ]
    out = []
    for i, leg in enumerate(legs):
        gdn.set_impl(leg["gdn"])
        text_model, lm_head, _ = load_checkpoint(model, dtype, "cuda", attn_implementation=leg["attn"])
        batches = tr.chat_batches(data, split, tok, leg["batch"], seq_len, block, seed=i)
        try:
            r = tr.train(text_model, lm_head, batches, block=block, mask_id=mask_id, steps=steps, lr=lr, device="cuda", checkpointing=leg["ckpt"], log=lambda *_: None)
        finally:
            del text_model, lm_head
            torch.cuda.empty_cache()
        r.pop("losses", None)
        r.update({"leg": leg["name"], "batch": leg["batch"], "attn": leg["attn"], "ckpt": leg["ckpt"], "gdn_forced": leg["gdn"]})
        out.append(r)
        print(f"LEG\t{leg['name']}\tseq-tok/s {r['seq_tokens_per_s']:.0f}\tpeak {r['peak_mem_gib']} GiB\tloss {r['loss_first']:.2f}->{r['loss_last']:.2f}")
    rep["legs"] = out

    d = _run_dir("lever-bench")
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


@app.function(image=image, gpu="H100:8", timeout=8 * 3600, volumes={"/cache": vol}, secrets=secrets)
def fsdp_remote(model: str, data: str, split: str, steps: int, batch: int, seq_len: int, block: int, lr: float, layers: int | None, nproc: int, grad_checkpoint: bool, mem_history: bool, ckpt: str = "", ckpt_every: int = 0, resume: bool = False, selftest: bool = False) -> dict:
    """FSDP training across the container's GPUs via torchrun (fsdp_train.py --module).

    batch is the GLOBAL rows per step; each rank takes batch/nproc rows and the complementary
    views double what each rank's forward sees. The 0.8B smoke at nproc=1 must land on
    train.py's loss trajectory (single-GPU parity); nproc=8 is the shape the 27B runs.
    mem_history turns on fsdp_train's OOM diagnostics (MEMHIST dump + MEMCENSUS).

    ckpt names a stable checkpoint dir on the volume (/cache/ckpts/<ckpt>) shared across
    windows: non-resume windows save at window end (and every ckpt_every steps); --resume
    chains the next window from the latest one. A window killed at the 8h timeout leaves
    only what the last committed save holds — hence the daemon committing the volume every
    5 min during the run (thread-safety of vol.commit() under concurrent torchrun writes is
    verified at the 0.8B gate; its failures are swallowed — the end-of-run commit below is
    the guarantee).
    """
    import subprocess
    import threading

    d = _run_dir("fsdp-train")
    argv = [
        "torchrun", f"--nproc_per_node={nproc}", "--master_port=29517", "--module", "dllm_qwen38.fsdp_train",
        "--model", model, "--data", data, "--split", split, "--steps", str(steps), "--batch", str(batch),
        "--seq-len", str(seq_len), "--block", str(block), "--lr", str(lr), "--out", d,
    ]
    if layers:
        argv += ["--layers", str(layers)]
    if grad_checkpoint:
        argv += ["--grad-checkpoint"]
    if mem_history:
        argv += ["--mem-history"]
    if ckpt:
        argv += ["--ckpt-dir", f"/cache/ckpts/{ckpt}"]
    if ckpt_every:
        argv += ["--ckpt-every", str(ckpt_every)]
    if resume:
        argv += ["--resume"]
    if selftest:
        argv += ["--resume-selftest", "--ckpt-fingerprint"]  # fingerprint probe: separates round-trip fidelity from update-path noise (2026-09-13)
    if ckpt and not resume:
        os.makedirs(f"/cache/ckpts/{ckpt}", exist_ok=True)
    stop = threading.Event()

    def _commit():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception:
                pass  # best-effort crash insurance; the end-of-run commit below is the guarantee

    committer = threading.Thread(target=_commit, daemon=True)
    committer.start()
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    stop.set()
    committer.join(timeout=310)
    # full logs to the run dir — the tail fields below are capped, and the first 27B smoke
    # died with its real traceback entirely outside the 4000-char stderr window (torchrun's
    # ChildFailedError chatter consumed it), leaving nothing to diagnose from
    for name, blob in (("stdout.log", p.stdout), ("stderr.log", p.stderr)):
        with open(f"{d}/{name}", "w") as f:
            f.write(blob)
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    rep.update({
        "exit": p.returncode,
        "wall_total_s": round(time.time() - t0, 1),
        "run_dir": d,
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-4000:],
        "ckpt": ckpt,
        "ckpt_every": ckpt_every,
        "resume": resume,
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", " | "),
    })
    if ckpt:
        rep["ckpts_present"] = sorted(e for e in os.listdir(f"/cache/ckpts/{ckpt}") if not e.endswith(".tmp"))
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


@app.local_entrypoint()
def fsdp_run(model: str = "Qwen/Qwen3.5-0.8B", data: str = "nvidia/Llama-Nemotron-Post-Training-Dataset", split: str = "chat", steps: int = 100, batch: int = 8, seq_len: int = 512, block: int = 32, lr: float = 1e-5, layers: int = 0, nproc: int = 8, grad_checkpoint: bool = False, mem_history: bool = False, ckpt: str = "", ckpt_every: int = 0, resume: bool = False, selftest: bool = False):
    rep = fsdp_remote.remote(model, data, split, steps, batch, seq_len, block, lr, layers or None, nproc, grad_checkpoint, mem_history, ckpt, ckpt_every, resume, selftest)
    print(json.dumps(rep, indent=1))


@app.function(image=image, gpu="H100", timeout=16 * 3600, volumes={"/cache": vol}, secrets=secrets)
def eval_remote(mode: str, model: str, ckpt: str | None, gstep: int | None, n: int, shots: int, threshold: float, sub_block: int | None, max_new: int, ar_batch: int, limit: int | None, max_hours: float | None) -> dict:
    """GSM8K subset eval (ar | dllm) — the measurement side of the stop literal.

    The results file (JSONL per problem, keyed by subset position) lives in
    --results-dir=/cache/evals and is resumed by position, so a 500-problem dLLM decode
    too long for one window is chunked with --limit across windows; the report only
    scores what is present. dLLM mode loads the FSDP DCP checkpoint via gsm8k_eval's
    resharding loader — no process group needed.
    """
    import subprocess

    from dllm_qwen38 import gsm8k_eval as ev

    d = _run_dir(f"eval-{mode}")
    argv = [
        "--mode", mode, "--model", model, "--n", str(n), "--shots", str(shots),
        "--threshold", str(threshold), "--max-new", str(max_new), "--block", "32",
        "--ar-batch", str(ar_batch), "--device", "cuda", "--out", d,
        "--results-dir", "/cache/evals",
    ]
    if ckpt:
        argv += ["--ckpt-dir", f"/cache/ckpts/{ckpt}"]
    if gstep:
        argv += ["--gstep", str(gstep)]
    if sub_block:
        argv += ["--sub-block", str(sub_block)]
    if limit:
        argv += ["--limit", str(limit)]
    if max_hours:
        argv += ["--max-hours", str(max_hours)]
    t0 = time.time()
    code = ev.main(argv)
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    rep.update({"exit": code, "wall_total_s": round(time.time() - t0, 1), "run_dir": d, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()})
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


@app.local_entrypoint()
def eval_run(mode: str, model: str = "Qwen/Qwen3.8-27B", ckpt: str = "", gstep: int = 0, n: int = 500, shots: int = 4, threshold: float = 0.9, sub_block: int = 0, max_new: int = 1024, ar_batch: int = 8, limit: int = 0, max_hours: float = 0.0):
    rep = eval_remote.remote(mode, model, ckpt or None, gstep or None, n, shots, threshold, sub_block or None, max_new, ar_batch, limit or None, max_hours or None)
    print(json.dumps(rep, indent=1))


@app.local_entrypoint()
def equivalence(model: str = "Qwen/Qwen3.8-27B", layers: int = 0, block: int = 32, nblocks: int = 4, dtype: str = "bfloat16", tol: float = -1.0, oracle_kernels: str = "reference"):
    rep = equivalence_remote.remote(model, layers or None, block, nblocks, dtype, None if tol < 0 else tol, oracle_kernels)
    print(json.dumps(rep, indent=1))


@app.local_entrypoint()
def train_run(model: str = "Qwen/Qwen3.5-0.8B", data: str = "nvidia/Llama-Nemotron-Post-Training-Dataset", split: str = "chat", steps: int = 200, batch: int = 4, seq_len: int = 512, block: int = 32, lr: float = 1e-5):
    rep = train_remote.remote(model, data, split, steps, batch, seq_len, block, lr)
    print(json.dumps(rep, indent=1))


@app.local_entrypoint()
def lever_bench(model: str = "Qwen/Qwen3.5-0.8B", data: str = "nvidia/Llama-Nemotron-Post-Training-Dataset", split: str = "chat", steps: int = 50, batch: int = 4, seq_len: int = 512, block: int = 32, lr: float = 1e-5):
    rep = lever_bench_remote.remote(model, data, split, steps, batch, seq_len, block, lr)
    print(json.dumps(rep, indent=1))
