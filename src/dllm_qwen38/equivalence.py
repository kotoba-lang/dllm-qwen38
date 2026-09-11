"""ADR-2609112640 §3a — the visibility equivalence test.

Answers, with numbers and not booleans, four questions about `block_diffusion_forward`:

  1. does the noised stream equal the oracle (unmodified HF model per block, block-causal mask)?
  2. does the clean stream equal the unmodified HF model under the block-causal mask?
  3. perturbation: does x_t block k ignore x_t block k-1 and x_0 block k, and react to x_0 block k-1?
  4. does every deliberate fault (concat-causal-gdn / mask-leak / no-conv-prefix) turn check 1 red?

Exit code 0 = every check passed AND every fault was detected; 1 = a check failed or a fault went
undetected; 2 = could not measure (model did not load, shapes impossible). A run that measures
nothing never exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from .blockdiff import BreakMode, block_diffusion_forward, reference_block_logits, reference_clean_logits, tiny_text_model

FAULTS = ("concat-causal-gdn", "mask-leak", "no-conv-prefix")


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def oracle_gdn_kernel() -> str:
    """Which implementation the HF oracle's chunked delta rule resolved to at import time."""
    fla = sys.modules.get("fla", None)
    return "fla" if fla is not None else "torch-reference"


def load_checkpoint(model_id: str, dtype: torch.dtype, device: str, layers: int | None = None):
    """Text tower + lm_head of a Qwen3.5/3.8 checkpoint (vision tower is loaded but unused).

    `layers` truncates the stack *before* the dtype cast: the 27B is loaded in bf16 (55 GB), the
    tail layers are dropped, and only the kept prefix is cast — fp32 on all 64 layers would be
    108 GB and does not fit one H100. The returned text_model has config.num_hidden_layers set
    to the truncated depth so the HF oracle and the candidate see the same stack.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id)
    load_dtype = torch.bfloat16 if (layers and dtype == torch.float32) else dtype
    kw = dict(dtype=load_dtype, attn_implementation="eager", device_map=device)
    if getattr(cfg, "text_config", None) is not None:
        m = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        text_model, lm_head = m.model.language_model, m.lm_head
        if hasattr(m.model, "visual"):
            m.model.visual = None
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        text_model, lm_head = m.model, m.lm_head
    if layers:
        del text_model.layers[layers:]
        text_model.config.num_hidden_layers = layers
        text_model.config.layer_types = list(text_model.config.layer_types[:layers])
    if load_dtype != dtype:
        text_model.to(dtype)
        lm_head.to(dtype)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab_rows = text_model.embed_tokens.weight.shape[0]
    spare = {"embedding_rows": vocab_rows, "tokenizer_len": len(tok), "spare_rows": vocab_rows - len(tok)}
    return text_model, lm_head, spare


def run(
    text_model,
    lm_head,
    *,
    block: int,
    nblocks: int,
    layers: int | None,
    tol: float,
    change_floor: float,
    seed: int,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    L = block * nblocks
    vocab = text_model.embed_tokens.weight.shape[0]
    x0 = torch.randint(0, vocab, (1, L), device=device)
    xt = x0.clone()
    # noise: mask roughly half of every block with a token id that exists (visibility does not
    # depend on which id plays [MASK])
    mask_id = vocab - 1
    noise = torch.rand(1, L, device=device) < 0.5
    xt[noise] = mask_id

    checks: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        lt, l0 = block_diffusion_forward(text_model, lm_head, x0, xt, block, layers=layers)
        ref_t = reference_block_logits(text_model, lm_head, x0, xt, block, layers=layers)
        ref_0 = reference_clean_logits(text_model, lm_head, x0, block, layers=layers)
    d = _maxdiff(lt, ref_t)
    checks.append({"name": "noised-stream-vs-oracle", "max_abs_diff": d, "tol": tol, "pass": d <= tol})
    d = _maxdiff(l0, ref_0)
    checks.append({"name": "clean-stream-vs-oracle", "max_abs_diff": d, "tol": tol, "pass": d <= tol})

    # perturbations, on the candidate alone, for every block k >= 1
    if nblocks >= 2:
        worst_ignore_xt, worst_ignore_x0k, weakest_react = 0.0, 0.0, float("inf")
        for k in range(1, nblocks):
            sl = slice(k * block, (k + 1) * block)
            prev = slice((k - 1) * block, k * block)

            xt2 = xt.clone()
            xt2[:, prev] = torch.randint(0, vocab, (1, block), device=device)
            with torch.no_grad():
                lt2, _ = block_diffusion_forward(text_model, lm_head, x0, xt2, block, layers=layers)
            worst_ignore_xt = max(worst_ignore_xt, _maxdiff(lt[:, sl], lt2[:, sl]))

            x02 = x0.clone()
            x02[:, sl] = torch.randint(0, vocab, (1, block), device=device)
            with torch.no_grad():
                lt3, _ = block_diffusion_forward(text_model, lm_head, x02, xt, block, layers=layers)
            worst_ignore_x0k = max(worst_ignore_x0k, _maxdiff(lt[:, sl], lt3[:, sl]))

            x03 = x0.clone()
            x03[:, prev] = torch.randint(0, vocab, (1, block), device=device)
            with torch.no_grad():
                lt4, _ = block_diffusion_forward(text_model, lm_head, x03, xt, block, layers=layers)
            weakest_react = min(weakest_react, _maxdiff(lt[:, sl], lt4[:, sl]))
        checks.append({"name": "xt-block-k-ignores-xt-block-k-1", "max_abs_diff": worst_ignore_xt, "tol": tol, "pass": worst_ignore_xt <= tol})
        checks.append({"name": "xt-block-k-ignores-x0-block-k", "max_abs_diff": worst_ignore_x0k, "tol": tol, "pass": worst_ignore_x0k <= tol})
        checks.append({"name": "xt-block-k-reacts-to-x0-block-k-1", "min_abs_diff": weakest_react, "floor": change_floor, "pass": weakest_react >= change_floor})

    # fault injection: each must be caught by check 1
    faults = []
    for name in FAULTS:
        with torch.no_grad():
            lf, _ = block_diffusion_forward(text_model, lm_head, x0, xt, block, fault=BreakMode(name), layers=layers)
        d = _maxdiff(lf, ref_t)
        faults.append({"fault": name, "max_abs_diff": d, "tol": tol, "detected": d > tol})

    return {
        "block": block,
        "nblocks": nblocks,
        "seq_len": L,
        "layers": layers if layers is not None else int(text_model.config.num_hidden_layers),
        "layer_types": list(text_model.config.layer_types[: layers or None]),
        "dtype": str(text_model.embed_tokens.weight.dtype),
        "device": device,
        "seconds": round(time.time() - t0, 2),
        "checks": checks,
        "faults": faults,
        "checks_passed": sum(c["pass"] for c in checks),
        "checks_total": len(checks),
        "faults_detected": sum(f["detected"] for f in faults),
        "faults_total": len(faults),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="HF id (e.g. Qwen/Qwen3.5-0.8B, Qwen/Qwen3.8-27B); omit for the tiny random model")
    p.add_argument("--layers", type=int, default=None, help="use only the first N layers (27B first-layers run)")
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--nblocks", type=int, default=4)
    p.add_argument("--tol", type=float, default=None, help="max |Δlogit| accepted (default: fp32 1e-3, bf16 0.25)")
    p.add_argument("--change-floor", type=float, default=None, help="min |Δlogit| a real dependency must produce (default 10×tol)")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write the report here as JSON")
    a = p.parse_args(argv)

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    tol = a.tol if a.tol is not None else (1e-3 if dtype == torch.float32 else 0.25)
    floor = a.change_floor if a.change_floor is not None else 10 * tol

    try:
        if a.model:
            text_model, lm_head, spare = load_checkpoint(a.model, dtype, a.device, a.layers)
            a.layers = None  # already truncated at load
        else:
            text_model, lm_head = tiny_text_model(seed=a.seed)
            text_model, lm_head = text_model.to(a.device), lm_head.to(a.device)
            spare = None
    except Exception as e:  # noqa: BLE001 — could not measure is its own exit code
        print(f"COULD-NOT-MEASURE model load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.layers is not None and a.layers > text_model.config.num_hidden_layers:
        print("COULD-NOT-MEASURE --layers exceeds the model", file=sys.stderr)
        return 2
    if a.nblocks < 1:
        print("COULD-NOT-MEASURE nblocks < 1", file=sys.stderr)
        return 2

    rep = run(text_model, lm_head, block=a.block, nblocks=a.nblocks, layers=a.layers, tol=tol, change_floor=floor, seed=a.seed, device=a.device)
    rep["model"] = a.model or "tiny-random"
    rep["oracle_gdn_kernel"] = oracle_gdn_kernel()
    if spare:
        rep["vocab"] = spare
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)

    for c in rep["checks"]:
        val = c.get("max_abs_diff", c.get("min_abs_diff"))
        print(f"{'PASS' if c['pass'] else 'FAIL'}\t{c['name']}\t{val:.3e}")
    for f in rep["faults"]:
        print(f"{'DETECTED' if f['detected'] else 'UNDETECTED'}\tfault:{f['fault']}\t{f['max_abs_diff']:.3e}")
    print(f"CHECKS\t{rep['checks_passed']}/{rep['checks_total']}\tFAULTS-DETECTED\t{rep['faults_detected']}/{rep['faults_total']}\tmodel={rep['model']} layers={rep['layers']} L={rep['seq_len']} dtype={rep['dtype']} oracle-gdn={rep['oracle_gdn_kernel']} {rep['seconds']}s")
    if spare:
        print(f"VOCAB\tembedding_rows={spare['embedding_rows']}\ttokenizer_len={spare['tokenizer_len']}\tspare_rows={spare['spare_rows']}")
    ok = rep["checks_passed"] == rep["checks_total"] and rep["faults_detected"] == rep["faults_total"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
