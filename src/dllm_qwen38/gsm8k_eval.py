"""GSM8K eval harness — the measurement side of the stop literal (ADR-2609112640).

Stop literal: "0.3B tokens of block-diffusion training buys a GSM8K subset score within
−5pt of AR Qwen3.8-27B, else don't continue." Both numbers must come from the SAME
subset, the SAME prompts, and the SAME scorer — the only thing allowed to differ is the
decode. So this module owns: a fixed seeded subset, one few-shot prompt builder used by
both paths, one extraction+scorer, and the two decode paths (AR baseline via HF
`.generate`; block-diffusion via the golden `decode.generate` on a DCP-restored
checkpoint).

Design decisions (owner: harness first, 2026-09-13; the 1% run is scored the hour its
window ends):
- Subset: 500 of 1319 test problems, seed 20260913; a subset_fingerprint (sha256 over
  subset questions + demo questions) is written into every report so AR and dLLM reports
  provably measured the same thing.
- Prompt: chat-template user message (one instruction + few-shot demos rendered as plain
  text inside it), `add_generation_prompt=True` and `enable_thinking=False` — the
  assistant turn starts after a pre-filled empty thinking block, so the 768-token budget
  goes to the answer, not a reasoning trace. Demos come from the TRAIN split only, seed
  20260914: a test question can never appear in its own demos.
- Scoring: last `#### <number>` in the generation; fallback to the last number (method
  recorded so fallback-mode acc can be read separately). Both sides run the text through
  a post-`` cut first (Qwen3 thinking mode), so a stray trace can't become a number.
  Gold unparseable → problem EXCLUDED from the denominator and counted (skipped ≠
  wrong: 検査の 4 問).
- dLLM decode: golden decoder unbatched, threshold 0.9, sub_block None (Fast-dLLM v2's
  block-wise confidence decode; `--sub-block 8` reproduces the paper's left-to-right
  sub-block order at ~3× the forwards). n_blocks is derived from max_new so both paths
  allow the same answer budget.
- Checkpoint restore: the training checkpoint is DCP-sharded under the `_Root` wrapper
  (`text_model.*` / `lm_head.*`, plus the extra `.layer.` hop the LayerStep replacement
  adds — stripped here). Load is resharding: plain full-tensor requests against the
  saved DTensor chunks, `no_dist=True`, so one H100 reads all 8 ranks' shards.
- Incremental runs: one JSONL line per problem appended to `--results-dir`, keyed by
  problem index; a rerun (new window, `--limit` chunking) skips already-done problems.
  AR batches; dLLM does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from decimal import Decimal, InvalidOperation

SUBSET_SEED = 20260913  # fixed forever: AR and dLLM must see identical problems
DEMO_SEED = 20260914  # demos drawn from the TRAIN split; a test question can't be its own demo

INSTRUCTION = (
    "Solve the math word problem. Think step by step, then give the final numeric answer "
    "on its own last line as '#### <answer>'.\n\n"
)

# ---------------------------------------------------------------- subset


def subset_indices(n_rows: int, n: int, seed: int = SUBSET_SEED) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), min(n, n_rows)))


def subset_rows(rows, n: int, seed: int = SUBSET_SEED) -> list:
    return [rows[i] for i in subset_indices(len(rows), n, seed)]


def subset_fingerprint(items, demos) -> str:
    """Binds seed+n+exact problems+exact demos; AR and dLLM reports must agree on this."""
    h = hashlib.sha256()
    for it in items:
        h.update(it["question"].strip().encode())
        h.update(b"\x00")
    for q, _ in demos:
        h.update(q.strip().encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


# ------------------------------------------------------- gold / extraction / scoring


def _to_decimal(s: str) -> Decimal | None:
    s = s.strip().strip("$").strip().replace(",", "")
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def gold_value(gsm8k_answer: str) -> Decimal | None:
    """GSM8K gold field: CoT lines then '#### <number>'."""
    if "####" not in gsm8k_answer:
        return None
    return _to_decimal(gsm8k_answer.rsplit("####", 1)[-1])


_ANS_RE = re.compile(r"####\s*\$?\s*([\-0-9.,]+)")
_NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def _post_think(text: str) -> str:
    """Text after the last ``</think>`` only — a reasoning-trace number must never score.

    A truncated trace (opened ``<think>``, never closed) scores nothing: the trace is not
    an answer, and counting its last number would flatter the model."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return ""
    return text


def extract_pred(text: str) -> tuple[str, str]:
    """(decimal-string or "", method) — method: '####' | 'last_number' | 'none'."""
    text = _post_think(text)
    matches = _ANS_RE.findall(text)
    if matches:
        return matches[-1], "####"
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1], "last_number"
    return "", "none"


def score(pred_text: str, gsm8k_answer: str) -> tuple[bool | None, str, str]:
    """(ok, pred, method); ok None = gold unparseable → excluded from the denominator."""
    g = gold_value(gsm8k_answer)
    p_raw, method = extract_pred(pred_text)
    p = _to_decimal(p_raw)
    if g is None:
        return None, p_raw, method
    ok = p is not None and p == g
    return ok, p_raw, method


def accuracy(results: list) -> dict:
    """results: list of (ok: bool|None, method: str). Counts skipped separately."""
    n_ok = sum(1 for ok, _ in results if ok is True)
    n_scored = sum(1 for ok, _ in results if ok is not None)
    skipped = sum(1 for ok, _ in results if ok is None)
    by_method: dict[str, int] = {}
    for ok, method in results:
        if ok is None:
            continue
        by_method[method] = by_method.get(method, 0) + 1
    return {"acc": round(n_ok / n_scored, 4) if n_scored else None, "n": n_scored, "skipped": skipped, "by_method": by_method}


# ---------------------------------------------------------------- prompts


def fewshot_demos(train_rows, shots: int, seed: int = DEMO_SEED) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(train_rows[i]["question"], train_rows[i]["answer"]) for i in rng.sample(range(len(train_rows)), shots)]


def fewshot_block(demos, item) -> str:
    """Instruction-free demo block + the live question. Pure (no tokenizer) — CPU tests pin it."""
    parts = []
    for q, a in demos:
        cot, final = a.rsplit("####", 1) if "####" in a else (a.strip(), "")
        parts.append(f"Question: {q}\nAnswer: {cot.strip()} #### {final.strip()}".rstrip())
    parts.append(f"Question: {item['question']}\nAnswer:")
    return "\n\n".join(parts)


def user_message(item, demos) -> str:
    return INSTRUCTION + fewshot_block(demos, item)


def prompt_ids(tok, item, demos) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": user_message(item, demos)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(ids, list):
        ids = list(ids)
    return ids


# ------------------------------------------------------------ decode-path helpers


def blocks_for(max_new: int, prompt_len: int, block: int) -> int:
    """Blocks whose generated capacity (room + (nb-1)*block) covers max_new.

    decode.generate emits `block - prompt_len % block` tokens in the first block (it
    resumes mid-block after the prompt tail), then `block` per block.
    """
    room = block - prompt_len % block
    if max_new <= room:
        return 1
    return 1 + (max_new - room + block - 1) // block


def cut_at_eos(tokens: list[int], eos_ids: set[int]) -> list[int]:
    """Cut at the FIRST eos — a block that hit eos keeps denoising to the block end."""
    for i, t in enumerate(tokens):
        if t in eos_ids:
            return tokens[:i]
    return tokens


# ---------------------------------------------------------------- loaders


def load_gsm8k() -> tuple[list, list]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    return list(ds["train"]), list(ds["test"])


def load_ar_model(model_id: str, device: str):
    """The original HF checkpoint, as-is (the AR side of the comparison)."""
    import torch
    from transformers import AutoModelForImageTextToText

    m = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="bfloat16", attn_implementation="sdpa"
    ).to(device)
    m.eval()
    return m


def plain_fqn(dcp_fqn: str) -> str | None:
    """Checkpoint FQN (under the _Root wrapper) → plain HF-model FQN; None = unexpected key.

    `text_model.layers.7.layer.mlp.x` → `layers.7.mlp.x` (the LayerStep `.layer.` hop);
    `text_model.X` → `X`; `lm_head.*` passes through for the separate lm_head module.
    """
    if dcp_fqn.startswith("text_model."):
        rest = dcp_fqn[len("text_model."):]
        m = re.match(r"^(layers\.\d+)\.layer\.(.+)$", rest)
        return f"{m.group(1)}.{m.group(2)}" if m else rest
    if dcp_fqn.startswith("lm_head."):
        return dcp_fqn
    return None


def dcp_model_tensors(ckpt_path: str) -> tuple[dict, int, int]:
    """Read the 'model' half of a DCP checkpoint as plain CPU tensors (+ step, seq_tokens).

    Resharding load: the saved entries are per-rank DTensor shards; requesting the global
    shape as a plain tensor makes the planner read every overlapping chunk. `no_dist` so a
    single-GPU process needs no process group.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load as dcp_load

    reader = FileSystemReader(ckpt_path)
    md = reader.read_metadata()
    model, step, seq_tokens = {}, torch.zeros((), dtype=torch.int64), torch.zeros((), dtype=torch.int64)
    for key, meta in md.state_dict_metadata.items():
        if not key.startswith("model."):
            continue
        fqn = key[len("model."):]
        if not hasattr(meta, "size"):
            raise ValueError(f"{ckpt_path}: non-tensor entry under 'model': {key}")
        model[fqn] = torch.empty(tuple(meta.size), dtype=meta.properties.dtype, device="cpu")
    dcp_load({"model": model, "step": step, "seq_tokens": seq_tokens}, storage_reader=reader, no_dist=True)
    return model, int(step.item()), int(seq_tokens.item())


def load_dcp_into_model(model_id: str, ckpt_path: str, device: str, layers: int | None = None, attn: str = "sdpa"):
    """Build the plain model from HF (reuses equivalence.load_checkpoint, whose 27B path is
    already measured) and overwrite every weight from the checkpoint. Missing/unexpected
    keys fail loudly — a partially restored model would otherwise score silently wrong.
    """
    import torch

    from .equivalence import load_checkpoint

    text_model, lm_head, spare = load_checkpoint(
        model_id, torch.bfloat16, device, layers=layers, attn_implementation=attn
    )
    tensors, step, seq_tokens = dcp_model_tensors(ckpt_path)
    restore_into(text_model, lm_head, tensors, ctx=ckpt_path)
    del tensors
    return text_model, lm_head, step, seq_tokens, spare


def restore_into(text_model, lm_head, tensors: dict, ctx: str = "ckpt") -> None:
    """Overwrite text_model/lm_head from the DCP 'model' tensor dict; raise on ANY mismatch.

    A partial restore scores silently wrong, so the loud check is ours, not torch's quiet
    strict=False partial: `model:` = restore incomplete, `model?`/`head?` = ckpt key the
    model doesn't have, `ckpt?` = key plain_fqn refused (neither text_model.* nor lm_head.*).
    """
    sd_text, sd_head, unexpected = {}, {}, []
    for fqn, t in tensors.items():
        p = plain_fqn(fqn)
        if p is None:
            unexpected.append(fqn)
        elif p.startswith("lm_head."):
            sd_head[p[len("lm_head."):]] = t
        else:
            sd_text[p] = t
    rm, um = text_model.load_state_dict(sd_text, strict=False)
    rh, uh = lm_head.load_state_dict(sd_head, strict=False)
    bad = (["model:" + k for k in rm] + ["head:" + k for k in rh]
           + ["model?" + k for k in um] + ["head?" + k for k in uh]
           + ["ckpt?" + k for k in unexpected])
    if bad:
        raise ValueError(f"{ctx}: restore incomplete ({len(bad)} problems, first 10: {bad[:10]})")


# ---------------------------------------------------------------- runners


def run_ar_batch(m, tok, ids_list: list[list[int]], max_new: int, device: str) -> list[str]:
    """Greedy AR batch, left-padded. Attention masks are built from lengths (never sniffed
    from token values — prompt tokens include <|im_end|> and pad_token_id collides with
    <|endoftext|>, so value-based masks are a trap)."""
    import torch

    maxlen = max(len(x) for x in ids_list)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(ids_list), maxlen), pad, dtype=torch.long, device=device)
    attn = torch.zeros((len(ids_list), maxlen), dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        input_ids[i, maxlen - len(ids):] = torch.tensor(ids, device=device)
        attn[i, maxlen - len(ids):] = 1
    out = m.generate(
        input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new,
        do_sample=False, num_beams=1, pad_token_id=pad,
    )
    texts = []
    for i, ids in enumerate(ids_list):
        gen = out[i, input_ids.shape[1]:]
        texts.append(tok.decode(gen, skip_special_tokens=True))
    return texts


def run_one_dllm(text_model, lm_head, tok, ids: list[int], *, block: int, threshold: float, sub_block: int | None, max_new: int, mask_id: int, eos_id: int | None):
    """One problem through the golden decoder, cut at eos, detokenized."""
    import time as _t

    from . import decode

    t0 = _t.time()
    nb = blocks_for(max_new, len(ids), block)
    tr = decode.generate(
        text_model, lm_head, ids, n_blocks=nb, block=block, mask_id=mask_id,
        threshold=threshold, sub_block=sub_block, eos_id=eos_id, temperature=0.0,
    )
    gen = cut_at_eos(tr.tokens[len(ids):], {eos_id} if eos_id else set())
    text = tok.decode(gen, skip_special_tokens=True)
    meta = {"forwards": tr.forwards, "generated": tr.generated, "n_blocks": nb, "steps_per_block": tr.steps_per_block, "wall_s": round(_t.time() - t0, 1)}
    return text, meta


# ---------------------------------------------------------------- main


def latest_gstep(ckpt_dir: str) -> int | None:
    from .fsdp_train import _latest_checkpoint

    return _latest_checkpoint(ckpt_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gsm8k_eval")
    p.add_argument("--mode", required=True, choices=["ar", "dllm"])
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--ckpt-dir", default=None, help="training checkpoint dir on the volume; dLLM mode reads step-<g> from it")
    p.add_argument("--gstep", type=int, default=None, help="checkpoint step (default: latest in --ckpt-dir)")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--sub-block", type=int, default=None)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--ar-batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="process at most this many REMAINING problems (0 = all); chunked windows resume from results")
    p.add_argument("--max-hours", type=float, default=0.0, help="soft stop (graceful report) after this many hours; 0 = no cap")
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-dir", default=None, help="stable per-mode results JSONL (default: --out); Modal passes /cache/evals so reruns resume")
    p.add_argument("--out", default=".")
    args = p.parse_args(argv)

    import torch  # noqa: F401  (transformers pulls it anyway; explicit for clarity)
    from transformers import AutoTokenizer

    from .objective import mask_token_id

    os.makedirs(args.out, exist_ok=True)
    results_dir = args.results_dir or args.out
    os.makedirs(results_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_rows, test_rows = load_gsm8k()
    demos = fewshot_demos(train_rows, args.shots)
    items = subset_rows(test_rows, args.n)
    fp = subset_fingerprint(items, demos)

    results_path = os.path.join(results_dir, f"{args.mode}-n{args.n}-fp{fp}.jsonl")
    done: dict[int, dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done[r["idx"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass  # a torn last line (window killed mid-append) is skippable, not fatal
    # key results by subset position, not by dataset id — the fingerprint pins the subset
    pending = [(i, it) for i, it in enumerate(items) if i not in done]
    if args.limit:
        pending = pending[: args.limit]

    rep = {
        "mode": args.mode, "model": args.model, "n_subset": len(items), "shots": args.shots,
        "subset_fingerprint": fp, "results_file": results_path,
        "done_before": len(done), "to_run": len(pending),
        "params": {k: getattr(args, k) for k in ("threshold", "sub_block", "max_new", "block", "ar_batch", "limit", "max_hours")},
    }

    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None

    if args.mode == "ar":
        m = load_ar_model(args.model, args.device)
        tok.padding_side = "left"
        bs = max(1, args.ar_batch)
        with torch.inference_mode():
            for b0 in range(0, len(pending), bs):
                chunk = pending[b0 : b0 + bs]
                if deadline is not None and time.time() > deadline and b0 > 0:
                    break
                ids_list = [prompt_ids(tok, it, demos) for _, it in chunk]
                tb = time.time()
                texts = run_ar_batch(m, tok, ids_list, args.max_new, args.device)
                wall = (time.time() - tb) / len(chunk)
                for (idx, it), ids, text in zip(chunk, ids_list, texts):
                    ok, pred, method = score(text, it["answer"])
                    r = {"idx": idx, "question": it["question"], "prompt_len": len(ids), "pred": pred, "method": method, "ok": ok, "wall_s": round(wall, 2), "max_new": args.max_new}
                    _append_result(results_path, r)
                    done[idx] = r
                if (b0 // bs) % 5 == 0 or b0 + bs >= len(pending):
                    _print_progress("AR", done, items, t0)
    else:
        if not args.ckpt_dir:
            raise SystemExit("--mode dllm needs --ckpt-dir")
        gstep = args.gstep if args.gstep is not None else latest_gstep(args.ckpt_dir)
        if gstep is None:
            raise SystemExit(f"no step-* checkpoint in {args.ckpt_dir}")
        ckpt_path = os.path.join(args.ckpt_dir, f"step-{gstep:08d}")
        rep["ckpt"] = ckpt_path
        text_model, lm_head, dcp_step, seq_tokens, spare = load_dcp_into_model(args.model, ckpt_path, args.device)
        rep["dcp_step"], rep["dcp_seq_tokens"], rep["spare_rows"] = dcp_step, seq_tokens, spare.get("spare_rows")
        mid = mask_token_id(text_model, tok)
        eos_id = tok.eos_token_id
        with torch.inference_mode():
            for i, (idx, it) in enumerate(pending):
                if deadline is not None and time.time() > deadline and i > 0:
                    break
                ids = prompt_ids(tok, it, demos)
                text, meta = run_one_dllm(text_model, lm_head, tok, ids, block=args.block, threshold=args.threshold, sub_block=args.sub_block, max_new=args.max_new, mask_id=mid, eos_id=eos_id)
                ok, pred, method = score(text, it["answer"])
                r = {"idx": idx, "question": it["question"], "prompt_len": len(ids), "pred": pred, "method": method, "ok": ok, "wall_s": meta.pop("wall_s"), "trace": meta}
                _append_result(results_path, r)
                done[idx] = r
                if i % args.report_every == 0 or i + 1 == len(pending):
                    _print_progress("DLLM", done, items, t0)

    scored = accuracy([(r["ok"], r["method"]) for r in done.values()])
    rep.update({"done": len(done), "subset_scored": scored, "wall_s": round(time.time() - t0, 1), "gpu": _gpu_line()})
    # full texts live in the JSONL; the report keeps aggregates only (texts are big)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"REPORT\t{json.dumps(rep, default=str)}", flush=True)
    return 0


def _append_result(results_path: str, r: dict) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(r) + "\n")


def _print_progress(tag: str, done: dict, items: list, t0: float) -> None:
    acc = accuracy([(r["ok"], r["method"]) for r in done.values()])
    print(f"PROGRESS\t{tag}\t{len(done)}/{len(items)}\tacc={acc['acc']}\tn={acc['n']}\tskipped={acc['skipped']}\telapsed_s={time.time() - t0:.0f}", flush=True)


def _gpu_line() -> str:
    import subprocess

    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
