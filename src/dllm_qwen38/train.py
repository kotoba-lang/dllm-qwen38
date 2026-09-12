"""Phase 2 training loop (ADR-2609112640 §4): Fast-dLLM v2 objective on the state-fork forward.

Two data sources:
  --data synthetic   a deterministic token rule; the tiny random model must drive the masked CE
                     toward zero and the reference decoder must then reproduce the rule. This is
                     the loop's own correctness test, runnable on a laptop CPU in a minute.
  --data <hf-id>     a chat dataset (default: nvidia/Llama-Nemotron-Post-Training-Dataset), rows
                     rendered with the model's chat template; prompt tokens clean and unscored.

Single process, single device. Full fine-tune of the text tower + lm_head. bf16 autocast on
CUDA. Reports loss, targets/step and sequence-tokens/s (the number the 27B cost extrapolation
in the ADR is waiting for). Nothing here is FSDP: the 27B needs that and it is not written yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

from .blockdiff import block_diffusion_forward, tiny_text_model
from .decode import generate
from .gdn import last_impl
from .objective import complementary_step_inputs, mask_token_id, masked_ce_from_hidden

# ---------------------------------------------------------------- data


def synthetic_rule(start: int, n: int, vocab: int, a: int = 7, c: int = 3) -> list[int]:
    out, x = [], start
    for _ in range(n):
        out.append(x)
        x = (x * a + c) % vocab
    return out


def synthetic_batches(B: int, L: int, vocab: int, block: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, vocab, (B,), generator=g).tolist()
        x0 = torch.tensor([synthetic_rule(s, L, vocab) for s in starts])
        yield x0, torch.full((B,), block), torch.full((B,), L)


def chat_batches(dataset: str, split: str, tokenizer, B: int, L: int, block: int, seed: int, streaming: bool = True):
    from datasets import load_dataset

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buf_x, buf_p, buf_v = [], [], []
    pass_n = 0
    while True:  # cycle: a real run's row budget (32k steps × B rows) exceeds any single split
        # (measured 2026-09-12: Nemotron SFT chat split is 39,792 rows — the 1% run would
        # StopIteration at ~step 4,975). Re-load per pass: streaming iterators are exhausted
        # after one walk. Pass 0 shuffles with the caller's seed verbatim, so ≤100-step
        # parity runs see byte-identical row order to the pre-cycling generator.
        ds = load_dataset(dataset, "SFT" if "Nemotron" in dataset else None, split=split, streaming=streaming)
        if streaming:
            ds = ds.shuffle(seed=seed + pass_n, buffer_size=2000)
        added = 0  # usable rows this pass — the degenerate-split guard below keys on it
        yielded = 0  # full batches this pass
        for row in ds:
            msgs = _messages(row)
            if msgs is None:
                continue
            prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(msgs, tokenize=True)
            if hasattr(prompt, "input_ids"):
                prompt, full = prompt["input_ids"], full["input_ids"]
            if len(prompt) >= L or len(full) <= len(prompt):
                continue
            ids = full[:L]
            valid = len(ids)
            ids = ids + [pad] * (L - valid)
            buf_x.append(ids)
            buf_p.append(len(prompt))
            buf_v.append(valid)
            added += 1
            if len(buf_x) == B:
                yield torch.tensor(buf_x), torch.tensor(buf_p), torch.tensor(buf_v)
                buf_x, buf_p, buf_v = [], [], []
                yielded += 1
        # a full pass that added nothing means the split has no usable rows at all — stop
        # instead of spinning forever re-loading it; and a pass whose usable rows exist but
        # never fill one batch would grow the buffer unboundedly on every pass — same stop
        if added == 0 or (yielded == 0 and pass_n >= 1):
            raise RuntimeError(f"{dataset}/{split}: usable rows can't fill one batch (added {added}, yielded {yielded}); cycling aborted")


def _messages(row: dict):
    """Nemotron rows (measured 2026-09-12, config SFT): `input` is a *string* holding a Python
    literal list of messages (single quotes, not JSON) and `output` is the assistant text."""
    if isinstance(row.get("messages"), list):
        return row["messages"]
    inp, out = row.get("input"), row.get("output")
    if isinstance(inp, str) and inp.lstrip().startswith("[{"):
        import ast

        try:
            inp = ast.literal_eval(inp)
        except (ValueError, SyntaxError):
            return None
    if isinstance(inp, list) and isinstance(out, str):
        return list(inp) + [{"role": "assistant", "content": out}]
    if isinstance(inp, str) and isinstance(out, str):
        return [{"role": "user", "content": inp}, {"role": "assistant", "content": out}]
    return None


# ---------------------------------------------------------------- loop


def train(
    text_model,
    lm_head,
    batches,
    *,
    block: int,
    mask_id: int,
    steps: int,
    lr: float,
    device: str,
    log_every: int = 10,
    seed: int = 0,
    grad_clip: float = 1.0,
    autocast: bool = True,
    checkpointing: bool = False,
    log=print,
) -> dict:
    # tied lm_head/embed_tokens (Qwen3.5-0.8B) would otherwise appear twice and be stepped twice
    seen, params = set(), []
    for p in list(text_model.parameters()) + list(lm_head.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator(device=device).manual_seed(seed)
    use_ac = autocast and device.startswith("cuda")
    losses, seq_tokens, t0, skipped = [], 0, time.time(), 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()  # the peak this number reports is the loop's, not the loader's
    text_model.train()
    for step in range(1, steps + 1):
        x0, prompt_len, valid_len = next(batches)
        x0, prompt_len, valid_len = x0.to(device), prompt_len.to(device), valid_len.to(device)
        x0_2, xt_2, m_2 = complementary_step_inputs(x0, prompt_len, block, mask_id, gen, valid_len)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            ht, _ = block_diffusion_forward(text_model, lm_head, x0_2, xt_2, block, output="hidden", checkpointing=checkpointing)
            loss, n = masked_ce_from_hidden(ht, lm_head, x0_2, m_2)
        if n == 0:
            skipped += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        losses.append(float(loss))
        seq_tokens += x0.numel()
        if step % log_every == 0 or step == steps:
            el = time.time() - t0
            log(f"step {step}\tloss {float(loss):.4f}\ttargets {n}\tseq-tok/s {seq_tokens / el:.0f}\telapsed {el:.0f}s")
    text_model.eval()
    el = time.time() - t0
    rep = {
        "steps": steps,
        "skipped_steps": skipped,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "seq_tokens": seq_tokens,
        "seq_tokens_per_s": seq_tokens / el if el else 0.0,
        "seconds": el,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.startswith("cuda") else None,
        "losses": losses,
    }
    # name the paths the loop actually ran (gdn.py sets these at call time; see gdn.set_impl)
    g_last, g_reason = last_impl()
    rep.update({"checkpointing": checkpointing, "gdn_impl": g_last, "gdn_reason": g_reason})
    return rep


def synthetic_eval(text_model, lm_head, *, block: int, mask_id: int, vocab: int, seed: int, threshold: float, n_prompts: int = 8, n_blocks: int = 2):
    """Decode with the reference decoder and score against the rule: fraction of generated
    tokens that equal the rule's continuation. Also reports tokens/forward (the parallelism)."""
    g = torch.Generator().manual_seed(seed + 12345)
    correct, total, forwards, gen_tokens = 0, 0, 0, 0
    for _ in range(n_prompts):
        start = int(torch.randint(0, vocab, (1,), generator=g))
        truth = synthetic_rule(start, block * (1 + n_blocks), vocab)
        tr = generate(text_model, lm_head, truth[:block], n_blocks=n_blocks, block=block, mask_id=mask_id, threshold=threshold)
        out = tr.tokens[block:]
        correct += sum(int(a == b) for a, b in zip(out, truth[block:]))
        total += len(out)
        forwards += tr.forwards
        gen_tokens += tr.generated
    return {"rule_accuracy": correct / total, "generated": total, "forwards": forwards, "tokens_per_forward": gen_tokens / forwards}


# ---------------------------------------------------------------- cli


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="HF id; omit for the tiny random model")
    p.add_argument("--data", default="synthetic", help="'synthetic' or an HF dataset id")
    p.add_argument("--split", default="chat")
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4, help="sequences per step; the two complementary views double the rows the forward sees")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn", default=None, help="attention backend of the loaded checkpoint (default: sdpa on cuda, eager on cpu)")
    p.add_argument("--gdn", default="auto", choices=["auto", "torch", "fla"], help="GDN core impl for the candidate path (auto = fla when importable and on CUDA)")
    p.add_argument("--grad-checkpoint", action="store_true", help="wrap each layer in torch.utils.checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", default=None, help="directory for checkpoint + report.json")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (macOS: grouped conv1d backward oversubscribes; 1 was 13x faster than 8)")
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)

    if a.seq_len % a.block:
        print("COULD-NOT-MEASURE seq-len must be a multiple of block", file=sys.stderr)
        return 2

    if a.model:
        from .equivalence import load_checkpoint
        from transformers import AutoTokenizer

        dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
        attn = a.attn or ("sdpa" if a.device.startswith("cuda") else "eager")
        text_model, lm_head, _ = load_checkpoint(a.model, dtype, a.device, attn_implementation=attn)
        tok = AutoTokenizer.from_pretrained(a.model)
        mask_id = mask_token_id(text_model, tok)
        vocab = text_model.embed_tokens.weight.shape[0]
        text_model.gradient_checkpointing = False
    else:
        attn = "eager"  # tiny_text_model hardcodes eager; the flag records what ran, not what was asked
        text_model, lm_head = tiny_text_model(seed=a.seed, layers=8, vocab=256)
        text_model, lm_head = text_model.to(a.device), lm_head.to(a.device)
        tok, vocab = None, 256
        mask_id = vocab - 1

    from . import gdn as _gdn

    _gdn.set_impl(a.gdn if a.gdn != "auto" else None)

    if a.data == "synthetic":
        batches = synthetic_batches(a.batch, a.seq_len, vocab if tok is None else vocab - 2, a.block, a.seed)
    else:
        if tok is None:
            print("COULD-NOT-MEASURE a chat dataset needs --model (tokenizer + template)", file=sys.stderr)
            return 2
        batches = chat_batches(a.data, a.split, tok, a.batch, a.seq_len, a.block, a.seed)

    rep = train(text_model, lm_head, batches, block=a.block, mask_id=mask_id, steps=a.steps, lr=a.lr, device=a.device, seed=a.seed, checkpointing=a.grad_checkpoint)
    rep.update({"model": a.model or "tiny-random", "data": a.data, "block": a.block, "seq_len": a.seq_len, "batch": a.batch, "lr": a.lr, "device": a.device, "dtype": a.dtype if a.model else "float32", "attn": attn})
    if a.data == "synthetic" and not a.no_eval:
        rep["eval"] = synthetic_eval(text_model, lm_head, block=a.block, mask_id=mask_id, vocab=vocab if tok is None else vocab - 2, seed=a.seed, threshold=a.threshold)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        from safetensors.torch import save_file

        state = {f"text_model.{k}": v.detach().cpu().contiguous() for k, v in text_model.state_dict().items()}
        state.update({f"lm_head.{k}": v.detach().cpu().contiguous() for k, v in lm_head.state_dict().items()})
        save_file(state, os.path.join(a.out, "model.safetensors"))
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
    print(f"TRAIN\tsteps {rep['steps']}\tskipped {rep['skipped_steps']}\tloss {rep['loss_first']:.4f} -> {rep['loss_last']:.4f}\tseq-tok/s {rep['seq_tokens_per_s']:.0f}\tpeak_mem_gib {rep['peak_mem_gib']}\t{rep['seconds']:.0f}s")
    if "eval" in rep:
        e = rep["eval"]
        print(f"EVAL\trule_accuracy {e['rule_accuracy']:.3f}\tgenerated {e['generated']}\ttokens_per_forward {e['tokens_per_forward']:.2f}")
    return 0 if (rep["loss_last"] is not None and math.isfinite(rep["loss_last"])) else 1


if __name__ == "__main__":
    sys.exit(main())
