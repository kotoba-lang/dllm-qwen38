"""FSDP training (H100×8) for the 27B on the state-fork forward — ADR-2609112640 §4.

The structural constraint the fork imposes: `block_diffusion_forward` never calls a layer
itself. It reaches straight into each layer's submodules (`layer.input_layernorm`,
`layer.linear_attn` via `gdn_mixer_forward`, `layer.self_attn`, `layer.mlp`) and, for DeltaNet
layers, runs two mixer calls with shape-changing batch rows (B*nb rows for the noised blocks,
per-block states [B, nb, H, K, V] and conv prefixes read off the clean pass). FSDP-wrapping
the HF layer itself would all-gather only when the *layer* is called — which never happens —
so every submodule call would read a **local shard**: silent wrong answers, no error.

The fix here is one reference `nn.Module` (`LayerStep`) holding the layer-step body exactly as
`blockdiff.py::_layer_step` has it, wrapped as the FSDP unit with `reshard_after_forward=False`
so the gathered weights stay resident between the two mixer calls and the two mlp calls (one
gather per layer per step, not four). `embed_tokens`, `norm`, and `lm_head` stay replicated
(unsharded): they sit outside every FSDP unit by construction, and at 27B that is ~2.5 GB bf16
per rank against a 52 GB sharded tower.

Loss aggregation must survive a rank with zero local targets (DDP sync): each rank backwards
its local sum CE divided by the *global* target count (all-reduced), with a graph-connected
zero when it has no targets, so FSDP's gradient averaging lands on the global mean CE; if the
global count is zero all ranks skip the step together. At world_size 1 this reduces exactly to
`train.py`'s step — that parity (0.8B, same seed, same data) is the correctness gate before
the 27B is trusted with this path.

Run (Modal, gpu="H100:8"): torchrun --nproc_per_node=8 --module dllm_qwen38.fsdp_train ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from .blockdiff import FULL, LINEAR, _rotary, complementary_mask
from .gdn import gdn_mixer_forward, last_impl
from .objective import complementary_step_inputs, mask_token_id
from .train import chat_batches


class LayerStep(torch.nn.Module):
    """One transformer layer for both streams — the FSDP unit.

    Body is the LINEAR/FULL arms of `blockdiff.py::_layer_step` verbatim; `pos` and `mask` ride
    the forward call (per-step data, not parameters). A late-binding closure over the loop
    variable is the severed-gradient failure measured 2026-09-12 — the checkpoint wrapper (and
    everything else) must bind the step at call time via partial, never a lambda.
    """

    def __init__(self, layer, kind: str):
        super().__init__()
        self.layer = layer
        self.kind = kind

    def forward(self, h0, ht, pos, mask, block: int):
        r0, rt = h0, ht
        layer = self.layer
        B, L = h0.shape[0], h0.shape[1]
        n0, nt = layer.input_layernorm(h0), layer.input_layernorm(ht)
        if self.kind == LINEAR:
            m = layer.linear_attn
            # pass 1: clean stream, block-causal by construction; keep S_k and the conv tail
            o0, states0, raw0 = gdn_mixer_forward(m, n0, chunk_size=block, need_states=True)
            nb = L // block
            # pass 2: every (sequence, block) pair is a batch row from the clean-prefix state
            init = torch.cat([torch.zeros_like(states0[:, :1]), states0[:, :-1]], 1)  # [B, nb, H, K, V]
            init = init.reshape(B * nb, *init.shape[2:])
            kmin1 = m.conv_kernel_size - 1
            padded = torch.cat([torch.zeros_like(raw0[:, :, :kmin1]), raw0], 2)
            prefix = padded.unfold(2, kmin1, block)[:, :, :nb].permute(0, 2, 1, 3).reshape(B * nb, -1, kmin1)
            ot, _, _ = gdn_mixer_forward(
                m, nt.reshape(B * nb, block, -1), chunk_size=block, conv_prefix=prefix, initial_state=init, need_states=False
            )
            ot = ot.reshape(B, L, -1)
        elif self.kind == FULL:
            out, _ = layer.self_attn(
                hidden_states=torch.cat([nt, n0], 1), position_embeddings=pos, attention_mask=mask
            )
            ot, o0 = out[:, :L], out[:, L:]
        else:
            raise ValueError(f"unknown layer type {self.kind}")
        h0, ht = r0 + o0, rt + ot
        h0 = h0 + layer.mlp(layer.post_attention_layernorm(h0))
        ht = ht + layer.mlp(layer.post_attention_layernorm(ht))
        return h0, ht


def build_and_shard(text_model, layers: int | None = None):
    """Wrap each decoder layer in a LayerStep and FSDP2-shard each unit.

    `reshard_after_forward=False`: the state fork reads the layer's weights four times per step
    (two mixer calls + two mlp calls); resharding between them would re-all-gather per read.
    """
    cfg = text_model.config
    stack = list(text_model.layers)[: cfg.num_hidden_layers if layers is None else layers]
    mp = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    for i, layer in enumerate(stack):
        step = LayerStep(layer, cfg.layer_types[i])
        text_model.layers[i] = step
        fully_shard(step, reshard_after_forward=False, mp_policy=mp)
    return stack


def fsdp_forward(text_model, steps, x0_ids, xt_ids, block: int, checkpointing: bool = False):
    """Training-time forward on the sharded stack; returns final-normed hiddens of both streams.

    Same shape contract as `block_diffusion_forward(output="hidden")`; [2L, 2L] mask, state
    fork, rotary — all identical, the only difference is that each step is an FSDP unit.
    """
    B, L = x0_ids.shape
    h0 = text_model.embed_tokens(x0_ids)
    ht = text_model.embed_tokens(xt_ids)
    cos, sin = _rotary(text_model, h0, L)
    pos = (torch.cat([cos, cos], 1), torch.cat([sin, sin], 1))
    mask = complementary_mask(L, block, h0.dtype, h0.device)

    def _run(step, h0_, ht_):
        return step(h0_, ht_, pos, mask, block)

    for step in steps:
        if checkpointing and torch.is_grad_enabled():
            h0, ht = _grad_checkpoint(partial(_run, step), h0, ht, use_reentrant=False)
        else:
            h0, ht = _run(step, h0, ht)
    return text_model.norm(ht), text_model.norm(h0)


def train_fsdp(
    text_model,
    lm_head,
    steps: list,
    batches,
    *,
    block: int,
    mask_id: int,
    n_steps: int,
    lr: float,
    checkpointing: bool = False,
    grad_clip: float = 1.0,
    log_every: int = 10,
    log=print,
) -> dict:
    """One FSDP data-parallel loop. Each rank consumes the same batch iterator and slices its
    own rows, so step alignment holds and the global batch is world × batch rows."""
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    # dedup by id exactly as train.py does — tied lm_head/embed_tokens must be stepped once
    seen, uniq = set(), []
    for p in list(text_model.parameters()) + list(lm_head.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            uniq.append(p)
    for p in uniq:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(uniq, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    local = None
    losses, seq_tokens, t0, skipped = [], 0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    text_model.train()
    for step in range(1, n_steps + 1):
        x0, prompt_len, valid_len = next(batches)
        x0, prompt_len, valid_len = x0.to(device), prompt_len.to(device), valid_len.to(device)
        if local is None:  # fixed after the first batch: batch must divide evenly across ranks
            assert x0.shape[0] % world == 0, f"batch {x0.shape[0]} not divisible by world {world}"
            local = x0.shape[0] // world
        sl = slice(rank * local, (rank + 1) * local)
        x0_r, pl_r, vl_r = x0[sl], prompt_len[sl], valid_len[sl]
        x0_2, xt_2, m_2 = complementary_step_inputs(x0_r, pl_r, block, mask_id, torch.Generator(device=device).manual_seed(step * 1000 + rank), vl_r)
        ht, _ = fsdp_forward(text_model, steps, x0_2, xt_2, block, checkpointing=checkpointing)
        n_local = int(m_2.sum())
        if n_local:
            logits = lm_head(ht[m_2])
            ce_sum = torch.nn.functional.cross_entropy(logits.float(), x0_2[m_2], reduction="sum")
        else:
            ce_sum = ht.float().sum() * 0.0  # graph-connected zero: grads must still flow
        n_all = torch.tensor([float(n_local)], device=device)
        dist.all_reduce(n_all)
        n_global = int(n_all.item())
        if n_global == 0:
            skipped += 1
            continue
        loss = ce_sum / n_global  # this rank's contribution to the global mean CE
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(uniq, grad_clip, foreach=False)
        opt.step()
        ce_all = ce_sum.detach().clone()
        dist.all_reduce(ce_all)
        losses.append(float(ce_all) / n_global)
        # same accounting as train.py: unique sequence tokens (batch × L), not the doubled view rows
        seq_tokens += x0_r.numel() * world
        if rank == 0 and (step % log_every == 0 or step == n_steps):
            el = time.time() - t0
            log(f"step {step}\tloss {losses[-1]:.4f}\ttargets {n_global}\tseq-tok/s {seq_tokens / el:.0f}\telapsed {el:.0f}s")
    el = time.time() - t0
    peak = torch.tensor([torch.cuda.max_memory_allocated()], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    g_last, g_reason = last_impl()
    rep = {
        "steps": n_steps,
        "skipped_steps": skipped,
        "world": world,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "seq_tokens": seq_tokens,
        "seq_tokens_per_s": seq_tokens / el if el else 0.0,
        "seconds": el,
        "peak_mem_gib": round(peak.item() / 2**30, 2),
        "checkpointing": checkpointing,
        "gdn_impl": g_last,
        "gdn_reason": g_reason,
    }
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--data", default="nvidia/Llama-Nemotron-Post-Training-Dataset")
    p.add_argument("--split", default="chat")
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4, help="GLOBAL rows per step, sliced across ranks; views double what the forward sees")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--gdn", default="torch", choices=["auto", "torch", "fla"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory for report.json (rank 0)")
    a = p.parse_args(argv)
    if a.seq_len % a.block:
        print("COULD-NOT-MEASURE seq-len must be a multiple of block", file=sys.stderr)
        return 2

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from .equivalence import load_checkpoint
    from transformers import AutoTokenizer

    from . import gdn as _gdn

    _gdn.set_impl(a.gdn if a.gdn != "auto" else None)
    text_model, lm_head, _ = load_checkpoint(a.model, torch.bfloat16, "cuda", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(a.model)
    mask_id = mask_token_id(text_model, tok)
    steps = build_and_shard(text_model, a.layers)
    if rank == 0:
        n_sharded = sum(p.numel() for s in steps for p in s.parameters())
        print(f"FSDP\tworld {dist.get_world_size()}\tsharded params {n_sharded/1e9:.2f}B\teplicated (embed+head+norm) {(sum(p.numel() for p in text_model.parameters()) - n_sharded)/1e9:.2f}B")

    batches = chat_batches(a.data, a.split, tok, a.batch, a.seq_len, a.block, a.seed)
    rep = train_fsdp(
        text_model, lm_head, steps, batches, block=a.block, mask_id=mask_id, n_steps=a.steps, lr=a.lr, checkpointing=a.grad_checkpoint
    )
    rep.update({"model": a.model, "data": a.data, "block": a.block, "seq_len": a.seq_len, "batch": a.batch, "lr": a.lr, "layers": a.layers})
    if rank == 0:
        line = json.dumps({k: v for k, v in rep.items() if k not in ("losses",)}, default=str)
        print(f"TRAIN\tsteps {rep['steps']}\tskipped {rep['skipped_steps']}\tloss {rep['loss_first']} -> {rep['loss_last']}\tseq-tok/s {rep['seq_tokens_per_s']:.0f}\tpeak_mem_gib {rep['peak_mem_gib']}\t{rep['seconds']:.0f}s")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "report.json"), "w") as f:
                json.dump(rep, f, indent=1, default=str)
        ok = rep["loss_last"] is not None and math.isfinite(rep["loss_last"])
        print("REPORT\t" + line)
        dist.destroy_process_group()
        return 0 if ok else 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
