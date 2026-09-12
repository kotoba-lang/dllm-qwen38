"""FSDP training (H100×8) for the 27B on the state-fork forward — ADR-2609112640 §4.

The structural constraint the fork imposes: `block_diffusion_forward` never calls a layer
itself. It reaches straight into each layer's submodules (`layer.input_layernorm`,
`layer.linear_attn` via `gdn_mixer_forward`, `layer.self_attn`, `layer.mlp`) and, for DeltaNet
layers, runs two mixer calls with shape-changing batch rows (B*nb rows for the noised blocks,
per-block states [B, nb, H, K, V] and conv prefixes read off the clean pass). FSDP-wrapping
the HF layer itself would all-gather only when the *layer* is called — which never happens —
so every submodule call would read a **local shard**: silent wrong answers, no error.

The fix here is one reference `nn.Module` (`LayerStep`) holding the layer-step body exactly as
`blockdiff.py::_layer_step` has it, wrapped as the FSDP unit with `reshard_after_forward=True`
(gather for the forward's four reads — two mixer calls + two mlp calls — reshard after;
measured at 0.8B: +21% throughput, identical loss trajectory). Topology, read against torch
2.14 FSDP2 source plus a world-1 probe: the 64 sibling `fully_shard` calls under a non-FSDP
parent each lazily init as their OWN root — 64 states of one param group each, no aggregation,
each with its own comm_ctx and a `post_forward_order` of just itself. Two consequences:
`_backward_prefetch` sees curr_index=0 in every state and never prefetches (backward is fully
synchronous per unit), and `reshard_after_backward` defaults True, so each unit's
post_backward hook reshards — copy-out storages are alloc/free pairs on
`FSDPParam.all_gather_outputs` and nothing here pins a buffer by hand. Forward is measured
clean at 27B scale (run fsdp-train-20260912-072552: after-forward 11.79 GiB — one 0.78 GiB
copy-out kept alive per unit would be ~62 GiB; at-OOM census 11 live / 0.55 GiB). The
step-1-backward OOM (76.75 GiB allocated on all 8 ranks, runs -071458/-072155/-072552) is
MEASURED (run fsdp-train-20260912-094911, `--mem-history`): `post_backward` parks one fp32
reduce-scatter input per unit (~1.42 GiB at 27B) in `comm_ctx.reduce_scatter_states` — a
list torch designs as ONE shared list with a global cap of 1 (`_fsdp_param_group.py`
"a single shared list governed by one cap"), but which the sibling topology turns into 64
private lists, so every backward-completed unit parks its input until the end-of-backward
callback. At unit 41 of step 1's backward that is 41 parked = 58.09 GiB (census), and the
allocation snapshot puts 63.2 GiB on the `post_backward` stack. The fix in `build_and_shard`
shares one list object across all units, restoring the intended recycling (≤1 input in
flight). At 0.8B neither lever changes the loss trajectory — gathering order moves WHEN
weights are gathered, never what they are.
`embed_tokens`, `norm`, and `lm_head` stay replicated
(unsharded): they sit outside every FSDP unit by construction, and at 27B that is ~2.5 GB bf16
per rank against a 52 GB sharded tower.

Loss aggregation must survive a rank with zero local targets (DDP sync): each rank backwards
its local sum CE divided by the *global* target count (all-reduced), with a graph-connected
zero when it has no targets, so FSDP's gradient averaging lands on the global mean CE; if the
global count is zero all ranks skip the step together. At world_size 1 this reduces exactly to
`train.py`'s step — that parity (0.8B, same seed, same data) is the correctness gate before
the 27B is trusted with this path.

Two world>1 correctness fixes live here that a world-1 run cannot see and a world-8 run must:
the replicated params (embed_tokens / norm / tied lm_head) sit outside every FSDP unit, so
their grads are DDP-averaged by hand before the step; and the global-norm clip is computed
here rather than via `torch.nn.utils.clip_grad_norm_`, which dies on the mixed DTensor/plain
grad list (both fixes measured on the H200 smokes 2026-09-12).

Run (Modal, gpu="H100:8"): torchrun --nproc_per_node=8 --module dllm_qwen38.fsdp_train ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
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

    `reshard_after_forward=True`: gather for the forward's four reads (two mixer calls + two
    mlp calls), reshard afterward — measured at 0.8B: +21% seq-tok/s, identical trajectory.
    Under the sibling topology (each unit its own FSDP root, one group per state) torch
    2.14's backward is fully synchronous: no prefetch (the loop needs curr_index > 0), and
    each unit's post_backward hook reshards (`reshard_after_backward` defaults True), so a
    unit's copy-out storage is freed inside backward, not held to its end.

    The third sibling-topology consequence is the reduce-scatter input parking, measured at
    27B (run fsdp-train-20260912-094911): `post_backward` parks each unit's fp32
    reduce-scatter input (~1.42 GiB) in `comm_ctx.reduce_scatter_states`, whose built-in
    recycle (`pop(0)` + `wait_event` + del before allocating a fresh input) is governed by a
    cap on a list torch's design comment calls "a single shared list" — under 64 private
    comm_ctxs each recycle only ever sees its own list, all 64 inputs stay parked until the
    end-of-backward callback, and step 1 OOMs at unit 41 (58.09 GiB parked). Sharing ONE
    Python list object across the units' comm_ctxs restores the intended recycling: the
    next unit's post_backward pops+waits the previous unit's input, so ≤1 is in flight
    (same ops, same order — only WHEN the ~1.4 GiB is freed changes). The install CANNOT
    happen here: FSDPCommContext has no __init__ — every attribute, `reduce_scatter_states`
    included, is created inside `lazy_init()`, which runs on each unit's first forward, so
    any attribute set before that is overwritten by a fresh empty list (measured at 27B,
    run fsdp-train-20260912-102033: the build-time install OOMed with the identical
    41-parked signature). The install therefore lives in `train_fsdp` right after the
    first forward, when all units have completed lazy init and before the first backward.
    """
    cfg = text_model.config
    stack = list(text_model.layers)[: cfg.num_hidden_layers if layers is None else layers]
    mp = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    steps = []
    for i, layer in enumerate(stack):
        step = LayerStep(layer, cfg.layer_types[i])
        text_model.layers[i] = step
        fully_shard(step, reshard_after_forward=True, mp_policy=mp)
        steps.append(step)
    return steps  # the LayerSteps, NOT `stack`: the raw HF layers must never be called again
    # (calling a raw layer reads its sharded DTensor params with no unshard hook — the
    # mixed-Tensor/DTensor crash of the first smoke, 2026-09-12)


def _install_shared_rs_states(steps: list, log=print) -> None:
    """Share ONE reduce_scatter_states list across all FSDP units — call after the first forward.

    Must run POST-lazy-init (see build_and_shard's docstring for the measured why): the
    comm_ctx object is stable from fully_shard on, but its `reduce_scatter_states`
    attribute is (re)created inside `lazy_init()`, which fires on each unit's first
    forward. After the first full model forward every unit has lazy-inited, and no
    backward has parked anything yet, so installing here is in time for step 1's parks
    and stays valid for the whole run (lazy_init never re-fires; the list is only ever
    mutated in place — `append`/`pop(0)`/`clear` by torch's recycle and drains).

    The sibling topology keeps 64 DISTINCT comm_ctx objects (each unit is its own root;
    `_init_shared_state` sees only its own subtree, so all_states = [self] per unit) —
    the sharing is by list-object identity, not comm_ctx identity. The probe asserts
    exactly that: one distinct list id across every state and group, loudly, because a
    partial share just moves the OOM to a different unit count.
    """
    shared: list = []
    list_ids: list[int] = []
    for st in steps:
        s = st._get_fsdp_state()
        s._comm_ctx.reduce_scatter_states = shared
        list_ids.append(id(shared))
        for grp in s._fsdp_param_groups:
            grp.comm_ctx.reduce_scatter_states = shared
            list_ids.append(id(grp.comm_ctx.reduce_scatter_states))
    n_distinct = len(set(list_ids))
    if n_distinct != 1:
        print(f"RS-SHARE\tWARN\t{n_distinct} distinct list objects (expected 1) — parking not shared, OOM risk", flush=True)
    else:
        print(f"RS-SHARE\tok\t{len(steps)} units share one list (id {list_ids[0] % 2**32:#x})", flush=True)


def _sync_replicated_grads(uniq) -> None:
    """DDP-average the grads of params outside every FSDP unit.

    FSDP2 reduce-scatters its units' grads (average), but embed_tokens / norm / the tied
    lm_head sit outside all units, so each rank backward's only its own rows: without this the
    optimizer steps would diverge across ranks — a silent wrong answer world-1 parity cannot
    see. "Plain" here means exactly the params whose grad is not a DTensor.
    """
    world = dist.get_world_size()
    for p in uniq:
        g = p.grad
        if g is None or isinstance(g, DTensor):
            continue
        dist.all_reduce(g)
        g.div_(world)


def _clip_grads(uniq, max_norm: float) -> None:
    """One global 2-norm clip across FSDP units (DTensor grads) and replicated params (plain grads).

    `torch.nn.utils.clip_grad_norm_` cannot take the mixed list — replicated params' grads are
    plain tensors while FSDP units' are DTensors, so its total_norm mixes the two and
    `g.mul_(clip_coef)` dies with `aten.mul_.Tensor got mixed torch.Tensor and DTensor` (H200
    smoke 2026-09-12). Same math as train.py's clip_grad_norm_: fp32 per-grad norms, coef =
    max_norm / (total + 1e-6) clamped at 1. The sharded grads' local sq-norms are summed over
    the world (each rank holds one shard); the plain grads were world-averaged by
    `_sync_replicated_grads` before this runs and are identical on every rank, so their sq-norm
    must be counted once — that is why the plain bucket is divided by world here (the all-reduce
    would otherwise count it once per rank).
    """
    pair = torch.zeros(2, dtype=torch.float32, device="cuda")
    for p in uniq:
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            pair[0] += g.to_local().float().norm() ** 2
        else:
            pair[1] += g.float().norm() ** 2
    dist.all_reduce(pair)  # SUM: [0] = sharded sq-norm across the world, [1] = world × plain sq-norm
    total = math.sqrt(pair[0].item() + pair[1].item() / dist.get_world_size())
    coef = min(1.0, max_norm / (total + 1e-6))
    if coef < 1.0:  # == 1.0 (incl. the all-zero-grad path) is an exact no-op, skip the mul
        for p in uniq:
            if p.grad is not None:
                p.grad.mul_(coef)  # python-float mul dispatches the Scalar overload — DTensor-safe


def _gather_census(steps, tag: str) -> None:
    """Live per-unit all-gather buffers — the leak-class probe at 27B scale.

    `FSDPParam.alloc_all_gather_outputs` and `free_unsharded_param` are a matched pair on the
    `all_gather_outputs` list; this counts the list members whose storage is allocated
    (`untyped_storage().nbytes() > 0`), so a freed buffer counts zero and a leaked live
    copy-out shows up as n_live × ~0.78 GiB at 27B (invisible at 0.8B, where the same
    per-buffer size is ~31 MB and the whole pool is 1 GiB). The `_all_gather_result` and
    deferred `comm_ctx.all_gather_state` counts are the group-level handles that could pin a
    result past its own wait. `rs_inputs` counts the parked reduce-scatter input buffers in
    the (now shared) `comm_ctx.reduce_scatter_states` — post_backward parks one per unit;
    before the shared-list fix (run fsdp-train-20260912-094911) the private-list topology
    left every backward-completed unit's input parked until the end-of-backward callback
    (41 parked = 58.09 GiB at the step-1 OOM); with the fix, the next unit's post_backward
    recycles the previous one's, so this reads ≤1 during backward and 0 between steps.
    """
    n_live, total, n_result, n_defer = 0, 0, 0, 0
    n_rs, rs_total = 0, 0
    for st in steps:
        state = st._get_fsdp_state()
        for grp in state._fsdp_param_groups:
            if grp._all_gather_result is not None:
                n_result += 1
            if grp.comm_ctx.all_gather_state is not None:
                n_defer += 1
            for rs in grp.comm_ctx.reduce_scatter_states:
                nb = rs.reduce_scatter_input.untyped_storage().nbytes()
                if nb > 0:
                    n_rs += 1
                    rs_total += nb
            for p in grp.fsdp_params:
                for t in p.all_gather_outputs:
                    nb = t.untyped_storage().nbytes()
                    if nb > 0:
                        n_live += 1
                        total += nb
    print(
        f"MEMCENSUS\t{tag}\tlive_copies={n_live}\ttotal_gib={total / 2**30:.2f}\tresults={n_result}\tdeferred={n_defer}\trs_inputs={n_rs}\trs_gib={rs_total / 2**30:.2f}",
        flush=True,
    )


def _dump_memhist(rank: int) -> None:
    """Aggregate live CUDA blocks by their allocation stack and print the top consumers.

    Called from the OOM handler with `torch.cuda.memory._record_memory_history` active, so
    each allocated block in the snapshot carries its alloc stack (`blocks[i]["frames"]`).
    One line per collapsed stack (last 3 frames): live bytes and block count.
    """
    try:
        snap = torch.cuda.memory._snapshot()
    except Exception as e:  # never mask the OOM we re-raise
        print(f"[MEMHIST rank {rank}] snapshot failed: {e!r}", flush=True)
        return
    live: dict[str, list[int]] = {}
    for seg in snap.get("segments", []):
        for b in seg.get("blocks", []):
            if b.get("state") != "active_allocated":  # snapshot state strings are lowercase-hyphen ("active_allocated"/"inactive"/"free" — measured 2026-09-12: "ALLOCATED" matches nothing and the dump prints 0.00 GiB)
                continue
            frames = b.get("frames") or []
            key = " <- ".join(f["name"].split("/")[-1] for f in frames[-3:]) or "<no stack>"
            ent = live.setdefault(key, [0, 0])
            ent[0] += int(b["size"])
            ent[1] += 1
    top = sorted(live.items(), key=lambda kv: -kv[1][0])[:25]
    tot = sum(v[0] for v in live.values())
    print(f"[MEMHIST rank {rank}] live {tot / 2**30:.2f} GiB in {sum(v[1] for v in live.values())} blocks", flush=True)
    for key, (bts, c) in top:
        print(f"[MEMHIST rank {rank}] {bts / 2**20:9.1f} MiB n={c:5d} {key}", flush=True)


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
    mem_history: bool = False,
    grad_clip: float = 1.0,
    log_every: int = 10,
    log=print,
) -> dict:
    """One FSDP data-parallel loop. Each rank consumes the same batch iterator and slices its
    own rows, so step alignment holds and the global batch is world × batch rows."""
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    # dedup by id exactly as train.py does — tied lm_head/embed_tokens must be stepped once
    seen, uniq, uniq_names = set(), [], []
    for n, p in list(text_model.named_parameters()) + list(lm_head.named_parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            uniq.append(p)
            uniq_names.append(n)
    for p in uniq:
        p.requires_grad_(True)
    # foreach=False: the mixed param list (FSDP units are DTensors, embed/norm/lm_head are
    # plain) would be grouped into one multi-tensor op by foreach AdamW and die with the same
    # mixed-Tensor/DTensor error as clip_grad_norm_ did; the single-tensor path never mixes
    opt = torch.optim.AdamW(uniq, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    local = None
    losses, seq_tokens, t0, skipped = [], 0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    text_model.train()
    if mem_history and rank == 0:
        # ring buffer of alloc/free stacks, last 60k events; rank 0 only — the OOM fires
        # identically on all ranks, and the snapshot's live view (segments + per-block alloc
        # stacks) is what we need. Live blocks carry their frames regardless of ring
        # truncation; 60k bounds the snapshot build so it fits inside the SIGKILL window
        # after the SIGTERM-ignored dump starts. History stops after step 3 (the OOM is in
        # step-1 backward, if it fires at all)
        torch.cuda.memory._record_memory_history(max_entries=60000)
    for step in range(1, n_steps + 1):
        if mem_history and rank == 0 and step == 4:
            torch.cuda.memory._record_memory_history(None)  # bound the recording overhead
        try:
            x0, prompt_len, valid_len = next(batches)
            x0, prompt_len, valid_len = x0.to(device), prompt_len.to(device), valid_len.to(device)
            if local is None:  # fixed after the first batch: batch must divide evenly across ranks
                assert x0.shape[0] % world == 0, f"batch {x0.shape[0]} not divisible by world {world}"
                local = x0.shape[0] // world
            sl = slice(rank * local, (rank + 1) * local)
            x0_r, pl_r, vl_r = x0[sl], prompt_len[sl], valid_len[sl]
            x0_2, xt_2, m_2 = complementary_step_inputs(x0_r, pl_r, block, mask_id, torch.Generator(device=device).manual_seed(step * 1000 + rank), vl_r)
            ht, _ = fsdp_forward(text_model, steps, x0_2, xt_2, block, checkpointing=checkpointing)
            if step == 1:
                # post-lazy-init shared RS list — see _install_shared_rs_states; every rank
                # installs (each rank owns its own Python objects). In time for step 1's
                # backward: the first forward completed all 64 units' lazy init.
                _install_shared_rs_states(steps, log=print if rank == 0 else (lambda *a, **k: None))
            if rank == 0 and step == 1:
                print(f"MEM\tafter fwd\tallocated {torch.cuda.memory_allocated()/2**30:.2f} GiB\treserved {torch.cuda.memory_reserved()/2**30:.2f} GiB")
            if mem_history and rank == 0 and step <= 2:
                _gather_census(steps, f"after fwd s{step}")
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
            if rank == 0 and step == 1:
                print(f"MEM\tafter bwd\tallocated {torch.cuda.memory_allocated()/2**30:.2f} GiB\treserved {torch.cuda.memory_reserved()/2**30:.2f} GiB")
            if mem_history and rank == 0 and step <= 2:
                _gather_census(steps, f"after bwd s{step}")
            _sync_replicated_grads(uniq)  # before the clip: the clip must see the averaged plain grads
            _clip_grads(uniq, grad_clip)
            opt.step()
            ce_all = ce_sum.detach().clone()
            dist.all_reduce(ce_all)
            losses.append(float(ce_all) / n_global)
            # same accounting as train.py: unique sequence tokens (batch × L), not the doubled view rows
            seq_tokens += x0_r.numel() * world
        except torch.OutOfMemoryError:
            if mem_history and rank == 0:
                # torchrun tears the pool down ~85ms after the FIRST rank to exit (measured on
                # run fsdp-train-20260912-093247: all 8 OOM together, the first exitcode-1 fired
                # the SIGTERMs and rank 0's snapshot build was killed silently mid-build). So the
                # fast census goes first — it is a plain attribute walk and prints in
                # microseconds — and only the slow stack snapshot runs under SIG_IGN (the agent
                # SIGKILLs ~30s later, the build needs seconds).
                print(f"MEM\tat OOM\tallocated {torch.cuda.memory_allocated()/2**30:.2f} GiB\treserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
                _gather_census(steps, "at OOM")
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                print("[MEMHIST rank 0] OOM caught; snapshotting alloc stacks (SIGTERM ignored)", flush=True)
                _dump_memhist(rank)
            raise
        if rank == 0 and (step % log_every == 0 or step == n_steps):
            el = time.time() - t0
            log(f"step {step}\tloss {losses[-1]:.4f}\ttargets {n_global}\tseq-tok/s {seq_tokens / el:.0f}\telapsed {el:.0f}s")
    el = time.time() - t0
    # weight-divergence probe over the replicated params only — the machinery this file adds
    # by hand. Sharded params are skipped on purpose: rank r holds only shard r, so per-rank
    # checksums differ by design and comparing them false-positives on a perfectly-synced run
    # (first probe measured exactly that: spread 1032 with a clean trajectory, 2026-09-12);
    # their consistency is FSDP2's reduce-scatter contract, not this file's. Equal fp64
    # checksums across ranks on a plain param mean bitwise-equal weights: same tensor, summed
    # in the same order. A trajectory match cannot substitute — ranks mask different rows, so
    # losses legitimately differ while weights silently drift apart.
    names, cks = [], []
    for n, p in zip(uniq_names, uniq):
        if isinstance(p, DTensor):
            continue
        names.append(n)
        cks.append(p.detach().sum(dtype=torch.float64))
    if cks:
        sp = torch.stack(cks)
        lo, hi = sp.clone(), sp
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        sp = hi - lo
        spread = float(sp.max().item())
        bad = [(names[i], float(sp[i].item())) for i in (sp > 0).nonzero().flatten().tolist()]
    else:
        spread, bad = 0.0, []
    if bad:
        print(f"WEIGHT-SPREAD\tdivergent {len(bad)}/{len(names)}\t{bad[:8]}")
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
        "weight_spread": spread,
        "divergent_params": bad[:8],
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
    p.add_argument("--mem-history", action="store_true", help="record alloc stacks (MEMHIST dump at OOM) + MEMCENSUS per-unit gather census")
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
        print(f"MEM\tafter shard\tallocated {torch.cuda.memory_allocated()/2**30:.2f} GiB\treserved {torch.cuda.memory_reserved()/2**30:.2f} GiB")
    if rank == 0:
        n_sharded = sum(p.numel() for s in steps for p in s.parameters())
        print(f"FSDP\tworld {dist.get_world_size()}\tsharded params {n_sharded/1e9:.2f}B\treplicated (embed+head+norm) {(sum(p.numel() for p in text_model.parameters()) - n_sharded)/1e9:.2f}B")

    batches = chat_batches(a.data, a.split, tok, a.batch, a.seq_len, a.block, a.seed)
    rep = train_fsdp(
        text_model, lm_head, steps, batches, block=a.block, mask_id=mask_id, n_steps=a.steps, lr=a.lr, checkpointing=a.grad_checkpoint, mem_history=a.mem_history
    )
    rep.update({"model": a.model, "data": a.data, "block": a.block, "seq_len": a.seq_len, "batch": a.batch, "lr": a.lr, "layers": a.layers})
    if rank == 0:
        line = json.dumps({k: v for k, v in rep.items() if k not in ("losses",)}, default=str)
        print(f"TRAIN\tsteps {rep['steps']}\tskipped {rep['skipped_steps']}\tloss {rep['loss_first']} -> {rep['loss_last']}\tseq-tok/s {rep['seq_tokens_per_s']:.0f}\tpeak_mem_gib {rep['peak_mem_gib']}\tweight_spread {rep['weight_spread']:.3g}\t{rep['seconds']:.0f}s")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "report.json"), "w") as f:
                json.dump(rep, f, indent=1, default=str)
        ok = rep["loss_last"] is not None and math.isfinite(rep["loss_last"]) and rep["weight_spread"] == 0.0
        print("REPORT\t" + line)
        dist.destroy_process_group()
        return 0 if ok else 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
