"""Fast-dLLM v2 block-diffusion forward for the Qwen3.5/3.8 hybrid (Gated DeltaNet + Gated Attention).

Visibility contract (the paper's three mask components, arXiv 2509.26328 §3), with
block(i) = i // block_size, x_t = noised stream, x_0 = clean stream:

  M_BD  : x_t token i sees x_t token j  iff block(j) == block(i)      (bidirectional in-block)
  M_OBC : x_t token i sees x_0 token j  iff block(j) <  block(i)      (clean prefix only)
  M_BC  : x_0 token i sees x_0 token j  iff block(j) <= block(i)      (block-causal)

Full-attention layers take this literally as a [2L, 2L] additive mask over cat([x_t, x_0]).
Linear-attention (DeltaNet) layers cannot take a mask; `block_diffusion_forward` realises the
same contract with a state fork: the clean stream is run once (block-causal by construction of
the recurrence), the recurrent state and conv tail at every block boundary are kept, and every
noised block is run as its own batch row starting from the state of the clean prefix before
it. Inside a noised block the recurrence stays left-to-right (that is the "semi-bidirectional"
caveat recorded in ADR-2609112640); across blocks the visibility is exactly M_OBC and no x_t
block sees any other x_t block.

`reference_block_logits` is the independent oracle: for each block k it runs the *unmodified*
HuggingFace model on cat([x_0 blocks < k, x_t block k]) with a block-causal 4D mask. The
DeltaNet path there is HF's own chunked kernel (chunk 64, no fork), so agreement between the
two is evidence about the fork, not a tautology.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from .gdn import gdn_mixer_forward

FULL = "full_attention"
LINEAR = "linear_attention"


def _neg(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def block_causal_mask(seq_len: int, block: int, dtype: torch.dtype, device) -> torch.Tensor:
    """[1, 1, S, S] additive mask: token i sees j iff block(j) <= block(i)."""
    idx = torch.arange(seq_len, device=device) // block
    allowed = idx[None, :] <= idx[:, None]
    return torch.zeros(seq_len, seq_len, dtype=dtype, device=device).masked_fill(~allowed, _neg(dtype))[None, None]


def complementary_mask(
    seq_len: int, block: int, dtype: torch.dtype, device, *, leak_xt_prefix: bool = False
) -> torch.Tensor:
    """[1, 1, 2L, 2L] additive mask over cat([x_t, x_0]) with the paper's three components.

    leak_xt_prefix=True is a deliberately WRONG mask (x_t block k also sees x_t blocks < k); it
    exists so the equivalence test can be shown to go red.
    """
    L = seq_len
    idx = torch.arange(L, device=device) // block
    bi, bj = idx[:, None], idx[None, :]
    tt = bj == bi if not leak_xt_prefix else bj <= bi  # M_BD
    tc = bj < bi  # M_OBC
    cc = bj <= bi  # M_BC
    ct = torch.zeros(L, L, dtype=torch.bool, device=device)  # x_0 never sees x_t
    allowed = torch.cat([torch.cat([tt, tc], 1), torch.cat([ct, cc], 1)], 0)
    return torch.zeros(2 * L, 2 * L, dtype=dtype, device=device).masked_fill(~allowed, _neg(dtype))[None, None]


@dataclass
class BreakMode:
    """Deliberate faults, each of which must make the equivalence test fail."""

    name: str = "none"

    @property
    def concat_causal_gdn(self) -> bool:
        # naive port: run DeltaNet over cat([x_t, x_0]) as one causal sequence (x_t sees earlier
        # x_t blocks, never the clean prefix; x_0 sees all of x_t)
        return self.name == "concat-causal-gdn"

    @property
    def leak_xt_prefix(self) -> bool:
        return self.name == "mask-leak"

    @property
    def drop_conv_prefix(self) -> bool:
        # fork the recurrent state but let the short conv start cold at every noised block
        return self.name == "no-conv-prefix"


def _rotary(text_model, hidden: torch.Tensor, seq_len: int):
    pos = torch.arange(seq_len, device=hidden.device).view(1, 1, -1).expand(3, hidden.shape[0], -1)
    return text_model.rotary_emb(hidden, pos)


def block_diffusion_forward(
    text_model,
    lm_head,
    x0_ids: torch.Tensor,
    xt_ids: torch.Tensor,
    block: int,
    *,
    fault: BreakMode | None = None,
    layers: int | None = None,
    output: str = "logits",
    checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-time forward. Returns (logits_t [B, L, V], logits_0 [B, L, V]) — or, with
    output="hidden", the final-normed hidden states [B, L, D] of both streams so the caller can
    project only the positions it scores (a full [B, L, 248k] logits tensor is what OOMed an
    80 GB H100 at batch 8 × 512 on the 0.8B, measured 2026-09-12).

    checkpointing=True wraps each layer in torch.utils.checkpoint (use_reentrant=False) so the
    mixer/attn activations are recomputed in backward instead of held for every layer at once.
    The checkpointed function must bind its layer at call time (functools.partial) and must
    never close over the loop variables: checkpoint recomputes at *backward* time, after the
    layer loop has advanced, so a lambda closing over `layer`/`kind` re-ran the *last* layer
    for every checkpoint — the forward stayed exactly right (torch.equal passed) while grads
    reached only the last checkpointed layer plus embed and norm, 13/108 parameters, ~3× wrong
    on embed_tokens, ~15× wrong on the last mlp, and 50 training steps left the loss on the
    starting value (measured 2026-09-12). That same bug — recomputation running different ops
    than forward — was also the non-reentrant ledger mismatch (219 vs 74 saved tensors on CPU
    with the torch core, 265 vs 77 on the H100 with the fla kernel); it was never about fla.
    Non-reentrant is chosen over reentrant on purpose: its ledger check throws loudly on this
    class of silent forward/backward divergence, where reentrant returned wrong gradients with
    no error at all. No-grad callers (the equivalence test) are unaffected either way.

    x0_ids / xt_ids: [B, L] with L a multiple of `block`. `layers` truncates the stack (the 27B
    first-N-layers run of ADR-2609112640 §3a).
    """
    fault = fault or BreakMode()
    cfg = text_model.config
    if x0_ids.shape != xt_ids.shape or x0_ids.ndim != 2:
        raise ValueError("x0_ids and xt_ids must both be [B, L]")
    B, L = x0_ids.shape
    if L % block:
        raise ValueError(f"L={L} is not a multiple of block={block}")
    nb = L // block

    h0 = text_model.embed_tokens(x0_ids)
    ht = text_model.embed_tokens(xt_ids)
    dtype, device = h0.dtype, h0.device
    cos, sin = _rotary(text_model, h0, L)
    pos_cat = (torch.cat([cos, cos], 1), torch.cat([sin, sin], 1))
    attn_mask = complementary_mask(L, block, dtype, device, leak_xt_prefix=fault.leak_xt_prefix)

    stack = text_model.layers[: cfg.num_hidden_layers if layers is None else layers]

    def _layer_step(layer, kind, h0, ht):
        """One full transformer layer for both streams; the checkpointing unit."""
        r0, rt = h0, ht
        n0, nt = layer.input_layernorm(h0), layer.input_layernorm(ht)
        if kind == LINEAR:
            m = layer.linear_attn
            if fault.concat_causal_gdn:
                out, _, _ = gdn_mixer_forward(m, torch.cat([nt, n0], 1), chunk_size=block, need_states=False)
                ot, o0 = out[:, :L], out[:, L:]
            else:
                # pass 1: clean stream, block-causal by construction; keep S_k and the conv tail
                o0, states0, raw0 = gdn_mixer_forward(m, n0, chunk_size=block, need_states=True)
                # pass 2: every (sequence, block) pair is a batch row starting from the state of
                # the clean prefix before it: rows ordered (b, k) → b*nb + k. The states are not
                # kept (need_states=False — nothing downstream uses them), which under fla's
                # Triton kernel turns nb sequential calls into one batched call.
                init = torch.cat([torch.zeros_like(states0[:, :1]), states0[:, :-1]], 1)  # [B, nb, H, K, V]
                init = init.reshape(B * nb, *init.shape[2:])
                kmin1 = m.conv_kernel_size - 1
                if fault.drop_conv_prefix:
                    prefix = None
                else:
                    padded = torch.cat([torch.zeros_like(raw0[:, :, :kmin1]), raw0], 2)  # zero pad = seq start
                    # [B, C, nb, kmin1] → [B*nb, C, kmin1]
                    prefix = padded.unfold(2, kmin1, block)[:, :, :nb].permute(0, 2, 1, 3).reshape(B * nb, -1, kmin1)
                ot, _, _ = gdn_mixer_forward(
                    m, nt.reshape(B * nb, block, -1), chunk_size=block, conv_prefix=prefix, initial_state=init, need_states=False
                )
                ot = ot.reshape(B, L, -1)
        elif kind == FULL:
            out, _ = layer.self_attn(
                hidden_states=torch.cat([nt, n0], 1), position_embeddings=pos_cat, attention_mask=attn_mask
            )
            ot, o0 = out[:, :L], out[:, L:]
        else:
            raise ValueError(f"unknown layer type {kind}")
        h0, ht = r0 + o0, rt + ot
        h0 = h0 + layer.mlp(layer.post_attention_layernorm(h0))
        ht = ht + layer.mlp(layer.post_attention_layernorm(ht))
        return h0, ht

    for i, layer in enumerate(stack):
        kind = cfg.layer_types[i]
        if checkpointing and torch.is_grad_enabled():
            # partial, never a closure over the loop variables: recompute runs at backward time,
            # after this loop has advanced, and a late-binding lambda would re-run the LAST layer
            # for every checkpoint (the severed-gradient failure, measured 2026-09-12)
            h0, ht = _grad_checkpoint(partial(_layer_step, layer, kind), h0, ht, use_reentrant=False)
        else:
            h0, ht = _layer_step(layer, kind, h0, ht)

    ht, h0 = text_model.norm(ht), text_model.norm(h0)
    if output == "hidden":
        return ht, h0
    return lm_head(ht), lm_head(h0)


@torch.no_grad()
def reference_block_logits(
    text_model, lm_head, x0_ids: torch.Tensor, xt_ids: torch.Tensor, block: int, *, layers: int | None = None
) -> torch.Tensor:
    """Oracle: per block k, unmodified HF forward on cat([x_0[:k*block], x_t block k]) under a
    block-causal 4D mask; returns the last block's logits, concatenated over k → [1, L, V]."""
    cfg = text_model.config
    L = x0_ids.shape[1]
    nb = L // block
    saved = cfg.num_hidden_layers
    if layers is not None:
        cfg.num_hidden_layers = layers  # the HF loop slices `layers[: num_hidden_layers]`
    try:
        outs = []
        for k in range(nb):
            # one HF call per block; rows of the batch are independent sequences of equal length
            seq = torch.cat([x0_ids[:, : k * block], xt_ids[:, k * block : (k + 1) * block]], 1)
            mask = block_causal_mask(seq.shape[1], block, text_model.embed_tokens.weight.dtype, seq.device)
            hs = text_model(input_ids=seq, attention_mask={FULL: mask, LINEAR: None}, use_cache=False).last_hidden_state
            outs.append(lm_head(hs[:, -block:]))
        return torch.cat(outs, 1)
    finally:
        cfg.num_hidden_layers = saved


@torch.no_grad()
def reference_clean_logits(text_model, lm_head, x0_ids: torch.Tensor, block: int, *, layers: int | None = None):
    """Oracle for the clean stream: unmodified HF forward on x_0 under the block-causal mask."""
    cfg = text_model.config
    saved = cfg.num_hidden_layers
    if layers is not None:
        cfg.num_hidden_layers = layers
    try:
        mask = block_causal_mask(x0_ids.shape[1], block, text_model.embed_tokens.weight.dtype, x0_ids.device)
        hs = text_model(input_ids=x0_ids, attention_mask={FULL: mask, LINEAR: None}, use_cache=False).last_hidden_state
        return lm_head(hs)
    finally:
        cfg.num_hidden_layers = saved


def tiny_text_model(seed: int = 0, layers: int = 8, vocab: int = 256):
    """A random-initialised Qwen3.5 text model small enough for a CPU unit test.

    The visibility contract is a property of the code path, not of trained weights, so the
    equivalence test on this model is the real test; the checkpoint run is the demonstration
    that the same code holds on the shapes the 27B has.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=4096,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25},
        attn_implementation="eager",
    )
    model = Qwen3_5TextModel(cfg).eval()
    # A_log must give a decay strictly inside (0,1) per token; random init already does, but make
    # in_proj_a / dt_bias produce a non-degenerate g so the recurrence actually carries state.
    lm_head = torch.nn.Linear(cfg.hidden_size, vocab, bias=False)
    return model, lm_head
