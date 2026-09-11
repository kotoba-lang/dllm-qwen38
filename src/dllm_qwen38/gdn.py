"""Gated DeltaNet forward with per-chunk recurrent states (the state fork).

`chunk_gated_delta_rule_with_states` is derived from
`transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule`
(HuggingFace, Apache-2.0). The single change: the recurrent state after *every* chunk is
returned, not only the final one. With chunk_size == block size, the state after chunk k is
exactly S_k, the state a noised block k+1 must start from.

`gdn_mixer_forward` re-implements `Qwen3_5GatedDeltaNet.forward` without the HF cache so that
(a) a conv prefix can be injected (the short causal conv must see the tail of the clean block
before the noised block, exactly as the cache-based decode path does) and (b) a per-batch-row
initial state can be supplied. Both are the mechanism of the state fork; they are not a new
layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, l2norm


def chunk_gated_delta_rule_with_states(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (core_attn_out [B, T, Hv, Dv], states [B, n_chunks, Hv, Dk, Dv]).

    states[:, i] is the recurrent state after chunk i (i.e. after tokens < (i+1)*chunk_size).
    T must be a multiple of chunk_size; the caller pads if it wants padding semantics.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    if sequence_length % chunk_size != 0:
        raise ValueError(f"sequence_length {sequence_length} is not a multiple of chunk_size {chunk_size}")
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    num_chunks = sequence_length // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf")).exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    states = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
        states.append(last_recurrent_state)

    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, torch.stack(states, dim=1)


def gdn_mixer_forward(
    m,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
    conv_prefix: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a `Qwen3_5GatedDeltaNet` module `m` on `hidden_states` [B, T, D].

    conv_prefix: [B, conv_dim, kernel-1] pre-activation qkv projections of the tokens that
      precede this segment (the clean tail), or None for "segment starts the sequence".
    initial_state: [B, Hv, Dk, Dv] recurrent state before the segment, or None for zeros.

    Returns (output [B, T, D], states [B, T/chunk, Hv, Dk, Dv], raw_qkv [B, conv_dim, T]).
    raw_qkv is the pre-conv projection, which is what a later segment's conv_prefix is cut from.
    """
    batch_size, seq_len, _ = hidden_states.shape
    raw_qkv = m.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, C, T]
    z = m.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)

    conv_in = raw_qkv if conv_prefix is None else torch.cat([conv_prefix, raw_qkv], dim=2)
    mixed_qkv = causal_conv1d_fn(conv_in, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation)
    mixed_qkv = mixed_qkv[:, :, -seq_len:].transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, m.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, m.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)

    core, states = chunk_gated_delta_rule_with_states(
        query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state, use_qk_l2norm_in_kernel=True
    )
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(batch_size, seq_len, -1)
    return m.out_proj(core), states, raw_qkv
