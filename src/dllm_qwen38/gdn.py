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
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn as transformers_causal_conv1d_fn, l2norm

# Which core the mixer used for its *last* call, and why — set so reports can name the path they
# measured rather than assuming the fast one ran ("fla" / "reference" / a fallback reason).
_FLA = {"fn": None, "tried": False, "force": None, "last": "reference", "reason": ""}

# The Transformers callable silently falls back when causal-conv1d is absent.  That is useful in
# production but invalid for an A/B: both legs could measure the same fallback while being named
# differently.  The experiment selector below has two explicit, fail-closed modes and keeps
# ``None`` as the existing auto/Transformers behaviour.
_CONV = {
    "fn": None,
    "tried": False,
    "force": None,
    "last": "unmeasured",
    "reason": "",
    "package_version": None,
}


def _fla_chunk():
    if not _FLA["tried"]:
        _FLA["tried"] = True
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            _FLA["fn"] = chunk_gated_delta_rule
        except Exception as e:  # noqa: BLE001 — the reason is the point
            _FLA["reason"] = f"{type(e).__name__}: {e}"
    return _FLA["fn"]


def set_impl(name):
    """'torch' | 'fla' | None (auto: fla when importable and on CUDA)."""
    _FLA["force"] = name


def last_impl():
    return _FLA["last"], _FLA["reason"]


def set_conv_impl(name):
    """Select ``reference`` / ``optimized`` / ``None`` (Transformers auto behaviour)."""
    if name not in (None, "reference", "optimized"):
        raise ValueError(f"unknown causal-conv1d implementation {name!r}")
    _CONV["force"] = name


def last_conv_impl():
    """Return the measured backend, diagnostic reason, and installed package version."""
    return _CONV["last"], _CONV["reason"], _CONV["package_version"]


def _optimized_conv():
    if not _CONV["tried"]:
        _CONV["tried"] = True
        try:
            import importlib.metadata

            import causal_conv1d_cuda
            from causal_conv1d import causal_conv1d_fn

            extension_path = str(getattr(causal_conv1d_cuda, "__file__", ""))
            if not extension_path.endswith((".so", ".pyd")):
                raise RuntimeError(f"causal_conv1d_cuda is not a compiled extension: {extension_path!r}")
            _CONV["fn"] = causal_conv1d_fn
            _CONV["package_version"] = importlib.metadata.version("causal-conv1d")
        except Exception as exc:  # noqa: BLE001 - the exact refusal reason belongs in the report
            _CONV["reason"] = f"{type(exc).__name__}: {exc}"
    return _CONV["fn"]


def _reference_causal_conv1d(x, weight, bias=None, activation=None):
    """The Transformers PyTorch fallback, made explicit for the control leg."""
    seq_len = x.shape[-1]
    out = F.conv1d(
        x.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=weight.shape[-1] - 1,
        groups=x.shape[1],
    )[:, :, :seq_len]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported causal conv activation {activation!r}")
    return out.to(x.dtype)


def _causal_conv1d(x, weight, bias=None, activation=None):
    force = _CONV["force"]
    if force == "reference":
        _CONV["last"], _CONV["reason"] = "reference", "forced PyTorch grouped conv1d"
        return _reference_causal_conv1d(x, weight, bias, activation)
    if force == "optimized":
        fn = _optimized_conv()
        if fn is None:
            _CONV["last"] = "unavailable"
            raise RuntimeError(f"optimized causal-conv1d forced but unavailable: {_CONV['reason']}")
        out = fn(x, weight, bias, activation=activation)
        _CONV["last"], _CONV["reason"] = "optimized", "compiled causal_conv1d_cuda extension"
        return out
    _CONV["last"], _CONV["reason"] = "transformers-auto", "Transformers dispatch/fallback"
    return transformers_causal_conv1d_fn(x, weight, bias, activation=activation)


def conv_kernel_diff(m, *, T: int, B: int = 2, seed: int = 0, device: str = "cuda") -> dict:
    """Compare optimized causal-conv1d with the explicit reference, including backward.

    The probe uses detached copies of the real mixer's convolution parameters, so it cannot
    populate or perturb model gradients.  ``fork_rng`` prevents the probe from changing the
    subsequent training stream.
    """
    previous = _CONV["force"]
    cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            dtype = next(m.parameters()).dtype
            channels = m.conv1d.weight.shape[0]
            base_x = torch.randn(B, channels, T, dtype=dtype, device=device)
            base_w = m.conv1d.weight.squeeze(1).detach()
            base_b = m.conv1d.bias.detach() if m.conv1d.bias is not None else None
            grad_out = torch.randn(B, channels, T, dtype=dtype, device=device)

            results = {}
            for implementation in ("reference", "optimized"):
                x = base_x.detach().clone().requires_grad_(True)
                w = base_w.detach().clone().requires_grad_(True)
                b = base_b.detach().clone().requires_grad_(True) if base_b is not None else None
                set_conv_impl(implementation)
                out = _causal_conv1d(x, w, b, activation=m.activation)
                inputs = (x, w, b) if b is not None else (x, w)
                grads = torch.autograd.grad(out, inputs, grad_outputs=grad_out)
                results[implementation] = (out.detach(), tuple(g.detach() for g in grads))

            ref_out, ref_grads = results["reference"]
            opt_out, opt_grads = results["optimized"]
            names = ("input", "weight", "bias")[: len(ref_grads)]
            grad_diffs = {
                name: float((got.float() - want.float()).abs().max())
                for name, want, got in zip(names, ref_grads, opt_grads)
            }
            grad_dot = sum(
                (want.float() * got.float()).sum() for want, got in zip(ref_grads, opt_grads)
            )
            ref_grad_sq = sum(want.float().square().sum() for want in ref_grads)
            opt_grad_sq = sum(got.float().square().sum() for got in opt_grads)
            grad_cosine = grad_dot / (ref_grad_sq.sqrt() * opt_grad_sq.sqrt()).clamp_min(1e-30)
            grad_norm_ratio = opt_grad_sq.sqrt() / ref_grad_sq.sqrt().clamp_min(1e-30)
            last, reason, version = last_conv_impl()
            if last != "optimized":
                raise RuntimeError(f"optimized causal-conv1d path not taken: {last} ({reason})")
            return {
                "out_max_abs_diff": float((opt_out.float() - ref_out.float()).abs().max()),
                "grad_max_abs_diff": max(grad_diffs.values()),
                "grad_max_abs_diff_by_input": grad_diffs,
                "grad_cosine": float(grad_cosine),
                "grad_norm_ratio": float(grad_norm_ratio),
                "all_finite": bool(
                    torch.isfinite(opt_out).all()
                    and all(torch.isfinite(grad).all() for grad in opt_grads)
                ),
                "dtype": str(base_x.dtype),
                "shape": list(base_x.shape),
                "optimized_backend": last,
                "backend_reason": reason,
                "package_version": version,
            }
    finally:
        set_conv_impl(previous)


def chunk_gated_delta_rule_with_states(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    need_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (core_attn_out [B, T, Hv, Dv], states [B, n_chunks, Hv, Dk, Dv]).

    states[:, i] is the recurrent state after chunk i (i.e. after tokens < (i+1)*chunk_size).
    T must be a multiple of chunk_size; the caller pads if it wants padding semantics.
    With need_states=False the per-chunk states are not stacked and None is returned — the
    loop is still sequential (it is a recurrence), so this saves only bookkeeping, which is
    why the reference never wanted this flag and fla does.
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

    states = [] if need_states else None
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
        if need_states:
            states.append(last_recurrent_state)

    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, torch.stack(states, dim=1) if need_states else None


def _fla_core(fn, q, k, v, g, beta, *, chunk_size, initial_state, need_states):
    """fla's Triton chunk kernel. With need_states, per-block states require calling it once per
    block (fla returns only the final state of a call), so the clean pass runs nb sequential
    calls at exactly the same block boundaries the reference chunks at. Without need_states the
    whole [B*nb, block] batch of independent rows is one call."""
    T = q.shape[1]
    if not need_states:
        o, _ = fn(q, k, v, g, beta, initial_state=initial_state, use_qk_l2norm_in_kernel=True, chunk_size=chunk_size)
        _FLA["last"] = "fla"
        return o, None
    nb = T // chunk_size
    outs, states, s = [], [], initial_state
    for i in range(nb):
        sl = slice(i * chunk_size, (i + 1) * chunk_size)
        o, s = fn(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=s, output_final_state=True, use_qk_l2norm_in_kernel=True, chunk_size=chunk_size)
        outs.append(o)
        states.append(s)
    _FLA["last"] = "fla"
    return torch.cat(outs, 1), torch.stack(states, 1)


def _core(q, k, v, g, beta, *, chunk_size, initial_state, need_states):
    """fla's Triton kernel when importable and on CUDA (unless forced off), else the reference —
    which is also the fallback when the kernel rejects the inputs (e.g. an unsupported dtype);
    the fallback re-runs the whole core so no output is ever half-and-half."""
    fn = _fla_chunk()
    force = _FLA["force"]
    if fn is not None and q.is_cuda and force != "torch":
        try:
            return _fla_core(fn, q, k, v, g, beta, chunk_size=chunk_size, initial_state=initial_state, need_states=need_states)
        except Exception as e:  # noqa: BLE001 — recorded, then measured again on the reference
            _FLA["last"] = f"reference (fla failed: {type(e).__name__}: {e})"
    elif force == "fla":
        raise RuntimeError(f"fla forced but unavailable: {_FLA['reason'] or 'not on CUDA'}")
    _FLA["last"] = "reference"
    return chunk_gated_delta_rule_with_states(
        q, k, v, g, beta, chunk_size=chunk_size, initial_state=initial_state, use_qk_l2norm_in_kernel=True, need_states=need_states
    )


def kernel_diff(m, *, T: int, B: int = 2, seed: int = 0, device: str = "cuda") -> dict:
    """Max |Δ| between the torch reference and fla's Triton kernel on a real mixer module `m`,
    same bf16 inputs, with per-block states — the fla-vs-reference question the ADR records as
    open. Refuses to answer when the fla path was not actually taken (not on CUDA, not
    importable) instead of reporting a fake 0 diff."""
    torch.manual_seed(seed)
    hs = torch.randn(B, T, m.in_proj_qkv.in_features, device=device, dtype=next(m.parameters()).dtype)
    _FLA["last"], _FLA["reason"] = "reference", ""
    try:
        set_impl("torch")
        with torch.no_grad():
            o_ref, s_ref, _ = gdn_mixer_forward(m, hs, chunk_size=T, need_states=True)
        set_impl("fla")
        with torch.no_grad():
            o_fla, s_fla, _ = gdn_mixer_forward(m, hs, chunk_size=T, need_states=True)
        last, reason = last_impl()
        if last != "fla":
            raise RuntimeError(f"fla path not taken (last={last}{' ' + reason if reason else ''}); diff would be meaningless")
        return {
            "out_max_abs_diff": float((o_fla.float() - o_ref.float()).abs().max()),
            "states_max_abs_diff": float((s_fla.float() - s_ref.float()).abs().max()),
            "dtype": str(hs.dtype),
        }
    finally:
        set_impl(None)


def gdn_mixer_forward(
    m,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
    conv_prefix: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    need_states: bool = True,
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
    mixed_qkv = _causal_conv1d(conv_in, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation)
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

    core, states = _core(query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state, need_states=need_states)
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(batch_size, seq_len, -1)
    return m.out_proj(core), states, raw_qkv
