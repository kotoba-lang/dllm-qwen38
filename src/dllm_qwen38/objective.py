"""Fast-dLLM v2 training objective: complementary masking + masked-token-only cross-entropy.

For every sequence x_0 draw one mask m (per block, ratio t ~ U(0, 1)); train on BOTH views
x_t^(m) and x_t^(1-m), so every response token is a target exactly once per step (paper §3:
"complementary masking ... all L tokens contribute to the loss"). Prompt tokens are never
noised and never scored: a block that lies entirely inside the prompt is clean in both views.

`[MASK]` is one embedding row the tokenizer does not use (`mask_token_id`); no resize when
spare rows exist, which is the case measured for Qwen3.5-0.8B (243 spare rows).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_token_id(text_model, tokenizer) -> int:
    rows = text_model.embed_tokens.weight.shape[0]
    n = len(tokenizer)
    if rows <= n:
        raise ValueError(f"no spare embedding row for [MASK]: rows={rows} tokenizer={n}; resize first")
    return n  # first unused row


def sample_complementary_masks(
    x0: torch.Tensor,
    prompt_len: torch.Tensor,
    block: int,
    generator: torch.Generator | None = None,
    valid_len: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (m, m_complement) boolean masks [B, L]; True = noised in that view.

    prompt_len: [B] number of leading clean (prompt) tokens per row; those positions are False
    in both views. valid_len: [B] positions >= valid_len are padding: never noised, never scored.
    Ratio t is drawn per (row, block) so the model sees every noise level within one sequence,
    as block-diffusion training does.
    """
    B, L = x0.shape
    nb = L // block
    t = torch.rand(B, nb, 1, generator=generator, device=x0.device)
    u = torch.rand(B, nb, block, generator=generator, device=x0.device)
    m = (u < t).reshape(B, L)
    pos = torch.arange(L, device=x0.device)[None, :]
    scorable = pos >= prompt_len[:, None]
    if valid_len is not None:
        scorable &= pos < valid_len[:, None]
    m = m & scorable
    mc = (~m) & scorable
    return m, mc


def apply_mask(x0: torch.Tensor, m: torch.Tensor, mask_id: int) -> torch.Tensor:
    xt = x0.clone()
    xt[m] = mask_id
    return xt


def masked_ce(logits_t: torch.Tensor, x0: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy over masked positions only. Returns (loss, n_targets).

    n_targets == 0 (every row fully prompt, or t drew ~0) makes the loss undefined; the caller
    must skip the step rather than average a zero — an averaged nothing is a silent green.
    """
    n = int(m.sum())
    if n == 0:
        return logits_t.new_zeros(()), 0
    return F.cross_entropy(logits_t[m].float(), x0[m]), n


def complementary_step_inputs(x0, prompt_len, block, mask_id, generator=None, valid_len=None):
    """Both views stacked along batch: returns (x0_2B, xt_2B, m_2B)."""
    m, mc = sample_complementary_masks(x0, prompt_len, block, generator, valid_len)
    xt_a, xt_b = apply_mask(x0, m, mask_id), apply_mask(x0, mc, mask_id)
    return torch.cat([x0, x0], 0), torch.cat([xt_a, xt_b], 0), torch.cat([m, mc], 0)


def masked_ce_from_hidden(hidden_t: torch.Tensor, lm_head, x0: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Same loss as `masked_ce`, projecting only the masked positions through lm_head."""
    n = int(m.sum())
    if n == 0:
        return hidden_t.new_zeros(()), 0
    return F.cross_entropy(lm_head(hidden_t[m]).float(), x0[m]), n
