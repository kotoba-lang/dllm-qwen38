"""Phase 3 reference decoder (ADR-2609112640): block-wise, confidence-threshold parallel decode.

This is the *golden* implementation, not the fast one: every denoising step re-runs the full
`block_diffusion_forward` over prompt + generated tokens, so there is no cache to get wrong.
Its temperature-0 token stream is what the llama.cpp block-diffusion decoder must reproduce.

Per block: start from all-[MASK], run the forward, take argmax + confidence (softmax max) at
the still-masked positions, commit every position whose confidence >= threshold and always at
least the single most confident one, repeat until the block is clean. `sub_block` (paper: 8)
restricts each step to the leftmost sub-block that still holds a mask, giving the paper's
left-to-right sub-block order; None decodes the whole block at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .blockdiff import block_diffusion_forward


@dataclass
class DecodeTrace:
    tokens: list[int]
    forwards: int = 0
    generated: int = 0
    steps_per_block: list[int] = field(default_factory=list)

    @property
    def tokens_per_forward(self) -> float:
        return self.generated / self.forwards if self.forwards else 0.0


@torch.no_grad()
def generate(
    text_model,
    lm_head,
    prompt_ids: list[int],
    *,
    n_blocks: int,
    block: int,
    mask_id: int,
    threshold: float = 0.9,
    sub_block: int | None = None,
    eos_id: int | None = None,
    temperature: float = 0.0,
    layers: int | None = None,
) -> DecodeTrace:
    device = text_model.embed_tokens.weight.device
    cur = list(prompt_ids)
    trace = DecodeTrace(tokens=cur)
    for _ in range(n_blocks):
        n = len(cur)
        kb = n // block
        L = (kb + 1) * block
        known = torch.full((L,), mask_id, dtype=torch.long, device=device)
        known[:n] = torch.tensor(cur, device=device)
        masked = torch.zeros(L, dtype=torch.bool, device=device)
        masked[n:] = True
        x0 = known.clone()  # block kb of x_0 is invisible to x_t block kb; its content is irrelevant
        steps = 0
        while bool(masked.any()):
            xt = known.clone()
            xt[masked] = mask_id
            lt, _ = block_diffusion_forward(text_model, lm_head, x0[None], xt[None], block, layers=layers)
            trace.forwards += 1
            steps += 1
            logits = lt[0, kb * block : L].float()
            if temperature > 0:
                probs = torch.softmax(logits / temperature, -1)
                pick = torch.multinomial(probs, 1)[:, 0]
                conf = probs.gather(1, pick[:, None])[:, 0]
            else:
                probs = torch.softmax(logits, -1)
                conf, pick = probs.max(-1)
            cand = masked[kb * block : L].clone()
            if sub_block:
                # leftmost sub-block that still has a mask
                first = int(torch.nonzero(cand)[0])
                s0 = (first // sub_block) * sub_block
                window = torch.zeros_like(cand)
                window[s0 : s0 + sub_block] = True
                cand &= window
            conf = conf.masked_fill(~cand, -1.0)
            commit = cand & (conf >= threshold)
            if not bool(commit.any()):
                commit[int(conf.argmax())] = True
            idx = torch.nonzero(commit)[:, 0] + kb * block
            known[idx] = pick[idx - kb * block]
            masked[idx] = False
        trace.steps_per_block.append(steps)
        new = known[n:L].tolist()
        cur.extend(new)
        trace.generated += len(new)
        if eos_id is not None and eos_id in new:
            break
    trace.tokens = cur
    return trace
