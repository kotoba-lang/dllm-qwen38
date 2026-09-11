# dllm-qwen38

**Qwen3.8-27B as a Fast-dLLM v2 block-diffusion LLM, and the road to a dLLM GGUF the murakumo
fleet can serve.** Subject-plane name (`kotoba-lang/dllm-qwen38`): the subject is "dLLM of
Qwen3.8", not a role and not an origin — the recipe is NVlabs' but the model adaptation is ours.
Plan and decisions: superproject ADR `adr-2609112640-qwen38-27b-fast-dllm-v2-block-diffusion`.

Nearest repos and the boundary: `kotoba-lang/murakumo` serves GGUFs (llama-server, `infer.edn`)
and is where the finished dLLM GGUF gets registered — nothing here serves; `kotoba-lang/vllm` is
a request definition against a vLLM server — nothing here is a server; `cloud-itonami/llm-dataset`
is where training data and checkpoints live (DataLad + B2) — nothing large is committed here.

## What is new, and what is not

Fast-dLLM v2 (arXiv 2509.26328) turns a pretrained AR model into a block-diffusion model with
~1B tokens of fine-tuning: noised sequence x_t and clean sequence x_0 are concatenated, a
complementary attention mask gives x_t bidirectional attention inside a block and access to the
clean blocks before it, and the loss is masked-token cross-entropy. The paper's base is Qwen2.5,
all layers softmax attention.

Qwen3.8-27B (and the whole Qwen3.5/3.6/3.8 dense family, llama.cpp arch `qwen35`) is a hybrid:
**48 of 64 layers are Gated DeltaNet**, a causal linear recurrence that cannot take an attention
mask. That is the only genuinely new part, and it is what this repo tests first.

### State fork (`src/dllm_qwen38/gdn.py`, `blockdiff.py`)

For a DeltaNet layer, run the clean stream once and keep the recurrent state and the conv tail at
every block boundary; run every noised block as its own batch row starting from the clean-prefix
state before it. Visibility is then exactly the paper's `M_OBC` (clean blocks `< k`) with no
`x_t → x_t` leakage across blocks. Inside a block the recurrence stays left-to-right, so the
bidirectional part is carried by the 16 attention layers only ("semi-bidirectional" — measured,
not assumed, in ADR §3b). Attention layers take the paper's `[2L, 2L]` mask literally.

### The equivalence test (`src/dllm_qwen38/equivalence.py`, ADR §3a)

Oracle = the **unmodified** HuggingFace model, run per block k on `cat([x_0 blocks < k, x_t block k])`
under a block-causal 4D mask. Its DeltaNet path is HF's own chunked kernel with no fork, so
agreement is evidence about the fork rather than a tautology. The test reports counts, not a
boolean, and refuses to pass unless every deliberate fault is caught:

| check | what |
|---|---|
| noised-stream-vs-oracle | max abs logit diff, candidate vs oracle |
| clean-stream-vs-oracle | the x_0 stream equals HF under the block-causal mask |
| xt-block-k-ignores-xt-block-k-1 | perturb x_t block k−1 → x_t block k logits unchanged |
| xt-block-k-ignores-x0-block-k | perturb x_0 block k → unchanged |
| xt-block-k-reacts-to-x0-block-k-1 | perturb x_0 block k−1 → must change by ≥ floor |
| fault `concat-causal-gdn` | naive port: DeltaNet over `cat([x_t, x_0])` as one causal sequence — must be red |
| fault `mask-leak` | x_t block k also sees x_t blocks < k — must be red |
| fault `no-conv-prefix` | state forked but the short conv starts cold — must be red |

Exit 0 only if all checks pass **and** all faults are detected; 1 otherwise; 2 = could not measure.

```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -q tests                       # tiny random model, seconds, no download
.venv/bin/python -m dllm_qwen38.equivalence --block 32 --nblocks 4            # tiny, block 32
.venv/bin/python -m dllm_qwen38.equivalence --model Qwen/Qwen3.5-0.8B --block 32 --nblocks 4 --json reports/x.json
.venv/bin/python -m dllm_qwen38.equivalence --model Qwen/Qwen3.8-27B --layers 8 --device cuda --dtype bfloat16
```

`--layers N` truncates the stack so the 27B run fits one GPU (ADR §3a: "27B の先頭数層を切り出して
1 GPU 上で"). With `--model` the report also prints how many embedding rows are unused by the
tokenizer — the `[MASK]` row question the ADR left unverified.

## Measured (see `reports/`)

2026-09-12, Apple M1 Max, CPU, float32, HF reference kernels (no fla / causal_conv1d), tol 1e-3:

| model | layers | L | oracle Δ (x_t / x_0) | ignores x_t k−1 / x_0 k | reacts to x_0 k−1 | faults detected | wall |
|---|---|---|---|---|---|---|---|
| tiny random Qwen3.5 config | 8 (6 DeltaNet + 2 attn) | 128 | 8.3e-7 / 8.3e-7 | 0 / 0 | 0.42 | 3/3 (0.39, 0.58, 0.39) | 2.7 s |
| `Qwen/Qwen3.5-0.8B` (`2fc06364`) | 24 (18 + 6) | 128 | 7.1e-5 / 4.8e-5 | 0 / 0 | 4.5 | 3/3 (17.1, 6.1, 3.7) | 702 s |

The 0.8B run also answers one of the ADR's unverified items for that checkpoint: 248,320 embedding
rows vs 248,077 tokenizer entries = **243 spare rows**, so `[MASK]` needs no embedding resize there.
The 27B tokenizer length has not been measured (same declared vocab size; not the same evidence).

**Not measured**: the 27B itself (bf16 download is 55.6 GB; `--layers 8 --device cuda` is the
intended run), any GPU, bf16 tolerances, fla kernels vs the reference path, training, decoding.

## Not here yet

Training loop (Phase 2), PyTorch reference decoder (Phase 3), llama.cpp block-diffusion decode for
`qwen35` and the GGUF conversion (Phase 4), benchmarks (Phase 5). Python here is mechanism, not
policy: decisions live in the ADR, and the fleet-facing surface stays in `murakumo`.

License: Apache-2.0. `gdn.py` derives from HuggingFace `transformers` (Apache-2.0).
