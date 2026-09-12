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
modal run modal_app.py::equivalence --model Qwen/Qwen3.8-27B --layers 16 --dtype float32   # H100
modal run modal_app.py::train_run --model Qwen/Qwen3.5-0.8B --steps 200                   # H100
```

`--layers N` truncates the stack *at load* (bf16 load, drop the tail, cast the prefix) so the 27B
fits one GPU in fp32 (ADR §3a: "27B の先頭数層を切り出して 1 GPU 上で"). `--oracle-kernels
reference|fla` (Modal) selects whether the HF oracle runs its torch reference or fla's Triton
kernel; the report names which. With `--model` the report also prints how many embedding rows
are unused by the tokenizer — the `[MASK]` row question the ADR left unverified.

## Measured (see `reports/`)

### §3a equivalence

| model | layers | L | dtype / oracle GDN | oracle Δ (x_t / x_0) | ignores x_t k−1 / x_0 k | reacts to x_0 k−1 | faults detected | where |
|---|---|---|---|---|---|---|---|---|
| tiny random Qwen3.5 config | 8 (6+2) | 128 | fp32 / torch | 8.3e-7 / 8.3e-7 | 0 / 0 | 0.42 | 3/3 (0.39, 0.58, 0.39) | M1 Max CPU, 2.7 s |
| `Qwen/Qwen3.5-0.8B` (`2fc06364`) | 24 (18+6) | 128 | fp32 / torch | 7.1e-5 / 4.8e-5 | 0 / 0 | 4.5 | 3/3 (17.1, 6.1, 3.7) | M1 Max CPU, 702 s |
| **`Qwen/Qwen3.8-27B`** (`1d4bf0f2`) | **16 (12+4)** | 128 | **fp32 / torch** | **3.6e-5 / 1.9e-5** | **0 / 0** | 2.5 | **3/3 (11.5, 1.47, 12.1)** | Modal H100, 2026-09-12 |
| `Qwen/Qwen3.8-27B` | 64 (48+16) | 128 | bf16 / torch | 0.59 / 0.35 | 0 / 0 | 9.3 | 3/3 (24.8, 14.9, 18.3) | Modal H100 |
| `Qwen/Qwen3.8-27B` | 64 (48+16) | 128 | bf16 / fla Triton | 1.31 / 0.38 | 0 / 0 | 9.3 | 3/3 (24.7, 14.9, 18.3) | Modal H100 |

The bold row is the ADR's §3a on the 27B itself, at fp32 tolerance 1e-3: the state fork agrees
with the unmodified HF model to 3.6e-5 and every fault is caught. The two bf16 rows do **not**
pass tol 0.25 — even the *clean* stream (same math, chunk 32 vs HF chunk 64) drifts 0.35 through 64
bf16 layers, so bf16 is not a precision at which this test can be read; the structural checks are
still exact zeros and the faults sit 25–40× above the drift. fla's Triton kernel vs HF's torch
reference adds another ~0.7 on the noised stream in bf16. fp32 on all 64 layers is 108 GB and
does not fit one H100.

Spare vocab rows (`[MASK]` needs no resize): 243 on both the 0.8B and the **27B** (248,320 rows,
248,077 tokenizer entries — 27B measured on Modal).

### Phase 2 loop smoke (Modal H100, `Qwen/Qwen3.5-0.8B`, Nemotron `SFT/chat`, 2026-09-12)

200 steps, batch 4 × 512 (×2 complementary views), block 32, lr 1e-5, bf16 autocast, full
fine-tune: masked CE **11.2 → 3.95** (mean of last 10: 4.32), 0 skipped steps, **2,576
sequence-tokens/s**, peak **53.1 GiB**. That throughput is the *reference torch kernels + eager
attention* baseline, not the number to extrapolate the 27B from — the levers are measured
below. batch 8 at 512 OOMed an 80 GB H100 before the loss stopped materialising the
full `[B, L, 248k]` logits (now only masked positions are projected).

### Throughput levers (Modal H100, `Qwen3.5-0.8B`, 50 steps per leg, fresh weights per leg)

One lever at a time: baseline→sdpa isolates the attention backend, sdpa-torch→fla-sdpa isolates
the GDN core, ckpt adds per-layer gradient checkpointing, b2x doubles the batch under the freed
memory. Same 50-step shape as the loop smoke (batch 4 × 512 × 2 views, block 32, lr 1e-5, bf16).

| leg | seq-tok/s | peak GiB | loss | note |
|---|---|---|---|---|
| baseline (eager + torch GDN core) | 2987 | 52.7 | 11.22→4.56 | reference kernels |
| +sdpa attention | 2967 | 51.0 | 10.74→4.94 | ±0 speed, −1.8 GiB |
| +fla GDN core | 798 | 35.9 | 10.33→4.57 | −3.7× speed, −15 GiB |
| sdpa + torch core + ckpt | 2019 | **10.1** | 11.18→4.79 | −32% speed, −42.6 GiB |
| …ckpt, batch ×2 | **2970** | **15.9** | 10.47→4.44 | baseline speed at ⅓ the memory |
| fla + sdpa + ckpt | 1598 | 10.1 | 10.93→4.60 | the other core under ckpt |

- **fla Triton kernel vs torch reference** (real mixer module, bf16, with per-block states,
  `gdn.kernel_diff`): max Δ out **1.5e-3**, states **4.9e-3** — bit-identical across two runs.
  The ADR's fla-vs-reference equivalence question is closed at bf16 tolerance.
- **The lever that pays is checkpointing + batch**: ckpt alone trades 32% speed for 42.6 GiB;
  giving the freed memory to batch ×2 returns to baseline throughput (2970 vs 2987) at
  15.9 GiB, and reaches the best loss of all legs. This is the run shape the 27B extrapolation
  should assume, not the baseline row.
- fla core is slow in *this* fork (per-block sequential calls for per-block states); it is
  recorded, not adopted. fla+ckpt measured faster than fla alone (1598 vs 798) — allocator
  state, do not read as a lever.
- Run-to-run throughput variance on one H100 node is ±20% (baseline 2498 vs 2987 on two runs
  of the same code); losses are seed-deterministic and identical across runs. Compare legs
  within a run, not across runs.
- **The first ckpt measurement was invalid and that failure is now part of the record**: the
  checkpointed function closed over the layer-loop variable, so recompute — which runs at
  backward time, after the loop has advanced — re-ran the *last* layer for every checkpoint.
  The forward stayed exactly right (`torch.equal`) while grads reached 13/108 parameters
  (embed ~3× over, last mlp ~15× over) and 50-step loss moved 11.18→10.95 (b2x: loss *up*).
  It was also the cause of the non-reentrant saved-tensor ledger mismatch (219 vs 74 on CPU
  with the torch core, 265 vs 77 on H100 with the fla kernel) — never about fla.
  `functools.partial` fixes it; non-reentrant mode is kept deliberately because its ledger
  check throws on this class of silent divergence where reentrant returned wrong gradients
  without an error. After the fix, plain and checkpointed loss trajectories and grad norms are
  bitwise identical (tiny model, 50 steps: 5.769→2.942, layer-0 mlp grad-norm 0.10651702… both).
  The strengthened test pins per-parameter grad equality against the plain path and a
  100-step learning check.

**Not measured**: any 27B training step (needs FSDP), decoding quality on any trained
checkpoint, `causal-conv1d` in the image (reference conv fallback is on every leg; candidate
lever, nvcc build risk).

## Objective, decoder, training loop (Phase 2 / 3 mechanics)

- `objective.py` — complementary masking: one mask m per sequence (ratio t drawn per block), both
  views x_t^(m) and x_t^(1−m) trained, so every response token is a target once per step; masked-
  token-only cross-entropy; prompt and padding never noised, never scored; a step with zero
  targets is *skipped and counted*, not averaged into a zero. `[MASK]` = the first embedding row
  the tokenizer does not use.
- `decode.py` — the Phase 3 **reference** decoder: block-wise, all-[MASK] start, commit every
  position with confidence ≥ threshold (always ≥ 1), optional left-to-right sub-blocks, no cache
  (every step re-runs the full forward — golden by construction, slow by design).
- `train.py` — single-device full fine-tune on the state-fork forward; `--data synthetic` is the
  loop's own test (a deterministic token rule the tiny model must learn and the decoder must then
  reproduce), `--data nvidia/Llama-Nemotron-Post-Training-Dataset` renders rows through the chat
  template (`input` there is a Python-literal string of messages, not JSON — measured).
- `modal_app.py` — GPU runs on Modal (owner directive): `equivalence` (§3a on the 27B itself) and
  `train_run` (loop smoke + seq-tokens/s). Volume `dllm-qwen38-cache`, secret `hf-token`.

Local end-to-end on the tiny model (CPU, 1 thread — macOS grouped-conv1d backward oversubscribes
at 8 threads, 13× slower): 300 steps, batch 8, L 64, block 8: masked CE **5.74 → 0.0035**; reference
decoder on held-out prompts: **rule accuracy 0.984**, **5.33 tokens per forward** at threshold 0.9
(block 8). That closes objective → forward → decoder mechanically; it says nothing about language.

## Not here yet

FSDP for the 27B (Phase 2 at scale), the llama.cpp block-diffusion decoder for `qwen35` and the
GGUF conversion (Phase 4), benchmarks (Phase 5). Python here is mechanism, not policy: decisions
live in the ADR, and the fleet-facing surface stays in `murakumo`.

License: Apache-2.0. `gdn.py` derives from HuggingFace `transformers` (Apache-2.0).
