# Qwen3.8-27B dLLM: step 8,000 decision-point record

Recorded: 2026-09-21 (JST)

This is a closed interim operational record, not final publication evidence. It separates
completed measurements from incomplete results and possible future work. This work session was
closed on 2026-09-22 JST; production remains paused.

## Decision state

- Production training is intentionally paused at global step 8,000 by owner direction.
- Do not train step 8,001 or later without a new explicit owner decision after interim evaluation.
- Do not publish the step-8,000 GSM8K result to Hugging Face as final evidence.
- The earlier 32,000-step / 131,072,000-token target remains a possible continuation, not an
  active authorization.

## Production training: measured and verified

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3.8-27B` |
| Data | Nemotron post-training, chat split |
| Shape | global batch 8, sequence length 512, block 32 |
| Optimizer setting | learning rate `1e-5`, gradient checkpointing enabled |
| Hardware | Modal H200 x8 |
| Completed call | `fc-01M2ZX5X4S618SBGDAQTDE5TYH` |
| Checkpoint namespace | `dllm-27b-1pct` |
| Completed step | 8,000 |
| Unique sequence tokens | 32,768,000 |
| Loss | 2.3832 -> 1.9294 |
| Throughput | 1,233.12 unique sequence tokens/s |
| Rank consistency | `weight_spread=0` |

The step-8,000 distributed checkpoint and cadence fingerprint exist. Independent zero-training
verification call `fc-01M30NEPBVVYB0PK8NQPS41TX8` reloaded model and optimizer state on all eight
ranks and verified the fingerprints, embedded step `8000`, and embedded sequence-token count
`32768000` (`verified_all_ranks=true`). The production deployment has no active training task.

## Interim GSM8K evaluation: preliminary, incomplete

The sole evaluation call is `fc-01M30NRYDCXWENWQW1KMJAQA69` in deployment
`dllm-qwen38-interim-eval` / app `ap-elHc2Mt5j7piY64SITUKqR`.

Fixed protocol:

- checkpoint: `dllm-27b-1pct`, global step 8,000
- published 500-item subset fingerprint: `1946abaa36eb78e9`
- 4-shot, greedy, maximum 1,024 generated tokens
- dLLM threshold 0.918; no sub-block override
- hardware: one H100

Final persisted partial result before closing:

| Field | Preliminary measurement |
|---|---|
| Evaluated | 63 / 500 |
| Exact-match correct | 53 / 63 |
| Accuracy | 0.8412698413 |
| Sum of per-item inference wall time | 14,274.7 s |
| Mean / median / p95 item latency | 226.58 s / 94.0 s / 852.2 s |
| Prompt / generated tokens | 44,279 / 23,934 |
| Partial JSONL SHA-256 | `7ef658955db8d0f48dc4bfbbb9c3b7e8f04be6c7dabf23ea0df06421128e6f2d` |

This accuracy is not a decision-quality 500-item result and must not be compared as though it
were the fixed 500-item score. The evaluator reports that
`causal_conv1d` is absent and is using the correct but much slower PyTorch reference fallback.
The call stopped making progress after the 63rd persisted row and had no active GPU container at
closing. The append-only partial JSONL remains in Modal volume `dllm-qwen38-cache` at
`evals/dllm-n500-fp1946abaa36eb78e9-cfg1457a4eeac53a2df.jsonl`. The unresolved call was cancelled
as part of closing; no replacement or duplicate evaluation was launched.

## MFU co-scientist: isolated evidence

These experiments use the isolated branch/worktree and never write the production checkpoint.

| Hypothesis | Result | Decision |
|---|---|---|
| Enable TF32 | 1,296.633 -> 1,359.002 tok/s; proxy MFU 10.6195% -> 11.1303%; 1.0481x | Rejected: last-10 loss mean differed about 0.872%, beyond the 0.5% gate, and required numerical/gradient/checkpoint gates were absent |
| Disable gradient checkpointing | 1,411.43 -> 235.84 tok/s; proxy MFU 11.56% -> 1.93%; 0.167x; peak 127.26 GiB with allocator OOM warnings | Rejected: loss, performance, and memory gates failed |
| Compiled optimized causal-conv1d | Reference 1,301.56 tok/s / 10.6599% proxy MFU; optimized 1,332.08 tok/s / 10.9099%; 1.02345x; both 51.63 GiB. Compiled `causal_conv1d_cuda` 1.7.0 was active and the kernel output/gradient probe passed, but paired-loss max absolute difference was 0.12895 and the required speedup was 1.08x | Rejected: paired-loss and performance gates failed |
| FSDP true-root topology | Candidate code and fail-closed gates committed in isolated branch at `64f2b6e`; paired H200 x8 call `fc-01M313EZJT0RMZNP68PH8NP6VY` remained capacity-waiting and never acquired a container | Unmeasured; call cancelled at closing, do not infer a speedup |

No rejected lever is combined into production. The causal-conv1d result is useful negative
evidence: the compiled backend was genuinely active and numerically close in the isolated kernel
probe, but that was insufficient to preserve the paired training trajectory or produce a material
end-to-end speedup. The next candidate is an isolated FSDP communication-overlap/root-topology A/B.

## Current billing boundary

- Production H200 x8 training: stopped; no active production GPU task.
- Interim evaluation: stopped/cancelled with 63 persisted rows; no active GPU task.
- MFU experiments: completed causal-conv1d call is rejected; capacity-waiting FSDP call cancelled;
  no active GPU task at closing.

This section records workload state, not a reconciled invoice or exact spend.

## Decisions required before continuation

1. Choose how to complete the fixed 500-item evaluation: a resumable/chunked continuation from
   the audited JSONL, or a separately validated optimized evaluator. Do not silently mix decoding
   backends in one reported result.
2. Review the completed 500-item interim accuracy and inference-performance report.
3. Explicitly decide whether production training may continue beyond step 8,000.
4. Only after an accepted MFU experiment passes all gates, decide whether to change the production
   training implementation or hardware.

## Closing disposition

- Preserve the fingerprint-verified step-8,000 production checkpoint.
- Preserve the 63-row partial evaluation only as incomplete diagnostic evidence.
- Do not resume training, evaluation, or MFU allocation automatically.
- Any future session must obtain an explicit owner decision, create a resumable bounded evaluator,
  and re-establish the exact fixed-subset protocol before claiming a 500-item result.
