# Local weight-space bench (JUN-797)

This is a development bench for Qwen3.6-35B-A3B and Clover-35B-A3B-Preview. It makes a short experiment repeatable, keeps the original response, refuses checkpoint mismatches, and resumes after the last completed problem. It does not train a model or establish improvement on a final test set.

The first lane is **quantized, greedy, non-thinking, causal decode**. It is a diagnostic lane, distinct from the BF16/thinking and Clover block-denoising/self-speculative qualifications in the parent Linear plan. A shorter token cap is a stress condition, not evidence of better reasoning. The same script can run thinking on as a separate condition.

## Prepare once

Only Python's standard library is required. `prepare` downloads GSM8K **train**, pinned to a source commit. It sorts questions by SHA256 and creates disjoint dev and validation sets of 32 problems each. Files are created exclusively; it never replaces an existing dataset. Keep final benchmark test sets outside the editor's working directory.

```sh
python3 tools/weight-space/bench.py prepare --output /absolute/evidence/data --size 32
python3 tools/weight-space/test_bench.py
```

An optional harder development lane uses only MATH train problems at level 4/5 with one plain integer boxed answer. It downloads no test split and uses the same runner/scorer. This selected subset does not represent full MATH accuracy. Its preparation needs `pyarrow==22.0.0`; inference and comparison still use only the standard library.

```sh
/absolute/task-venv/bin/python -m pip install pyarrow==22.0.0
/absolute/task-venv/bin/python tools/weight-space/prepare_math.py \
  --output /absolute/evidence/math-data --size 32
```

The source is `EleutherAI/hendrycks_math` at revision `21a5633873b6a120296cce3e2df9d5550074f4a3`; output files record the checksum of each downloaded train shard. Both lanes are chosen before looking at model scores.

## Start the pinned engine

Build a llama.cpp revision compatible with the exact artifact. For the official Clover GGUF at revision `32e22a02b885e173177b7ce7661ad13448b4f89e`, this experiment uses the earlier full PR implementation `cb09d4355d8bf968bb907c5de03cf42d8aa073ce`. Main `749db7f034d43d6de01a01f6d8c872d02afa6c7b` has AR-only support and rejects that GGUF: its metadata lacks `expert_shared_count`, and it has 120 diffusion tensors that this loader does not consume. An override for the missing count alone did not repair the tensor-count mismatch. Preserve those failures separately from inference scores. Record build options, version, launch command, and logs. Use only the selected checkpoint; do not run multiple large checkpoints concurrently on a memory-constrained host.

```sh
/absolute/llama-server -m /absolute/model.gguf --host 127.0.0.1 --port 17977 \
  -c 2048 -np 1 -b 256 -ub 128 -ngl 99 --cpu-moe -t 6 -tb 6 \
  --flash-attn on --no-warmup
```

The memory profile above keeps routed expert weights on CPU. Confirm that each model/backend actually supports the requested configuration. Store any changes as a new experiment condition. A GPU-memory or backend error is an unavailable run, never a zero score.

## Run and resume

Verify the download against the distributor's SHA256 first. Pass that same hash to the runner. It hashes the local checkpoint again and verifies the server's `/props` model path. The request uses raw ChatML with an explicitly closed thinking prefix when thinking is off. Templates and greedy settings are experiment conditions, not an assertion that this reproduces the official model-card benchmarks.

```sh
python3 tools/weight-space/bench.py run \
  --data /absolute/evidence/data/dev.json \
  --model /absolute/model.gguf --model-sha256 DISTRIBUTOR_SHA256 \
  --backend-commit cb09d4355d8bf968bb907c5de03cf42d8aa073ce \
  --condition baseline-causal-q4 --thinking off --max-tokens 1024 \
  --output /absolute/evidence/runs/baseline
```

Use `--limit 4` and a separate output directory for an initial smoke test. Omit it for the fixed 32-problem lane. Repeating the exact command resumes completed rows. Checkpoint, dataset, settings, script, and case-list changes require a new output directory. No request goes to a remote inference service; only explicit loopback HTTP endpoints are accepted.

The scorer requires exactly one `FINAL: <number>` line outside thinking. An unfinished thinking trace or ambiguous answers fail extraction. This strict formatting is part of the diagnostic, so report extraction failures separately from wrong numeric answers. Repeated four-gram fraction and output-limit hits are separate fields; neither is called a reasoning loop automatically.

For repeated experiments, `python3 tools/weight-space/launch.py /absolute/experiment.json` manages the local engine lifetime, waits for health, runs/resumes the bench, and stops only its own child process. The JSON contains the `engine`, `engine_sha256`, `backend_commit`, `model`, `model_sha256`, `engine_args` array, `data`, `output`, `condition`, and optional `limit`, `thinking`, `max_tokens`, `request_timeout`, and `port` fields. It records launch identity, engine logs, and completion/failure status in the output directory. A checksum mismatch, occupied port, or changed launch config fails before inference. Use a new output directory for a new condition.

## Compare an edit

Keep the original checkpoint immutable. Store the edited checkpoint, patch, changed tensor/axis/range, parameter count, norm, and before/after checksums separately. Reload the engine with the candidate. Use the same dataset/settings/backend/hardware and a new run directory.

```sh
python3 tools/weight-space/bench.py compare \
  /absolute/evidence/runs/baseline /absolute/evidence/runs/candidate \
  --output /absolute/evidence/paired-comparison.json
```

Comparison requires complete, matching case lists and generation conditions. It reports paired wins/losses, a problem-level bootstrap interval, exact McNemar p, tokens, and wall time per correct answer. A 32-problem development experiment cannot establish the parent's +3 percentage point improvement target. Select a candidate on validation, freeze it, then use the isolated final evaluation and regression tasks in JUN-802. A decode-only comparison has different generation settings and must be reported separately rather than passed off as a paired weight-only change.

All raw run files and downloaded weights stay outside Git under the task evidence workspace. Commit only code and compact reports with no private inputs. Public GSM8K questions remain under their source license. This bench adds no CI, service, schedule, or deployment.

## Explicit projection-edit primitive

`surgery.py` is an experimental editing tool, not a successful edit recipe. It requires the pinned `gguf` library from the engine source and NumPy. A spec records a hypothesis and an `edits` array containing `tensor`, `direction` (a local `.npy` vector), and `strength` in [0,1]. It supports only distinct shared-expert down-projection tensors with F32/F16/Q4_0/Q8_0 storage. Directions must come from a documented development-only observation; weight statistics alone do not establish a behavioral cause.

```sh
/absolute/task-venv/bin/python tools/weight-space/surgery.py \
  --source /absolute/original.gguf --source-sha256 VERIFIED_SHA256 \
  --spec /absolute/explicit-edit.json --output /absolute/new-candidate.gguf
/absolute/task-venv/bin/python tools/weight-space/test_surgery.py
```

The operation is `W_new = W - strength * d * (d^T W)`, with a unit direction `d` in the residual output dimension. The tool clones or exclusively copies the original, preserves quantized layouts, re-encodes only declared tensors, and verifies the original checksum and every byte outside declared tensor ranges. Its journal records actual quantized deltas and before/after checksums. Use strength zero for a byte-preserving no-op. To measure encoding drift separately, set `requantize_only: true` with strength zero in an edit spec. A valid externally quantized tensor need not be a fixed point of this encoder: the published Clover Q4_0 tensor changed under the former zero-strength re-encoding path. Match random-direction controls by their observed delta norm and report the reference checkpoint explicitly. Tests use tiny artificial GGUFs and prove mechanics, not model improvement. Integrity checks alone do not establish a target-model performance gain.

`capture.cpp` is a bounded observation helper built against the **same headers and libraries** as the pinned engine. It uses the public evaluation callback API, performs one explicit prefill call per input (which can contain several microbatch graphs), generates no tokens, and dumps only F32 residual tensors `l_out-31`, `l_out-35`, and `l_out-39`. It also counts first-layer normalization graph visits rather than confusing a prefill API call with one graph execution. `WSP_CAPTURE_DIR` must be an existing empty directory. Optionally, `WSP_CAPTURE_LIST` names a file containing at most eight local prompt-file paths, one per line. Each prompt is limited to 64 KiB and 4096 tokens. Cache state is cleared between inputs. Capture status, per-input counts, tensor shapes, prompt indexes and raw `.f32` files remain in the output directory. Use development prompts/prefixes only, count these observations against the forward budget, and verify file hashes and finite values before deriving a direction. The last tensor column is the final position within each microbatch; preserve callback and prompt indexes so that positions are not mixed.

For a source directory `ENGINE_SOURCE` and its matching build directory `ENGINE_BUILD`, compile with C++17, include `common`, `include`, and `ggml/include` from that source, and link the build's `libllama-common`, `libllama`, `libggml`, and `libggml-base` dynamic libraries. On macOS add `-Wl,-rpath,ENGINE_BUILD/bin`. This helper adds no code to llama.cpp itself. Its initial functional check used an artificial 40-layer, hidden-width-8 CPU fixture and captured all three finite residual vectors; artificial fixture checks alone do not qualify target MoE/GDN captures.

Use `capture_data.load_vectors()` to read the last nonempty residual for each prompt/layer. Early microbatches can legitimately have zero selected output rows; their empty tensors are counted rather than indexed as final vectors. Missing nonempty finals, inconsistent byte counts, unsupported layouts, nonfinite values, or API/graph accounting mismatches fail closed. The first real Qwen observation collected eight prefill API calls, 50 first-layer graph visits, 6139 input tokens, and 24 final vectors; the identical-prefix reset controls matched exactly. This qualifies that observation path only, not a causal edit or performance gain.

The first Qwen selected-development screen used eight GSM8K cases, including the three direction-source cap hits. Original, zero-strength sham, layer39 strength0.25 projection and norm-matched random direction all scored 3/8 with 4383 generated tokens and three cap hits. The sham response text matched the original exactly. A separate concise-prompt control scored 7/8 with 668 tokens and no cap hits. This does not support promoting that weight candidate; the prompt result is exploratory and is not a weight-space gain. Full identities, paired results and integrity journals are in [the Qwen pilot report](../../reports/weight-space-qwen-pilot-20261103.json). Validation and final tests were not used.

To reproduce a previously declared direction from saved captures, use `derive_direction.py`. Select the layer, pair indexes and consistency gate before looking at candidate scores. The command performs no inference, refuses a changed reset control, records every capture-file checksum, and writes no direction when the gate fails. Use a fresh output directory:

```sh
/absolute/task-venv/bin/python tools/weight-space/derive_direction.py \
  --capture /absolute/evidence/observation/capture \
  --output /absolute/evidence/rederived-direction \
  --layer 39 --pairs 0:1 2:3 4:5 --reset 6:7 --min-mean-cosine 0.05
```

This command reproduces both saved target-pilot `.npy` direction hashes exactly; it does not add observations or demonstrate improvement.

Use `collect.py` to produce a bundle for a complete, managed multi-condition experiment. It validates all case lists and stored summaries, recorded engine shutdown, dataset checksums, scorer revisions and paired engine profiles. It refuses incomplete runs and files that change during collection; it performs no inference or regrading. A plan names run directories relative to itself:

```json
{
  "conditions": [
    {"name": "original", "run": "runs/original"},
    {"name": "no-op", "run": "runs/no-op"},
    {"name": "candidate", "run": "runs/candidate"}
  ],
  "paired": [["original", "no-op"], ["original", "candidate"]],
  "no_op": [["original", "no-op"]],
  "qualification": "selected development diagnostic; not held-out improvement"
}
```

```sh
python3 tools/weight-space/collect.py /absolute/evidence/collection-plan.json \
  --output /absolute/evidence/completed-bundle.json
python3 tools/weight-space/test_collect.py
```

The collector adds scorer-revision and launch-profile checks beyond the current two-run `compare` command. Keep the recorded datasets and launch files available. Whole checkpoint files are not rehashed at collection time; their verified launch identities and artifact checksums are retained. A no-op entry records whether checkpoint hashes and response text agree, without counting a repeated answer as a new independent problem.

`meter.cpp` is a separate bounded causal-generation observer. Build it against the same pinned headers/libraries as `capture.cpp`. It forces a fresh greedy sampler (temperature0, top-k1, seed42, neutral penalties), ignores sampler CLI overrides, disables warmup, clears memory for each input, rejects adapters/speculative/embedding modes, and requires an explicit output cap1..1024. `WSP_METER_LIST` optionally names up to eight local prompt files (each at most64KiB/4096 tokens); otherwise the raw CLI prompt is used. `WSP_METER_DIR` must be an existing empty directory. Verify checkpoint and binary/library checksums before invocation.

```sh
WSP_METER_DIR=/absolute/evidence/meter-output \
WSP_METER_LIST=/absolute/evidence/prompt-files.txt \
  /absolute/meter-binary -m /absolute/verified-model.gguf \
  -c 4096 -b 512 -ub 256 -ngl 99 -t 6 -tb 6 --flash-attn on --fit off -n 16
```

The helper saves raw content bytes, sampled token IDs (including EOG), per-call trace and per-input/overall counts. Prefill API calls, single-token decode API calls, sampled/emitted tokens and observed `attn_norm-0` graph visits are distinct fields. Each call is synchronized and times host wall duration; this is not GPU-seconds. Any call without the expected graph marker or graph outside a declared call invalidates completion. Its artificial CPU fixture checks with both pinned SDKs used13 prefill calls/23 graphs plus9 decode calls/9 graphs, while emitting12 tokens across three inputs; identical-input replay matched. Caps outside1..1024, speculative mode and nonempty output were rejected. These checks do not qualify target MoE/GDN generation or equivalence to the HTTP server. The callback/synchronization also changes the timing profile. See [the meter fixture report](../../reports/weight-space-native-meter-fixture-20261104.json).

The completed Clover hard-integer MATH selected-development screen scored2/8 original,2/8 true no-op,3/8 encoding-only,2/8 projection and3/8 norm-matched random. The no-op response text matches on all eight inputs. The projection has zero paired wins and one loss vs encoding-only. A separate concise-prompt condition scores7/8 with3390 tokens and no cap hits, but predicts614 instead of240 on one geometry case. The frozen projection candidate is not promoted to final evaluation; separately fixed validation measures its independent effects without further tuning. Full provenance and limitations are in [the Clover pilot report](../../reports/weight-space-clover-pilot-20261104.json).

`uncertainty.py` provides a supplementary conservative interval without changing frozen bootstrap/McNemar outputs. For independent identically distributed problem pairs, win and loss counts have binomial marginals. Compute each marginal interval at97.5% confidence and subtract endpoints; the Bonferroni inequality gives at least95% joint coverage without requiring independence between the two indicators. This uses [NIST binomial-tail inversion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) and the [general Bonferroni inequality](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm). It is not a tight exact paired-risk-difference interval or an across-model multiplicity adjustment.

```sh
python3 tools/weight-space/uncertainty.py /absolute/evidence/completed-bundle.json \
  --output /absolute/evidence/supplementary-uncertainty.json
python3 tools/weight-space/test_uncertainty.py
```

With no observed wins or losses, the empirical bootstrap can collapse to[0,0], while this construction gives approximately[-42.18pp,+42.18pp] at n8 and[-12.80pp,+12.80pp] at n32. Failure-selected development screens do not satisfy the population-sampling assumptions; these intervals illustrate finite-sample uncertainty and do not qualify regression absence. The primary analyses remain unchanged. Published examples, independent Decimal tail residuals, boundary formulas and a finite coverage grid are checked in the test. See [the supplementary report](../../reports/weight-space-uncertainty-diagnostics-20261104.json).


`symbolic_score.py` is a separate boxed-answer prototype for future fraction/expression evaluation. It does not change or regrade the running numeric benchmark. Install its five core wheels in a separate Python environment using `requirements-symbolic.lock` with `pip --require-hashes`; keep the active inference environment unchanged. It extracts the final balanced top-level `\boxed{...}` outside balanced thinking, rejects incomplete formats, and uses the pinned Math-Verify LaTeX parser without raw-string fallback. Explicit comparator settings are rounding12, precision30, strict comparison and no set/relation coercion. These settings do not define a uniform numeric tolerance or a formal equality proof. Internal grader failures return unavailable rather than a wrong-answer score.

Run `test_symbolic_score.py` in that environment for synthetic fractions, radicals, sets, expressions, incorrect near-values and thinking/box boundaries. The parser requires the main thread for POSIX alarms; each operation has its own five-second limit, and a separate process deadline is still needed for a hard whole-case budget. This prototype has not been qualified against full MATH, official scoring, or target-model symbolic outputs. No final dataset was fetched. See [the synthetic-only receipt](../../reports/weight-space-symbolic-grader-prototype-20261104.json).
