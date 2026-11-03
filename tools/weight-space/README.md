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

The operation is `W_new = W - strength * d * (d^T W)`, with a unit direction `d` in the residual output dimension. The tool clones or exclusively copies the original, preserves quantized layouts, re-encodes only declared tensors, and verifies the original checksum and every byte outside declared tensor ranges. Its journal records actual quantized deltas and before/after checksums. Test strength zero as a quantization sham before attributing any effect to an edit; match random-direction controls by their observed delta norm. Tests use tiny artificial GGUFs and prove mechanics, not model improvement. No target-model weight edit has been qualified yet.
