"""Read bounded capture outputs, including valid empty microbatch outputs."""
import json
from pathlib import Path

import numpy as np


def load_vectors(directory, width=2048, prompts=8, layers=(31, 35, 39)):
    directory = Path(directory)
    status = json.loads((directory / "capture-status.json").read_text())
    calls = [json.loads(line) for line in (directory / "prefill-runs.jsonl").read_text().splitlines()]
    if (status["prefill_api_calls"] != prompts or status["generated_tokens"] != 0
            or status["process_exit"] or status["capture_failed"] or len(calls) != prompts
            or [r["prompt_index"] for r in calls] != list(range(prompts))
            or any(r["api_exit"] or r["first_layer_graph_visits"] < 1 for r in calls)
            or sum(r["first_layer_graph_visits"] for r in calls) != status["first_layer_graph_visits"]):
        raise ValueError("capture call/graph accounting failed")
    vectors, empty = {}, 0
    expected = {(p, f"l_out-{layer}") for p in range(prompts) for layer in layers}
    for row in map(json.loads, (directory / "tensors.jsonl").read_text().splitlines()):
        key = row["prompt_index"], row["tensor"]
        shape = row["shape"]
        if key not in expected or len(shape) != 4 or shape[0] != width or shape[2:] != [1, 1]:
            raise ValueError("unexpected residual layout")
        values = np.fromfile(directory / row["file"], dtype="<f4")
        if values.nbytes != row["bytes"] or values.size != int(np.prod(shape)):
            raise ValueError("residual byte count mismatch")
        if values.size == 0:
            empty += 1
            continue
        if not np.isfinite(values).all():
            raise ValueError("nonfinite residual")
        vectors[key] = values.reshape(-1, width)[-1].copy()
    if set(vectors) != expected:
        raise ValueError("missing nonempty final residual")
    return vectors, {"capture_status": status, "empty_microbatch_outputs": empty,
                     "nonempty_final_vectors": len(vectors)}
