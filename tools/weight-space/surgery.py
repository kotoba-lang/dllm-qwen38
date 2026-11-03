#!/usr/bin/env python3
"""Apply explicit projection edits to a copy of shared-expert output weights."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

import gguf
import numpy as np

from bench import digest


def clone_new(source, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        clone = library.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(output), 0) == 0:
            output.chmod(0o600)
            return "APFS clonefile"
        code = ctypes.get_errno()
        if code not in (errno.ENOTSUP, errno.EXDEV, errno.ENOSYS):
            raise OSError(code, os.strerror(code), str(output))
    with source.open("rb") as src, output.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return "exclusive byte copy"


def checksums(path, excluded):
    whole, outside = hashlib.sha256(), hashlib.sha256()
    offset = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            whole.update(chunk)
            view = memoryview(chunk)
            cursor, end = offset, offset + len(chunk)
            for start_skip, end_skip in excluded:
                if end_skip <= cursor or start_skip >= end:
                    continue
                outside.update(view[cursor - offset:max(cursor, start_skip) - offset])
                cursor = min(end, end_skip)
            outside.update(view[cursor - offset:])
            offset = end
    return whole.hexdigest(), outside.hexdigest()


def apply(source, expected_sha256, output, spec_path):
    source, output, spec_path = map(pathlib.Path, (source, output, spec_path))
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("output must be a new checkpoint path")
    if digest(source) != expected_sha256:
        raise ValueError("source checksum mismatch")
    spec = json.loads(spec_path.read_text())
    if not spec.get("hypothesis") or not spec.get("edits"):
        raise ValueError("explicit hypothesis and edits are required")
    reader = gguf.GGUFReader(source, "r")
    tensors = {t.name: t for t in reader.tensors}
    plans, names = [], set()
    supported = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                 gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0}
    for item in spec["edits"]:
        name = item["tensor"]
        if not re.fullmatch(r"blk\.\d+\.ffn_down_shexp\.weight", name) or name in names:
            raise ValueError("only distinct shared-expert down projections are supported")
        names.add(name)
        tensor = tensors[name]
        if len(tensor.shape) != 2 or tensor.tensor_type not in supported:
            raise ValueError("unsupported tensor shape or precision")
        path = (spec_path.parent / item["direction"]).resolve()
        direction = np.load(path, allow_pickle=False).astype(np.float32)
        if direction.shape != (int(tensor.shape[1]),) or not np.isfinite(direction).all():
            raise ValueError("direction must be a finite output-dimension vector")
        norm = float(np.linalg.norm(direction))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("direction norm must be finite and nonzero")
        strength = float(item["strength"])
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        plans.append((tensor, direction / norm, strength, digest(path)))
    journal_path = pathlib.Path(str(output) + ".edit.json")
    journal = {"state": "preparing", "source_sha256": expected_sha256,
               "spec_sha256": digest(spec_path), "hypothesis": spec["hypothesis"], "edits": []}
    with journal_path.open("x") as f:
        json.dump(journal, f, indent=2)
    try:
        journal["copy_method"] = clone_new(source, output)
        if source.stat().st_ino == output.stat().st_ino and source.stat().st_dev == output.stat().st_dev:
            raise ValueError("candidate must not be a hard link to the original")
        candidate = gguf.GGUFReader(output, "r+")
        candidate_tensors = {t.name: t for t in candidate.tensors}
        ranges = []
        for tensor, direction, strength, direction_hash in plans:
            weights = gguf.dequantize(tensor.data, tensor.tensor_type)
            if not np.isfinite(weights).all() or np.linalg.norm(weights) == 0:
                raise ValueError("source tensor must be finite and nonzero")
            before_projection = direction @ weights
            edited = weights - strength * np.outer(direction, before_projection)
            encoded = gguf.quantize(edited, tensor.tensor_type)
            dest = candidate_tensors[tensor.name]
            if encoded.shape != dest.data.shape or encoded.dtype != dest.data.dtype:
                raise ValueError("encoded tensor layout changed")
            dest.data[...] = encoded
            restored = gguf.dequantize(encoded, tensor.tensor_type)
            delta = restored - weights
            journal["edits"].append({"tensor": tensor.name, "type": tensor.tensor_type.name,
                "shape": tensor.shape.tolist(), "strength": strength, "direction_sha256": direction_hash,
                "parameters_touched": int(weights.size), "logical_values_changed": int(np.count_nonzero(delta)),
                "delta_l2": float(np.linalg.norm(delta)), "relative_delta_l2": float(np.linalg.norm(delta) / np.linalg.norm(weights)),
                "projection_norm_before": float(np.linalg.norm(before_projection)),
                "projection_norm_after": float(np.linalg.norm(direction @ restored)),
                "byte_range": [int(tensor.data_offset), int(tensor.data_offset + tensor.n_bytes)]})
            ranges.append((int(tensor.data_offset), int(tensor.data_offset + tensor.n_bytes)))
        candidate.data.flush()
        ranges.sort()
        source_hash, source_outside = checksums(source, ranges)
        output_hash, output_outside = checksums(output, ranges)
        if source_hash != expected_sha256 or source_outside != output_outside:
            raise ValueError("original changed or bytes outside declared tensors changed")
        journal.update(state="complete", candidate_sha256=output_hash,
                       unchanged_ranges_sha256=source_outside,
                       original_preserved=True, bytes_outside_declared_tensors_equal=True)
    except BaseException as exc:
        journal.update(state="incomplete", error=str(exc))
        raise
    finally:
        journal_path.write_text(json.dumps(journal, indent=2) + "\n")
    return journal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spec", required=True)
    args = p.parse_args()
    print(json.dumps(apply(args.source, args.source_sha256, args.output, args.spec), indent=2))


if __name__ == "__main__":
    main()
