#!/usr/bin/env python3
"""Pinned hard MATH-train integer-answer subset; never downloads test."""
import argparse
import hashlib
import io
import json
import pathlib
import re
import urllib.request

import pyarrow.parquet as parquet

from bench import canonical, write_new

REPO = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=32)
    args = p.parse_args()
    if args.size < 1:
        raise ValueError("size must be positive")
    sources, cases = [], []
    seen = set()
    for subject in SUBJECTS:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{subject}/train-00000-of-00001.parquet"
        raw = urllib.request.urlopen(url, timeout=60).read()
        sources.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest()})
        for row in parquet.read_table(io.BytesIO(raw)).to_pylist():
            if row["level"] not in ("Level 4", "Level 5"):
                continue
            answers = re.findall(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}", row["solution"])
            if len(answers) != 1:
                continue
            key = canonical(row["problem"])
            if key in seen:
                continue
            seen.add(key)
            cases.append({"id": key[:20], "question": row["problem"],
                          "answer": answers[0].replace(",", ""),
                          "subject": subject, "level": row["level"]})
    cases.sort(key=lambda c: c["id"])
    if len(cases) < args.size * 2:
        raise ValueError("not enough unique eligible problems")
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for split, subset in (("dev", cases[:args.size]), ("validation", cases[args.size:args.size * 2])):
        write_new(root / f"{split}.json", {"schema": "wsp.math-integer-mini/v1", "sources": sources,
            "source_revision": REVISION, "split": split,
            "selection": "train only; level 4/5; one plain integer boxed answer; question SHA256 ascending",
            "cases": subset})
    print(json.dumps({"eligible": len(cases), "size_per_split": args.size,
                      "qualification": "selected development subset, not full MATH accuracy"}))


if __name__ == "__main__":
    main()
