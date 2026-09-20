"""Pure helpers for interruption-safe checkpoint recovery."""

from __future__ import annotations

import os


def format_fingerprint_tsv(fingerprints: list[str]) -> str:
    """Serialize per-rank ``model-fp opt-fp`` pairs in the checkpoint format."""
    lines = []
    for rank, fingerprint in enumerate(fingerprints):
        parts = fingerprint.split()
        if len(parts) != 2:
            raise ValueError(f"rank {rank} fingerprint must contain model and optimizer hashes")
        lines.append(f"{rank}\t{parts[0]}\t{parts[1]}\n")
    return "".join(lines)


def resolve_n_steps(start_step: int, requested_steps: int, target_step: int | None) -> int:
    """Return this invocation's steps while enforcing an optional absolute stop."""
    if target_step is None:
        return requested_steps
    if target_step < start_step:
        raise ValueError(
            f"target step {target_step} is behind checkpoint step {start_step}; refusing to overshoot or rewind"
        )
    return target_step - start_step


def latest_atomic_dcp_checkpoint(ckpt_dir: str) -> int | None:
    """Find the newest renamed DCP directory carrying its completion metadata."""
    try:
        entries = os.listdir(ckpt_dir)
    except FileNotFoundError:
        return None
    completed = []
    for entry in entries:
        if not entry.startswith("step-") or entry.endswith(".tmp"):
            continue
        path = os.path.join(ckpt_dir, entry)
        if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, ".metadata")):
            continue
        try:
            completed.append(int(entry.split("-")[1]))
        except ValueError:
            continue
    return max(completed) if completed else None
