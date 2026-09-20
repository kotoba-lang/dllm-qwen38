"""Pure result calculations for isolated MFU tournaments."""

from __future__ import annotations

import math


H200_BF16_DENSE_FLOPS = 989e12
MODEL_PARAMS = 27e9
EXECUTED_POSITIONS_PER_UNIQUE_TOKEN = 4.0


def proxy_mfu(seq_tokens_per_s: float, world: int = 8) -> float:
    """Return the 6N useful-training proxy against dense BF16 peak."""
    useful_flops_s = (
        6.0 * MODEL_PARAMS * EXECUTED_POSITIONS_PER_UNIQUE_TOKEN * seq_tokens_per_s
    )
    return useful_flops_s / (world * H200_BF16_DENSE_FLOPS)


def checkpoint_comparison(
    baseline: dict,
    candidate: dict,
    *,
    required_measured_steps: int = 30,
    min_speedup: float = 1.10,
    max_loss_abs_diff: float = 0.01,
    max_peak_mem_gib: float = 136.0,
) -> dict:
    """Compare full-checkpointing against no-checkpointing with explicit gates.

    The loss ceiling is deliberately stricter than a general approximate-kernel gate:
    disabling activation recomputation is intended to preserve the training math.  It is
    still above the repository's 5e-3 ambient resume band to allow two independent FSDP
    processes.  The memory ceiling reserves roughly 5 GiB on a 141 GiB H200.
    """
    base_tps = float(baseline.get("measured_seq_tokens_per_s") or 0.0)
    cand_tps = float(candidate.get("measured_seq_tokens_per_s") or 0.0)
    speedup = cand_tps / base_tps if base_tps > 0.0 else None
    base_losses = baseline.get("losses") or []
    cand_losses = candidate.get("losses") or []
    paired = min(len(base_losses), len(cand_losses), required_measured_steps)
    base_measured_losses = base_losses[-paired:] if paired else []
    cand_measured_losses = cand_losses[-paired:] if paired else []
    loss_diffs = [
        abs(float(cand_measured_losses[i]) - float(base_measured_losses[i]))
        for i in range(paired)
    ]
    loss_max = max(loss_diffs) if loss_diffs else None
    loss_mean = sum(loss_diffs) / paired if paired else None
    candidate_peak = candidate.get("peak_mem_gib")

    finite_reports = (
        base_tps > 0.0
        and cand_tps > 0.0
        and all(math.isfinite(float(x)) for x in base_losses + cand_losses)
    )
    gates = {
        "both_exited_zero": baseline.get("exit") == 0 and candidate.get("exit") == 0,
        "both_tf32_disabled": baseline.get("tf32", {}).get("enabled") is False
        and candidate.get("tf32", {}).get("enabled") is False,
        "checkpoint_modes_correct": baseline.get("checkpointing") is True
        and candidate.get("checkpointing") is False,
        "measured_steps_complete": baseline.get("measured_steps") == required_measured_steps
        and candidate.get("measured_steps") == required_measured_steps,
        "paired_loss_steps_complete": paired == required_measured_steps,
        "finite_reports": finite_reports,
        "both_weight_spread_zero": baseline.get("weight_spread") == 0.0
        and candidate.get("weight_spread") == 0.0,
        "paired_loss_within_limit": loss_max is not None and loss_max <= max_loss_abs_diff,
        "candidate_memory_within_limit": candidate_peak is not None
        and float(candidate_peak) <= max_peak_mem_gib,
        "minimum_speedup_met": speedup is not None and speedup >= min_speedup,
    }
    correctness_keys = (
        "both_exited_zero",
        "both_tf32_disabled",
        "checkpoint_modes_correct",
        "measured_steps_complete",
        "paired_loss_steps_complete",
        "finite_reports",
        "both_weight_spread_zero",
        "paired_loss_within_limit",
    )
    correctness_pass = all(gates[key] for key in correctness_keys)
    safety_pass = gates["candidate_memory_within_limit"]
    performance_pass = gates["minimum_speedup_met"]
    return {
        "baseline_measured_seq_tokens_per_s": base_tps,
        "candidate_measured_seq_tokens_per_s": cand_tps,
        "candidate_speedup": speedup,
        "candidate_cost_ratio": (1.0 / speedup) if speedup else None,
        "baseline_proxy_mfu": proxy_mfu(base_tps),
        "candidate_proxy_mfu": proxy_mfu(cand_tps),
        "proxy_mfu_definition": "6*N*4*unique_sequence_tokens_per_s/(8*989e12), N=27e9",
        "paired_loss_steps": paired,
        "loss_max_abs_diff": loss_max,
        "loss_mean_abs_diff": loss_mean,
        "candidate_peak_mem_gib": candidate_peak,
        "thresholds": {
            "required_measured_steps": required_measured_steps,
            "min_speedup": min_speedup,
            "max_loss_abs_diff": max_loss_abs_diff,
            "max_peak_mem_gib": max_peak_mem_gib,
        },
        "gates": gates,
        "correctness_pass": correctness_pass,
        "safety_pass": safety_pass,
        "performance_pass": performance_pass,
        "promotion_eligible": correctness_pass and safety_pass and performance_pass,
        "mfu_milestones": {
            "20_percent": proxy_mfu(cand_tps) >= 0.20,
            "30_percent": proxy_mfu(cand_tps) >= 0.30,
        },
    }
