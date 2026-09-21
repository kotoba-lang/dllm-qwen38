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


def causal_conv_comparison(
    baseline: dict,
    candidate: dict,
    *,
    required_measured_steps: int = 30,
    min_speedup: float = 1.08,
    max_loss_abs_diff: float = 0.01,
    max_kernel_out_abs_diff: float = 0.02,
    min_kernel_grad_cosine: float = 0.999,
    min_kernel_grad_norm_ratio: float = 0.99,
    max_kernel_grad_norm_ratio: float = 1.01,
    max_peak_mem_gib: float = 136.0,
    max_peak_mem_regression_gib: float = 2.0,
) -> dict:
    """Compare explicit PyTorch and compiled causal-conv1d training legs.

    Backend identity is a correctness condition: importing the package is not sufficient because
    Transformers otherwise falls back silently.  The real-weight kernel probe checks both forward
    and backward before either full training leg is admitted.
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
    probe = candidate.get("causal_conv_probe") or {}
    base_peak = baseline.get("peak_mem_gib")
    cand_peak = candidate.get("peak_mem_gib")
    finite_reports = (
        base_tps > 0.0
        and cand_tps > 0.0
        and all(math.isfinite(float(x)) for x in base_losses + cand_losses)
    )
    backend_modes = (
        baseline.get("causal_conv", {}).get("requested") == "reference"
        and baseline.get("causal_conv", {}).get("active") == "reference"
        and candidate.get("causal_conv", {}).get("requested") == "optimized"
        and candidate.get("causal_conv", {}).get("active") == "optimized"
        and bool(candidate.get("causal_conv", {}).get("package_version"))
    )
    gates = {
        "both_exited_zero": baseline.get("exit") == 0 and candidate.get("exit") == 0,
        "both_tf32_disabled": baseline.get("tf32", {}).get("enabled") is False
        and candidate.get("tf32", {}).get("enabled") is False,
        "both_full_checkpointing": baseline.get("checkpointing") is True
        and candidate.get("checkpointing") is True,
        "backend_modes_correct": backend_modes,
        "measured_steps_complete": baseline.get("measured_steps") == required_measured_steps
        and candidate.get("measured_steps") == required_measured_steps,
        "paired_loss_steps_complete": paired == required_measured_steps,
        "finite_reports": finite_reports,
        "both_weight_spread_zero": baseline.get("weight_spread") == 0.0
        and candidate.get("weight_spread") == 0.0,
        "paired_loss_within_limit": loss_max is not None and loss_max <= max_loss_abs_diff,
        "optimized_probe_active": probe.get("optimized_backend") == "optimized"
        and bool(probe.get("package_version")),
        "optimized_probe_finite": probe.get("all_finite") is True,
        "kernel_output_within_limit": probe.get("out_max_abs_diff") is not None
        and float(probe["out_max_abs_diff"]) <= max_kernel_out_abs_diff,
        "kernel_gradient_within_limit": probe.get("grad_cosine") is not None
        and float(probe["grad_cosine"]) >= min_kernel_grad_cosine
        and probe.get("grad_norm_ratio") is not None
        and min_kernel_grad_norm_ratio
        <= float(probe["grad_norm_ratio"])
        <= max_kernel_grad_norm_ratio,
        "candidate_memory_within_absolute_limit": cand_peak is not None
        and float(cand_peak) <= max_peak_mem_gib,
        "candidate_memory_regression_within_limit": base_peak is not None
        and cand_peak is not None
        and float(cand_peak) <= float(base_peak) + max_peak_mem_regression_gib,
        "minimum_speedup_met": speedup is not None and speedup >= min_speedup,
    }
    correctness_keys = (
        "both_exited_zero",
        "both_tf32_disabled",
        "both_full_checkpointing",
        "backend_modes_correct",
        "measured_steps_complete",
        "paired_loss_steps_complete",
        "finite_reports",
        "both_weight_spread_zero",
        "paired_loss_within_limit",
        "optimized_probe_active",
        "optimized_probe_finite",
        "kernel_output_within_limit",
        "kernel_gradient_within_limit",
    )
    safety_keys = (
        "candidate_memory_within_absolute_limit",
        "candidate_memory_regression_within_limit",
    )
    correctness_pass = all(gates[key] for key in correctness_keys)
    safety_pass = all(gates[key] for key in safety_keys)
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
        "kernel_probe": probe,
        "baseline_peak_mem_gib": base_peak,
        "candidate_peak_mem_gib": cand_peak,
        "thresholds": {
            "required_measured_steps": required_measured_steps,
            "min_speedup": min_speedup,
            "max_loss_abs_diff": max_loss_abs_diff,
            "max_kernel_out_abs_diff": max_kernel_out_abs_diff,
            "min_kernel_grad_cosine": min_kernel_grad_cosine,
            "min_kernel_grad_norm_ratio": min_kernel_grad_norm_ratio,
            "max_kernel_grad_norm_ratio": max_kernel_grad_norm_ratio,
            "max_peak_mem_gib": max_peak_mem_gib,
            "max_peak_mem_regression_gib": max_peak_mem_regression_gib,
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


def fsdp_root_comparison(
    baseline: dict,
    candidate: dict,
    *,
    required_measured_steps: int = 30,
    expected_layers: int = 64,
    min_speedup: float = 1.08,
    max_loss_abs_diff: float = 0.01,
    min_grad_norm_ratio: float = 0.99,
    max_grad_norm_ratio: float = 1.01,
    max_peak_mem_gib: float = 136.0,
    max_peak_mem_regression_gib: float = 5.0,
) -> dict:
    """Compare the legacy sibling roots with one called FSDP2 root.

    Topology evidence is a correctness gate, not metadata: a fast candidate that does not
    establish one root/shared context/full layer order did not test implicit prefetching.
    """
    base_tps = float(baseline.get("measured_seq_tokens_per_s") or 0.0)
    cand_tps = float(candidate.get("measured_seq_tokens_per_s") or 0.0)
    speedup = cand_tps / base_tps if base_tps > 0.0 else None
    base_losses = baseline.get("losses") or []
    cand_losses = candidate.get("losses") or []
    paired = min(len(base_losses), len(cand_losses), required_measured_steps)
    base_measured = base_losses[-paired:] if paired else []
    cand_measured = cand_losses[-paired:] if paired else []
    diffs = [abs(float(a) - float(b)) for a, b in zip(base_measured, cand_measured)]
    loss_max = max(diffs) if diffs else None
    loss_mean = sum(diffs) / paired if paired else None
    base_grad = baseline.get("grad_norm_first")
    cand_grad = candidate.get("grad_norm_first")
    grad_ratio = (
        float(cand_grad) / float(base_grad)
        if base_grad is not None and cand_grad is not None and float(base_grad) > 0.0
        else None
    )
    base_peak = baseline.get("peak_mem_gib")
    cand_peak = candidate.get("peak_mem_gib")
    topology = candidate.get("topology_probe") or {}
    base_schema = baseline.get("checkpoint_schema") or {}
    cand_schema = candidate.get("checkpoint_schema") or {}
    finite = (
        base_tps > 0.0
        and cand_tps > 0.0
        and all(math.isfinite(float(value)) for value in base_losses + cand_losses)
        and base_grad is not None
        and cand_grad is not None
        and math.isfinite(float(base_grad))
        and math.isfinite(float(cand_grad))
    )
    topology_valid = (
        topology.get("passed") is True
        and topology.get("all_ranks_match") is True
        and topology.get("layers") == expected_layers
        and topology.get("child_param_groups") == expected_layers
        and topology.get("root_is_root") is True
        and topology.get("child_root_count") == 0
        and topology.get("distinct_comm_contexts") == 1
        and topology.get("distinct_state_contexts") == 1
        and topology.get("root_all_states") == expected_layers + 1
        and topology.get("post_forward_order") == expected_layers
        and topology.get("post_forward_unique_groups") == expected_layers
        and topology.get("distinct_group_comm_contexts") == 1
        and topology.get("distinct_rs_state_lists") == 1
    )
    gates = {
        "both_exited_zero": baseline.get("exit") == 0 and candidate.get("exit") == 0,
        "both_tf32_disabled": baseline.get("tf32", {}).get("enabled") is False
        and candidate.get("tf32", {}).get("enabled") is False,
        "both_reference_causal_conv": baseline.get("causal_conv", {}).get("requested") == "reference"
        and baseline.get("causal_conv", {}).get("active") == "reference"
        and candidate.get("causal_conv", {}).get("requested") == "reference"
        and candidate.get("causal_conv", {}).get("active") == "reference",
        "only_topology_differs": baseline.get("fsdp_topology") == "sibling"
        and candidate.get("fsdp_topology") == "root",
        "both_full_non_reentrant_checkpointing": baseline.get("checkpointing") is True
        and candidate.get("checkpointing") is True
        and baseline.get("checkpoint_impl") == "non_reentrant_per_layer"
        and candidate.get("checkpoint_impl") == "non_reentrant_per_layer"
        and baseline.get("checkpoint_regions_per_forward") == expected_layers
        and candidate.get("checkpoint_regions_per_forward") == expected_layers,
        "candidate_root_topology_valid": topology_valid,
        "measured_steps_complete": baseline.get("measured_steps") == required_measured_steps
        and candidate.get("measured_steps") == required_measured_steps,
        "paired_loss_steps_complete": paired == required_measured_steps,
        "finite_reports": finite,
        "both_weight_spread_zero": baseline.get("weight_spread") == 0.0
        and candidate.get("weight_spread") == 0.0,
        "paired_loss_within_limit": loss_max is not None and loss_max <= max_loss_abs_diff,
        "gradient_norm_within_limit": grad_ratio is not None
        and min_grad_norm_ratio <= grad_ratio <= max_grad_norm_ratio,
        "checkpoint_schema_matches": bool(base_schema.get("sha256"))
        and base_schema == cand_schema,
        "candidate_memory_within_absolute_limit": cand_peak is not None
        and float(cand_peak) <= max_peak_mem_gib,
        "candidate_memory_regression_within_limit": base_peak is not None
        and cand_peak is not None
        and float(cand_peak) <= float(base_peak) + max_peak_mem_regression_gib,
        "minimum_speedup_met": speedup is not None and speedup >= min_speedup,
    }
    correctness_keys = (
        "both_exited_zero",
        "both_tf32_disabled",
        "both_reference_causal_conv",
        "only_topology_differs",
        "both_full_non_reentrant_checkpointing",
        "candidate_root_topology_valid",
        "measured_steps_complete",
        "paired_loss_steps_complete",
        "finite_reports",
        "both_weight_spread_zero",
        "paired_loss_within_limit",
        "gradient_norm_within_limit",
        "checkpoint_schema_matches",
    )
    safety_keys = (
        "candidate_memory_within_absolute_limit",
        "candidate_memory_regression_within_limit",
    )
    correctness_pass = all(gates[key] for key in correctness_keys)
    safety_pass = all(gates[key] for key in safety_keys)
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
        "gradient_norm_ratio": grad_ratio,
        "baseline_peak_mem_gib": base_peak,
        "candidate_peak_mem_gib": cand_peak,
        "candidate_topology": topology,
        "checkpoint_schema": cand_schema,
        "thresholds": {
            "required_measured_steps": required_measured_steps,
            "expected_layers": expected_layers,
            "min_speedup": min_speedup,
            "max_loss_abs_diff": max_loss_abs_diff,
            "min_grad_norm_ratio": min_grad_norm_ratio,
            "max_grad_norm_ratio": max_grad_norm_ratio,
            "max_peak_mem_gib": max_peak_mem_gib,
            "max_peak_mem_regression_gib": max_peak_mem_regression_gib,
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
