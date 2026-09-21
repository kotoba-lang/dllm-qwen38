from dllm_qwen38.mfu_metrics import (
    causal_conv_comparison,
    checkpoint_comparison,
    fsdp_root_comparison,
)


def _report(*, tps, checkpointing, losses=None, peak=120.0, exit=0, spread=0.0):
    return {
        "measured_seq_tokens_per_s": tps,
        "measured_steps": 30,
        "checkpointing": checkpointing,
        "tf32": {"enabled": False},
        "losses": losses or [3.0 - i * 0.01 for i in range(30)],
        "peak_mem_gib": peak,
        "exit": exit,
        "weight_spread": spread,
    }


def test_checkpoint_candidate_passes_all_registered_gates():
    result = checkpoint_comparison(
        _report(tps=1300.0, checkpointing=True),
        _report(
            tps=1500.0,
            checkpointing=False,
            losses=[3.001 - i * 0.01 for i in range(30)],
        ),
    )

    assert result["correctness_pass"] is True
    assert result["safety_pass"] is True
    assert result["performance_pass"] is True
    assert result["promotion_eligible"] is True
    assert result["candidate_speedup"] == 1500.0 / 1300.0


def test_oom_or_missing_candidate_fails_closed():
    result = checkpoint_comparison(
        _report(tps=1300.0, checkpointing=True),
        {"exit": 1, "checkpointing": False, "tf32": {"enabled": False}},
    )

    assert result["correctness_pass"] is False
    assert result["safety_pass"] is False
    assert result["performance_pass"] is False
    assert result["promotion_eligible"] is False


def test_speed_does_not_override_loss_or_memory_failure():
    result = checkpoint_comparison(
        _report(tps=1300.0, checkpointing=True),
        _report(
            tps=2000.0,
            checkpointing=False,
            losses=[3.2 - i * 0.01 for i in range(30)],
            peak=139.0,
        ),
    )

    assert result["performance_pass"] is True
    assert result["correctness_pass"] is False
    assert result["safety_pass"] is False
    assert result["promotion_eligible"] is False


def _conv_report(
    *,
    tps,
    mode,
    losses=None,
    peak=120.0,
    exit=0,
    spread=0.0,
    probe=True,
):
    package = "1.7.0" if mode == "optimized" else None
    return {
        "measured_seq_tokens_per_s": tps,
        "measured_steps": 30,
        "checkpointing": True,
        "tf32": {"enabled": False},
        "losses": losses or [3.0 - i * 0.01 for i in range(30)],
        "peak_mem_gib": peak,
        "exit": exit,
        "weight_spread": spread,
        "causal_conv": {
            "requested": mode,
            "active": mode,
            "package_version": package,
        },
        "causal_conv_probe": (
            {
                "optimized_backend": "optimized",
                "package_version": "1.7.0",
                "all_finite": True,
                "out_max_abs_diff": 0.005,
                "grad_max_abs_diff": 0.006,
                "grad_cosine": 0.9999,
                "grad_norm_ratio": 1.001,
            }
            if probe
            else None
        ),
    }


def test_causal_conv_candidate_passes_backend_correctness_and_performance_gates():
    result = causal_conv_comparison(
        _conv_report(tps=1300.0, mode="reference"),
        _conv_report(
            tps=1450.0,
            mode="optimized",
            losses=[3.001 - i * 0.01 for i in range(30)],
        ),
    )

    assert result["correctness_pass"] is True
    assert result["safety_pass"] is True
    assert result["performance_pass"] is True
    assert result["promotion_eligible"] is True


def test_causal_conv_import_without_active_compiled_backend_fails_closed():
    candidate = _conv_report(tps=1600.0, mode="optimized")
    candidate["causal_conv"]["active"] = "transformers-auto"
    candidate["causal_conv_probe"] = None

    result = causal_conv_comparison(
        _conv_report(tps=1300.0, mode="reference"), candidate
    )

    assert result["gates"]["backend_modes_correct"] is False
    assert result["gates"]["optimized_probe_active"] is False
    assert result["correctness_pass"] is False
    assert result["promotion_eligible"] is False


def test_causal_conv_speed_does_not_override_numerical_or_memory_failure():
    candidate = _conv_report(
        tps=1800.0,
        mode="optimized",
        losses=[3.1 - i * 0.01 for i in range(30)],
        peak=139.0,
    )
    candidate["causal_conv_probe"]["grad_cosine"] = 0.9

    result = causal_conv_comparison(
        _conv_report(tps=1300.0, mode="reference"), candidate
    )

    assert result["performance_pass"] is True
    assert result["correctness_pass"] is False
    assert result["safety_pass"] is False
    assert result["promotion_eligible"] is False


def _root_report(
    *, tps, topology, losses=None, peak=120.0, exit=0, spread=0.0, grad=2.0
):
    layers = 64
    probe = {
        "requested": topology,
        "layers": layers,
        "child_param_groups": layers,
        "root_is_root": topology == "root",
        "child_root_count": 0 if topology == "root" else layers,
        "distinct_comm_contexts": 1 if topology == "root" else layers,
        "distinct_state_contexts": 1 if topology == "root" else layers,
        "root_all_states": layers + 1 if topology == "root" else 0,
        "post_forward_order": layers if topology == "root" else 0,
        "post_forward_unique_groups": layers if topology == "root" else 0,
        "distinct_group_comm_contexts": 1 if topology == "root" else layers,
        "distinct_rs_state_lists": 1,
        "errors": [],
        "passed": True,
        "all_ranks_match": True,
    }
    return {
        "measured_seq_tokens_per_s": tps,
        "measured_steps": 30,
        "checkpointing": True,
        "checkpoint_impl": "non_reentrant_per_layer",
        "checkpoint_regions_per_forward": layers,
        "fsdp_topology": topology,
        "topology_probe": probe,
        "checkpoint_schema": {"sha256": "same", "tensor_count": 999},
        "tf32": {"enabled": False},
        "causal_conv": {"requested": "reference", "active": "reference"},
        "losses": losses or [3.0 - i * 0.01 for i in range(30)],
        "grad_norm_first": grad,
        "peak_mem_gib": peak,
        "exit": exit,
        "weight_spread": spread,
    }


def test_fsdp_true_root_candidate_passes_every_gate():
    result = fsdp_root_comparison(
        _root_report(tps=1300.0, topology="sibling"),
        _root_report(
            tps=1420.0,
            topology="root",
            losses=[3.001 - i * 0.01 for i in range(30)],
            grad=2.01,
            peak=123.0,
        ),
    )

    assert result["correctness_pass"] is True
    assert result["safety_pass"] is True
    assert result["performance_pass"] is True
    assert result["promotion_eligible"] is True


def test_fsdp_root_speed_cannot_override_missing_structural_order():
    candidate = _root_report(tps=1700.0, topology="root")
    candidate["topology_probe"]["post_forward_order"] = 1
    candidate["topology_probe"]["passed"] = False

    result = fsdp_root_comparison(
        _root_report(tps=1300.0, topology="sibling"), candidate
    )

    assert result["performance_pass"] is True
    assert result["gates"]["candidate_root_topology_valid"] is False
    assert result["correctness_pass"] is False
    assert result["promotion_eligible"] is False


def test_fsdp_root_fails_closed_on_gradient_schema_loss_or_memory_drift():
    candidate = _root_report(
        tps=1800.0,
        topology="root",
        losses=[3.2 - i * 0.01 for i in range(30)],
        grad=2.2,
        peak=140.0,
    )
    candidate["checkpoint_schema"]["sha256"] = "different"

    result = fsdp_root_comparison(
        _root_report(tps=1300.0, topology="sibling"), candidate
    )

    assert result["performance_pass"] is True
    assert result["correctness_pass"] is False
    assert result["safety_pass"] is False
    assert result["promotion_eligible"] is False
