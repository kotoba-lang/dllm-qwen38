from dllm_qwen38.mfu_metrics import causal_conv_comparison, checkpoint_comparison


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
