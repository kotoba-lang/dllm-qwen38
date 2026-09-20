from dllm_qwen38.mfu_metrics import checkpoint_comparison


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
