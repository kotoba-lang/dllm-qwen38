import pytest

from dllm_qwen38.recovery import (
    format_fingerprint_tsv,
    latest_atomic_dcp_checkpoint,
    resolve_n_steps,
)


def test_target_step_limits_resume_to_absolute_remainder():
    assert resolve_n_steps(start_step=1000, requested_steps=8000, target_step=8000) == 7000


def test_target_step_retry_at_target_is_zero_step_noop():
    assert resolve_n_steps(start_step=8000, requested_steps=8000, target_step=8000) == 0


def test_target_step_absent_preserves_window_semantics():
    assert resolve_n_steps(start_step=1000, requested_steps=8000, target_step=None) == 8000


def test_target_step_refuses_rewind_or_overshoot():
    with pytest.raises(ValueError, match="behind checkpoint step"):
        resolve_n_steps(start_step=1001, requested_steps=8000, target_step=1000)


def test_fingerprint_tsv_has_one_model_and_optimizer_hash_per_rank():
    assert format_fingerprint_tsv(["model0 optim0", "model1 optim1"]) == (
        "0\tmodel0\toptim0\n1\tmodel1\toptim1\n"
    )


def test_fingerprint_tsv_rejects_incomplete_rank_evidence():
    with pytest.raises(ValueError, match="model and optimizer"):
        format_fingerprint_tsv(["only-model"])


def test_latest_atomic_checkpoint_ignores_tmp_and_incomplete_directories(tmp_path):
    complete = tmp_path / "step-00001000"
    complete.mkdir()
    (complete / ".metadata").write_text("complete")
    (tmp_path / "step-00002000.tmp").mkdir()
    (tmp_path / "step-00003000").mkdir()
    (tmp_path / "step-00004000").write_text("not a directory")

    assert latest_atomic_dcp_checkpoint(str(tmp_path)) == 1000
