"""Unit test for ADR-2609112640 §3a on the tiny random model (no download, CPU, seconds).

Counts are asserted, not booleans: every check must pass and every fault must be detected,
otherwise the equivalence test is theater.
"""

from dllm_qwen38.blockdiff import tiny_text_model
from dllm_qwen38.equivalence import FAULTS, run


def test_state_fork_matches_oracle_and_faults_are_caught():
    text_model, lm_head = tiny_text_model(seed=0)
    rep = run(text_model, lm_head, block=8, nblocks=4, layers=None, tol=1e-3, change_floor=0.1, seed=0, device="cpu")
    failed = [c["name"] for c in rep["checks"] if not c["pass"]]
    assert failed == [], rep["checks"]
    assert rep["checks_total"] == 5
    undetected = [f["fault"] for f in rep["faults"] if not f["detected"]]
    assert undetected == [], rep["faults"]
    assert rep["faults_total"] == len(FAULTS) == 3


def test_layer_truncation_runs_the_first_n_layers_only():
    text_model, lm_head = tiny_text_model(seed=1)
    rep = run(text_model, lm_head, block=8, nblocks=2, layers=3, tol=1e-3, change_floor=0.05, seed=1, device="cpu")
    assert rep["layers"] == 3
    assert rep["layer_types"] == ["linear_attention", "linear_attention", "linear_attention"]
    assert rep["checks_passed"] == rep["checks_total"]
