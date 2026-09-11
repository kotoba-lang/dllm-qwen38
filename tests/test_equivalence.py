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


def test_batched_forward_equals_per_row_and_oracle():
    import torch

    from dllm_qwen38.blockdiff import block_diffusion_forward, reference_block_logits

    text_model, lm_head = tiny_text_model(seed=2)
    block, nb, B = 8, 3, 3
    torch.manual_seed(2)
    x0 = torch.randint(0, 256, (B, block * nb))
    xt = x0.clone()
    xt[torch.rand(B, block * nb) < 0.5] = 255
    with torch.no_grad():
        lt, l0 = block_diffusion_forward(text_model, lm_head, x0, xt, block)
        rows = [block_diffusion_forward(text_model, lm_head, x0[i : i + 1], xt[i : i + 1], block)[0] for i in range(B)]
        ref = reference_block_logits(text_model, lm_head, x0, xt, block)
    assert lt.shape == (B, block * nb, 256)
    assert float((lt - torch.cat(rows, 0)).abs().max()) < 1e-5
    assert float((lt - ref).abs().max()) < 1e-3


def test_training_loop_learns_the_rule_and_lr_zero_does_not():
    import torch

    from dllm_qwen38.train import synthetic_batches, train

    torch.set_num_threads(1)
    text_model, lm_head = tiny_text_model(seed=3)
    b = synthetic_batches(4, 32, 256, 8, seed=3)
    rep = train(text_model, lm_head, b, block=8, mask_id=255, steps=100, lr=3e-3, device="cpu", log=lambda *_: None)
    assert rep["skipped_steps"] == 0
    assert rep["loss_last"] < 0.5 * rep["loss_first"], (rep["loss_first"], rep["loss_last"])

    text_model, lm_head = tiny_text_model(seed=3)
    b = synthetic_batches(4, 32, 256, 8, seed=3)
    rep0 = train(text_model, lm_head, b, block=8, mask_id=255, steps=30, lr=0.0, device="cpu", log=lambda *_: None)
    assert abs(rep0["loss_last"] - rep0["loss_first"]) < 0.05 * rep0["loss_first"]
