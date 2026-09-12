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


def test_gdn_need_states_false_matches_true_and_skips_states():
    import torch

    from dllm_qwen38.gdn import gdn_mixer_forward, last_impl

    text_model, _ = tiny_text_model(seed=4)
    m = text_model.layers[0].linear_attn
    torch.manual_seed(4)
    hs = torch.randn(2, 16, text_model.config.hidden_size)

    o1, s1, raw1 = gdn_mixer_forward(m, hs, chunk_size=16, need_states=True)
    o2, s2, raw2 = gdn_mixer_forward(m, hs, chunk_size=16, need_states=False)
    assert float((o1 - o2).abs().max()) == 0.0  # need_states changes bookkeeping, never the output
    assert s1.shape[:2] == (2, 1) and s1.shape[-2:] == (16, 16)  # [B, n_chunks, Hv, K, V]
    assert s2 is None
    assert raw1.shape == raw2.shape
    assert last_impl()[0] == "reference"  # cpu: the reference is the only core, fla never tried


def test_checkpointing_matches_plain_forward_grads_and_learns():
    import torch

    from dllm_qwen38.blockdiff import block_diffusion_forward
    from dllm_qwen38.train import synthetic_batches, train

    torch.set_num_threads(1)
    text_model, lm_head = tiny_text_model(seed=6)
    text_model.train()
    torch.manual_seed(6)
    x0 = torch.randint(0, 256, (2, 24))
    xt = x0.clone()
    xt[torch.rand(2, 24) < 0.5] = 255
    with torch.enable_grad():
        a, _ = block_diffusion_forward(text_model, lm_head, x0, xt, 8)
        a.sum().backward()
        plain = {n: p.grad.detach().clone() for n, p in text_model.named_parameters() if p.grad is not None}
        for p in text_model.parameters():
            p.grad = None
        b, _ = block_diffusion_forward(text_model, lm_head, x0, xt, 8, checkpointing=True)
        assert torch.equal(a, b)  # same math; checkpointing only moves activations to recompute time
        b.sum().backward()
        n_params = len(list(text_model.parameters()))
        have = {n: p.grad for n, p in text_model.named_parameters() if p.grad is not None}
        # the severed-chain failure (closure over the layer loop variable) left 13/108 params
        assert len(have) == n_params, f"grads on {len(have)}/{n_params} params"
        wrong = [n for n, g in have.items() if not torch.equal(g, plain[n])]
        assert wrong == [], wrong[:5]
    for p in text_model.parameters():
        p.grad = None
    # and it learns: the same loop hyperparams the plain test passes, under checkpointing
    b = synthetic_batches(4, 32, 256, 8, seed=6)
    rep = train(text_model, lm_head, b, block=8, mask_id=255, steps=100, lr=3e-3, device="cpu", checkpointing=True, log=lambda *_: None)
    assert rep["skipped_steps"] == 0
    assert rep["loss_last"] < 0.5 * rep["loss_first"], (rep["loss_first"], rep["loss_last"])


def test_kernel_diff_refuses_to_answer_off_cuda():
    import torch

    from dllm_qwen38 import gdn

    text_model, _ = tiny_text_model(seed=5)
    m = text_model.layers[0].linear_attn
    if torch.cuda.is_available():
        return  # the cuda path is lever_bench's job; this machine has none
    try:
        gdn.kernel_diff(m, T=8, device="cpu")
    except RuntimeError as e:
        # the two refusal literals of gdn._core / gdn.kernel_diff: fla missing vs fla present but not on CUDA
        assert "fla path not taken" in str(e) or "fla forced but unavailable" in str(e), e
    else:
        raise AssertionError("kernel_diff must refuse to answer where the fla path cannot run")
