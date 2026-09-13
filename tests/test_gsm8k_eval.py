"""gsm8k_eval unit tests — CPU, no download, seconds. Both directions per check (検査の 8 問).

The Qwen3 think tags are assembled from chr() everywhere in this file: a literal tag in
the source did not survive the tooling channel intact (measured 2026-09-13 — the write
dropped or rewrote it). gsm8k_eval's own inline tag literals are pinned *through
behavior*: a score() that only passes when _post_think recognizes exactly the
chr-assembled open/close tags.
"""

import pytest

from dllm_qwen38.gsm8k_eval import (
    accuracy,
    blocks_for,
    cut_at_eos,
    extract_pred,
    fewshot_block,
    fewshot_demos,
    gold_value,
    plain_fqn,
    prompt_ids,
    restore_into,
    score,
    subset_fingerprint,
    subset_indices,
    subset_rows,
    user_message,
    _post_think,
)

_T_OPEN = chr(60) + "think" + chr(62)
_T_CLOSE = chr(60) + "/think" + chr(62)


def test_post_think_cuts_close_open_and_neither():
    assert _post_think("aaa" + _T_CLOSE + "bbb") == "bbb"
    # the LAST close wins — only the tail after both traces can be trusted
    assert _post_think(_T_OPEN + "t1" + _T_CLOSE + "mid" + _T_CLOSE + "end") == "end"
    # a trace that never closed scores nothing — not its last number
    assert _post_think(_T_OPEN + "the trace cost 17 dollars") == ""
    assert _post_think("plain answer #### 5") == "plain answer #### 5"


def test_score_both_directions_and_trace_cannot_score():
    gold = "she has 3 bags\n#### 72"
    # the reasoning trace carries #### 999 before the close tag; the real answer follows it
    text = _T_OPEN + "plan it out, #### 999" + _T_CLOSE + " final #### 72"
    assert score(text, gold) == (True, "72", "####")
    assert score(text.replace("#### 72", "#### 71"), gold) == (False, "71", "####")
    # aggregate counts, not booleans: 1 correct, 1 wrong, 1 skipped — and the empty case
    a = accuracy([(True, "####"), (False, "####"), (None, "none")])
    assert a["acc"] == 0.5 and a["n"] == 2 and a["skipped"] == 1
    assert a["by_method"] == {"####": 2}
    assert accuracy([]) == {"acc": None, "n": 0, "skipped": 0, "by_method": {}}


def test_extract_pred_edges():
    assert extract_pred("so #### 1,234 total") == ("1,234", "####")
    assert extract_pred("count #### 18.") == ("18.", "####")
    assert extract_pred("owes #### -3") == ("-3", "####")
    assert extract_pred("costs #### $3.50") == ("3.50", "####")
    assert extract_pred("#### 1 ... #### 77") == ("77", "####")  # last marker wins
    assert extract_pred("the answer is 42.") == ("42", "last_number")
    assert extract_pred("no digits at all?") == ("", "none")
    # a trace full of numbers, never closed → nothing to score (no flattery)
    assert extract_pred(_T_OPEN + "5 apples then 12") == ("", "none")
    # trace closed, but nothing after the cut → no number can leak from the trace
    assert extract_pred(_T_OPEN + "5 apples" + _T_CLOSE + "done") == ("", "none")


def test_score_comma_and_currency_equivalence_and_gold_unparseable():
    assert score("#### 1,234", "receipt\n#### 1234")[0] is True
    assert score("#### $3.5", "x\n#### 3.50")[0] is True
    assert score("#### 4", "no marker in gold")[0] is None  # skipped, not wrong
    assert score("#### 4", "x\n#### abc")[0] is None
    assert gold_value("step\n#### 42") == 42
    assert gold_value("#### $7") == 7
    assert gold_value("####") is None


def test_subset_indices_stable_sorted_and_seed_sensitive():
    n1 = subset_indices(1319, 500)
    assert n1 == subset_indices(1319, 500) and len(n1) == 500
    assert n1 == sorted(n1)
    assert 0 <= n1[0] and n1[-1] < 1319
    assert subset_indices(1319, 500, seed=1) != n1
    # rows follow the same indices
    rows = [{"question": f"q{i}"} for i in range(1319)]
    picked = subset_rows(rows, 500)
    assert [r["question"] for r in picked] == [f"q{i}" for i in n1]
    # the fingerprint binds exact problems and exact demos
    demos = [("d1", "a1"), ("d2", "a2")]
    fp = subset_fingerprint(picked, demos)
    assert fp == subset_fingerprint(picked, demos)
    assert fp != subset_fingerprint(picked, demos[:1])
    assert fp != subset_fingerprint(picked[:499], demos)


def _fake_train(n=50):
    return [{"question": f"q{i}", "answer": f"cot {i}\n#### {i}"} for i in range(n)]


def test_fewshot_demos_from_train_deterministic_and_seed_sensitive():
    train = _fake_train()
    d1 = fewshot_demos(train, 4)
    assert d1 == fewshot_demos(train, 4) and len(d1) == 4
    assert all(q in {r["question"] for r in train} for q, _ in d1)
    assert fewshot_demos(train, 4, seed=7) != d1


def test_fewshot_block_format_and_live_question_last():
    demos = [("qA", "two steps\n#### 12"), ("qB", "plain")]
    item = {"question": "zzz-live"}
    block = fewshot_block(demos, item)
    assert block.count("Question: ") == 3
    assert block.strip().endswith("Answer:")
    assert "#### 12" in block
    # a demo without #### keeps its text without inventing a marker
    assert "Answer: plain" in block
    assert block.index("zzz-live") > block.index("qB")
    assert user_message(item, demos).startswith("Solve the math word problem")


def test_blocks_for_minimal_capacity_and_boundaries():
    for pl in (0, 1, 7, 8, 15, 32, 33, 1700):
        for mn in (1, 24, 32, 100, 512, 1024, 2000):
            nb = blocks_for(mn, pl, 32)
            room = 32 - pl % 32
            cap = room + (nb - 1) * 32
            assert cap >= mn, (mn, pl, nb, cap)  # covers max_new
            assert nb == 1 or room + (nb - 2) * 32 < mn, (mn, pl, nb)  # nb-1 was not enough
    assert blocks_for(28, 32 * 5, 32) == 1  # max_new == room, aligned prompt
    assert blocks_for(33, 32 * 5, 32) == 2  # first value beyond the aligned room (29 still fits: room=32)
    assert blocks_for(1024, 1700, 32) == 33


def test_prompt_ids_flattens_every_tokenizer_shape():
    """The first AR run died in run_ar_batch with ValueError("too many dimensions 'str'"):
    transformers v5 flipped apply_chat_template to return a BatchEncoding, so the old
    list(ids) iterated the dict KEYS and torch.tensor received strings. prompt_ids must
    normalize every shape to a flat list of ints — and a fake tok pins it without a model.
    """
    from collections import UserDict

    item = {"question": "qq"}
    expect = [11, 12, 13]

    class _BatchEnc(UserDict):  # BatchEncoding subclasses UserDict, not dict
        pass

    class _Tok:
        def __init__(self, ret):
            self.ret = ret

        def apply_chat_template(self, conv, tokenize=True, **kw):
            assert kw.get("add_generation_prompt") is True
            assert kw.get("enable_thinking") is False
            assert conv == [{"role": "user", "content": user_message(item, [])}]
            return self.ret

    # v5's actual return: BatchEncoding with 1-D ids
    assert prompt_ids(_Tok(_BatchEnc(input_ids=expect, attention_mask=[1, 1, 1])), item, []) == expect
    # a single conversation may come back nested
    assert prompt_ids(_Tok(_BatchEnc(input_ids=[expect], attention_mask=[[1, 1, 1]])), item, []) == expect
    # a plain dict (no .input_ids attribute) must also flatten — same .get duck path
    assert prompt_ids(_Tok({"input_ids": expect}), item, []) == expect
    # the v4-era flat list still lands as ints
    assert prompt_ids(_Tok(expect), item, []) == expect
    # eager 2-D tensor: 1-D row elements are not list/tuple, so the .dim() duck test must catch them
    import torch

    assert prompt_ids(_Tok(torch.tensor([[31, 32, 33]])), item, []) == [31, 32, 33]
    # one conversation in → one prompt row out; multi-row is a shape we don't understand
    # — fail loudly rather than silently score the wrong (truncated) prompt
    with pytest.raises(ValueError, match="2 prompt rows"):
        prompt_ids(_Tok([[21], [22]]), item, [])


def test_cut_at_eos_cuts_at_first():
    assert cut_at_eos([1, 2, 3], {5}) == [1, 2, 3]
    assert cut_at_eos([1, 2, 5, 2, 5], {5}) == [1, 2]
    assert cut_at_eos([5, 9], {5, 9}) == []


def test_plain_fqn_mapping():
    assert plain_fqn("text_model.layers.0.layer.weight") == "layers.0.weight"
    assert plain_fqn("text_model.norm.weight") == "norm.weight"
    assert plain_fqn("text_model.embed_tokens.weight") == "embed_tokens.weight"
    assert plain_fqn("lm_head.weight") == "lm_head.weight"
    assert plain_fqn("optim.step") is None


def _tiny_pair():
    import torch
    from torch import nn

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
            self.norm = nn.Linear(3, 3)

    # lm_head in the real path is a bare nn.Linear (keys weight/bias), matching DCP's
    # lm_head.weight/bias after the lm_head. prefix strip — not a wrapper module
    return Text(), nn.Linear(3, 2)


def _save_synth_ckpt(root, tensors, step=7, seq=1234):
    """A DCP checkpoint in the production layout: {'model': {...}, 'step': t, 'seq_tokens': t}."""
    import torch
    from torch.distributed.checkpoint import FileSystemWriter, save as dcp_save

    d = root / "step-00000007"
    dcp_save(
        {"model": tensors, "step": torch.tensor(step), "seq_tokens": torch.tensor(seq)},
        storage_writer=FileSystemWriter(str(d)),
        no_dist=True,
    )
    return d


def test_dcp_restore_roundtrip_and_all_three_failures(tmp_path):
    import torch

    from dllm_qwen38.gsm8k_eval import dcp_model_tensors

    text, head = _tiny_pair()
    tensors = {
        "text_model.layers.0.layer.weight": text.layers[0].weight.data + 1.0,
        "text_model.layers.0.layer.bias": text.layers[0].bias.data + 1.0,
        "text_model.layers.1.layer.weight": text.layers[1].weight.data + 2.0,
        "text_model.layers.1.layer.bias": text.layers[1].bias.data + 2.0,
        "text_model.norm.weight": text.norm.weight.data + 3.0,
        "text_model.norm.bias": text.norm.bias.data + 3.0,
        "lm_head.weight": head.weight.data + 4.0,
        "lm_head.bias": head.bias.data + 4.0,
    }
    d = _save_synth_ckpt(tmp_path, tensors)
    loaded, step, seq = dcp_model_tensors(str(d))
    assert (step, seq) == (7, 1234)
    assert len(loaded) == 8
    restore_into(text, head, loaded, ctx=str(d))
    assert torch.equal(text.layers[0].weight.data, tensors["text_model.layers.0.layer.weight"])
    assert torch.equal(head.weight.data, tensors["lm_head.weight"])

    text2, head2 = _tiny_pair()
    t2 = dict(tensors)
    del t2["text_model.norm.weight"]
    with pytest.raises(ValueError) as e:
        restore_into(text2, head2, t2)
    assert "model:norm.weight" in str(e.value)  # missing key is named, not swallowed
    t3 = dict(tensors)
    t3["text_model.ghost"] = tensors["lm_head.weight"]
    with pytest.raises(ValueError) as e2:
        restore_into(text2, head2, t3)
    assert "model?ghost" in str(e2.value)
    t4 = dict(tensors)
    t4["optim.exp_avg"] = tensors["lm_head.weight"]
    with pytest.raises(ValueError) as e3:
        restore_into(text2, head2, t4)
    assert "ckpt?optim.exp_avg" in str(e3.value)
    with pytest.raises(ValueError) as e4:
        restore_into(text2, head2, {})
    assert "restore incomplete" in str(e4.value)


def test_dtensor_shard0_roundtrip_via_plain_request(tmp_path):
    """The real training checkpoint stores FSDP DTensor shards; the eval loader requests
    plain full-shape tensors and reads every overlapping chunk. This 1-rank gloo Shard(0)
    roundtrip exercises that resharding path before it runs on Modal."""
    import os

    import torch

    from dllm_qwen38.gsm8k_eval import dcp_model_tensors

    try:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard
    except Exception as e:  # a torch build without the distributed pieces
        pytest.skip(f"torch.distributed unavailable: {e}")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    try:
        mesh = init_device_mesh("cpu", (1,))
        full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        dt = DTensor.from_local(full.clone(), mesh, [Shard(0)])
        d = _save_synth_ckpt(tmp_path, {"text_model.norm.weight": dt}, step=1, seq=2)
        loaded, step, seq = dcp_model_tensors(str(d))
        assert (step, seq) == (1, 2)
        # keys come back as DCP model keys sans the 'model.' prefix; plain_fqn is
        # restore_into's job, not this loader's
        assert loaded["text_model.norm.weight"].shape == (2, 3)
        assert torch.equal(loaded["text_model.norm.weight"], full)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_run_one_dllm_seam_smoke():
    import torch

    from dllm_qwen38.blockdiff import tiny_text_model
    from dllm_qwen38.gsm8k_eval import run_one_dllm

    torch.set_num_threads(1)
    text_model, lm_head = tiny_text_model(seed=0)

    class _Tok:
        eos_token_id = 3

        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(i) for i in ids)

    ids = [10, 11, 12, 13, 14]
    with torch.inference_mode():
        text, meta = run_one_dllm(
            text_model, lm_head, _Tok(), ids, block=8, threshold=0.9,
            sub_block=None, max_new=24, mask_id=255, eos_id=3,
        )
    # blocks_for(24, 5, 8) = 4, and the decoder always fills each block: 3 + 3*8 = 27
    assert meta["n_blocks"] == 4
    assert meta["generated"] == 27
    assert len(meta["steps_per_block"]) == 4
    toks = [int(x) for x in text.split(",")]
    assert len(toks) <= 27
    assert 3 not in toks  # cut_at_eos: the first eos never survives into the scored text
