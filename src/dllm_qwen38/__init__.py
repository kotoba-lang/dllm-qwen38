"""Qwen3.8 (Gated DeltaNet + Gated Attention hybrid) adapted to Fast-dLLM v2 block diffusion.

The only genuinely new part relative to the paper (arXiv 2509.26328, Qwen2.5 base) is the
48/64 recurrent layers: the paper's complementary attention mask cannot be applied to a
recurrence, so this package realises the same *visibility* with a state fork (see gdn.py and
blockdiff.py) and ships the test that proves the two agree (equivalence.py).
"""
