"""Explicit floating-point policy for reproducible performance experiments."""

from __future__ import annotations


def configure_tf32(torch_module, enabled: bool) -> dict:
    """Set and report the CUDA matmul TF32 policy.

    The baseline is deliberately explicit rather than inheriting a PyTorch or image default.
    Keeping this helper independent of a concrete torch import also makes the configuration
    contract unit-testable on machines without CUDA/PyTorch.
    """
    enabled = bool(enabled)
    torch_module.set_float32_matmul_precision("high" if enabled else "highest")
    torch_module.backends.cuda.matmul.allow_tf32 = enabled
    return {
        "enabled": enabled,
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
    }
