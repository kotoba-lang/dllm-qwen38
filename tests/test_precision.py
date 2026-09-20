from types import SimpleNamespace

from dllm_qwen38.precision import configure_tf32


class FakeTorch:
    def __init__(self):
        self._precision = "unset"
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)))

    def set_float32_matmul_precision(self, value):
        self._precision = value

    def get_float32_matmul_precision(self):
        return self._precision


def test_tf32_baseline_is_explicitly_disabled():
    torch = FakeTorch()

    report = configure_tf32(torch, False)

    assert report == {
        "enabled": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
    }


def test_tf32_candidate_enables_high_precision_tensor_core_mode():
    torch = FakeTorch()

    report = configure_tf32(torch, True)

    assert report == {
        "enabled": True,
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
    }
