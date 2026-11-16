import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from capture_data import load_vectors


class CaptureDataTest(unittest.TestCase):
    def test_empty_intermediate_is_valid_but_missing_or_corrupt_final_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "capture-status.json").write_text(json.dumps({"prefill_api_calls": 1,
                "generated_tokens": 0, "process_exit": 0, "capture_failed": False,
                "first_layer_graph_visits": 2}))
            (root / "prefill-runs.jsonl").write_text(json.dumps({"prompt_index": 0,
                "api_exit": 0, "first_layer_graph_visits": 2}) + "\n")
            (root / "empty.f32").write_bytes(b"")
            np.array([3, 7], dtype="<f4").tofile(root / "final.f32")
            empty = {"prompt_index": 0, "tensor": "l_out-39", "shape": [2, 0, 1, 1],
                     "file": "empty.f32", "bytes": 0}
            final = {**empty, "shape": [2, 1, 1, 1], "file": "final.f32", "bytes": 8}
            index = root / "tensors.jsonl"
            index.write_text(json.dumps(empty) + "\n" + json.dumps(final) + "\n")
            values, info = load_vectors(root, width=2, prompts=1, layers=(39,))
            np.testing.assert_array_equal(values[(0, "l_out-39")], [3, 7])
            self.assertEqual(info["empty_microbatch_outputs"], 1)
            index.write_text(json.dumps(empty) + "\n")
            with self.assertRaisesRegex(ValueError, "missing nonempty"):
                load_vectors(root, width=2, prompts=1, layers=(39,))
            index.write_text(json.dumps(final) + "\n")
            (root / "final.f32").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "byte count"):
                load_vectors(root, width=2, prompts=1, layers=(39,))


if __name__ == "__main__":
    unittest.main()
