"""Tests on tiny artificial GGUF files, not evidence of model improvement."""
import json
import pathlib
import tempfile
import unittest

import gguf
import numpy as np

import surgery
from bench import digest


class SurgeryTest(unittest.TestCase):
    def test_projection_quantization_and_original_preservation(self):
        for dtype in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q4_0,
                      gguf.GGMLQuantizationType.Q8_0]:
            with self.subTest(dtype=dtype.name), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                source = root / "source.gguf"
                weights = np.random.default_rng(797).normal(size=(3, 32)).astype(np.float32)
                writer = gguf.GGUFWriter(source, "llama")
                name = "blk.39.ffn_down_shexp.weight"
                writer.add_tensor(name, gguf.quantize(weights, dtype), raw_dtype=dtype)
                writer.add_tensor("unrelated.weight", np.ones(8, dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                source_hash = digest(source)
                np.save(root / "direction.npy", np.array([1, 0, 0], dtype=np.float32))
                spec = root / "spec.json"
                spec.write_text(json.dumps({"hypothesis": "test only", "edits": [
                    {"tensor": name, "direction": "direction.npy", "strength": 0.5}]}))
                candidate = root / "candidate.gguf"
                result = surgery.apply(source, source_hash, candidate, spec)
                self.assertEqual(result["state"], "complete")
                self.assertTrue(result["original_preserved"])
                self.assertEqual(digest(source), source_hash)
                self.assertNotEqual(digest(candidate), source_hash)
                metric = result["edits"][0]
                self.assertAlmostEqual(metric["projection_norm_after"] / metric["projection_norm_before"], 0.5, delta=0.04)
                with self.assertRaisesRegex(ValueError, "new checkpoint"):
                    surgery.apply(source, source_hash, candidate, spec)
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    surgery.apply(source, "0" * 64, root / "bad.gguf", spec)
                self.assertFalse((root / "bad.gguf").exists())
                spec.write_text(json.dumps({"hypothesis": "quantization sham", "edits": [
                    {"tensor": name, "direction": "direction.npy", "strength": 0.0}]}))
                sham = surgery.apply(source, source_hash, root / "sham.gguf", spec)
                self.assertEqual(sham["candidate_sha256"], source_hash)


if __name__ == "__main__":
    unittest.main()
