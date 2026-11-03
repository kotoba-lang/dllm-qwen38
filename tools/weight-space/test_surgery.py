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


    def test_no_op_retains_noncanonical_q4_encoding(self):
        # Legal Q4_0 scales need not use the encoder's signed-maximum range.
        # This block has scale1 and levels +7/+1/-1, but no -8 endpoint.
        dtype = gguf.GGMLQuantizationType.Q4_0
        raw = np.array([[0, 60, 0xF9] + [0x97] * 15] * 3, dtype=np.uint8)
        values = gguf.dequantize(raw, dtype)
        self.assertFalse(np.array_equal(gguf.quantize(values, dtype), raw))
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "noncanonical.gguf"
            writer = gguf.GGUFWriter(source, "llama")
            name = "blk.39.ffn_down_shexp.weight"
            writer.add_tensor(name, raw, raw_dtype=dtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            original = digest(source)
            np.save(root / "d.npy", np.array([1, 0, 0], dtype=np.float32))
            spec = root / "spec.json"
            edit = {"tensor": name, "direction": "d.npy", "strength": 0.0}
            spec.write_text(json.dumps({"hypothesis": "no-op control", "edits": [edit]}))
            result = surgery.apply(source, original, root / "noop.gguf", spec)
            self.assertEqual(result["candidate_sha256"], original)
            self.assertEqual(result["edits"][0]["logical_values_changed"], 0)
            edit["requantize_only"] = True
            spec.write_text(json.dumps({"hypothesis": "encoding control", "edits": [edit]}))
            result = surgery.apply(source, original, root / "reencode.gguf", spec)
            self.assertNotEqual(result["candidate_sha256"], original)
            self.assertGreater(result["edits"][0]["delta_l2"], 0)
            self.assertEqual(result["edits"][0]["operation"], "requantize_only")
            self.assertEqual(digest(source), original)


if __name__ == "__main__":
    unittest.main()
