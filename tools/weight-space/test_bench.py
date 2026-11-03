import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("bench", pathlib.Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ScoringTest(unittest.TestCase):
    def test_final_answer_only(self):
        for text, expected in [("42", None), ("FINAL: 1,234", "1234"),
                               ("<think>FINAL: 7", None),
                               ("<think>FINAL: 7</think>\nFINAL: 42", "42"),
                               ("FINAL: 1\nFINAL: 2", None),
                               ("FINAL: 4.20", "4.2")]:
            with self.subTest(text=text):
                value = bench.final_number(text)
                self.assertEqual(None if value is None else str(bench.decimal.Decimal(value)), expected)

    def test_unfinished_thinking_does_not_pass(self):
        self.assertIsNone(bench.final_number("<think>\nThe calculation suggests\nFINAL: 42"))

    def test_repetition_is_separate_from_correctness(self):
        self.assertGreater(bench.repeated_fourgram_fraction("a b c d " * 10), 0.5)
        self.assertEqual(bench.repeated_fourgram_fraction("a b c d e f"), 0)


if __name__ == "__main__":
    unittest.main()
