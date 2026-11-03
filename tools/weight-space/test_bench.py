import importlib.util
import pathlib
import argparse
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def test_new_server_limit_field(self):
        rows = [{"correct": False, "wall_seconds": 1, "predicted": None,
                 "response": {"stop_type": "limit", "tokens_predicted": 20}}]
        result = bench.summary(rows)
        self.assertEqual(result["limit_hits"], 1)
        self.assertIsNone(result["seconds_per_correct"])


class LocalServerTest(unittest.TestCase):
    def test_run_resume_identity_and_wrong_answers(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model = root / "model.gguf"
            model.write_bytes(b"test checkpoint identity, not a language model")
            data = root / "dev.json"
            data.write_text(json.dumps({"split": "dev", "cases": [
                {"id": "a", "question": "1+1?", "answer": "2"},
                {"id": "b", "question": "2+2?", "answer": "4"}]}))
            requests = []

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({"model_path": str(model),
                        "default_generation_settings": {"n_ctx": 2048}}).encode())

                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    requests.append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({"content": "FINAL: 2", "tokens_predicted": 5,
                                               "stop_type": "eos"}).encode())

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                args = argparse.Namespace(server=f"http://127.0.0.1:{server.server_port}",
                    model=str(model), model_sha256=bench.digest(model), data=str(data),
                    limit=0, max_tokens=32, timeout=5, seed=42, thinking="off",
                    backend_commit="test-backend", condition="test", output=str(root / "run"))
                with contextlib.redirect_stdout(io.StringIO()):
                    bench.run(args)
                    bench.run(args)
                self.assertEqual(len(requests), 2, "resume must not repeat completed requests")
                result = json.loads((root / "run/summary.json").read_text())
                self.assertEqual((result["n"], result["correct"]), (2, 1))
                self.assertFalse(requests[0]["cache_prompt"])
                args.max_tokens = 33
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    bench.run(args)
                self.assertEqual(len(requests), 2)
                args.server = "https://example.com"
                with self.assertRaisesRegex(ValueError, "loopback"):
                    bench.run(args)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
