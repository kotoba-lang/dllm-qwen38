"""Lifecycle test with a fake local engine; this is not model inference evidence."""
import hashlib
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest


ENGINE = '''#!{python}
import json,sys,signal
from http.server import BaseHTTPRequestHandler,HTTPServer
port=int(sys.argv[sys.argv.index('--port')+1])
model=sys.argv[sys.argv.index('-m')+1]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers()
  self.wfile.write(json.dumps({{"model_path":model,"default_generation_settings":{{"n_ctx":2048}}}}).encode())
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.end_headers()
  self.wfile.write(json.dumps({{"content":"FINAL: 2","tokens_predicted":5,"stop_type":"eos"}}).encode())
signal.signal(signal.SIGTERM,lambda *args:sys.exit(0))
HTTPServer.allow_reuse_address=True
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
'''


class LaunchTest(unittest.TestCase):
    def test_owned_process_stops_and_bad_digest_never_launches(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            engine, model, data = (root / name for name in ("fake-engine", "model.gguf", "dev.json"))
            engine.write_text(ENGINE.format(python=sys.executable))
            engine.chmod(0o700)
            model.write_bytes(b"fake checkpoint for lifecycle test")
            data.write_text(json.dumps({"split": "dev", "cases": [
                {"id": "a", "question": "1+1?", "answer": "2"}]}))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = {"engine": str(engine), "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "model": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "backend_commit": "fake-test", "data": str(data), "output": str(root / "run"),
                "condition": "fake-test", "port": port, "startup_timeout": 10, "request_timeout": 5}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            command = [sys.executable, str(pathlib.Path(__file__).with_name("launch.py")), str(path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((root / "run/process-status.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertTrue(status["engine_stopped"])
            self.assertEqual(json.loads((root / "run/summary.json").read_text())["correct"], 1)
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            with socket.socket() as sock:
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)
            config.update(engine_sha256="0" * 64, output=str(root / "bad"))
            path.write_text(json.dumps(config))
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("engine checksum mismatch", result.stderr)
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
