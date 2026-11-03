#!/usr/bin/env python3
"""Run one JSON-configured bench and stop only the engine started by this run."""
import argparse
import datetime
import json
import pathlib
import socket
import subprocess
import time

import bench


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=pathlib.Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if bench.digest(config["engine"]) != config["engine_sha256"]:
        raise ValueError("engine checksum mismatch")
    if bench.digest(config["model"]) != config["model_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    port = int(config.get("port", 17977))
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    output = pathlib.Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    launch_file = output / "launch.json"
    if launch_file.exists():
        if json.loads(launch_file.read_text())["config"] != config:
            raise ValueError("launch config changed; use a new output directory")
    else:
        bench.write_new(launch_file, {"config": config, "launcher_sha256": bench.digest(__file__)})
    command = [config["engine"], *config.get("engine_args", []),
               "-m", config["model"], "--host", "127.0.0.1", "--port", str(port)]
    base = f"http://127.0.0.1:{port}"
    status = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command, "state": "starting"}
    status_file = output / "process-status.json"

    def save():
        status_file.write_text(json.dumps(status, indent=2) + "\n")

    with (output / "engine.log").open("a") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        status["pid"] = proc.pid
        save()
        try:
            deadline = time.monotonic() + config.get("startup_timeout", 300)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"engine exited {proc.returncode}; see engine.log")
                try:
                    bench.request(base, "/health", timeout=2)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("engine startup deadline; see engine.log")
                    time.sleep(1)
            status["state"] = "running"
            save()
            print(json.dumps(status), flush=True)
            bench.run(argparse.Namespace(server=base, model=config["model"],
                model_sha256=config["model_sha256"], backend_commit=config["backend_commit"],
                condition=config["condition"], data=config["data"], output=str(output),
                limit=config.get("limit", 0), max_tokens=config.get("max_tokens", 1024),
                thinking=config.get("thinking", "off"), seed=config.get("seed", 42),
                presence_penalty=config.get("presence_penalty", 0.0),
                repeat_penalty=config.get("repeat_penalty", 1.0),
                timeout=config.get("request_timeout", 600)))
            status["state"] = "completed"
        except BaseException as exc:
            status["state"] = "incomplete"
            status["error"] = str(exc)
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            status["engine_exit_code"] = proc.returncode
            status["engine_stopped"] = True
            status["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
