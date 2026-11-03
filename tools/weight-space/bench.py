#!/usr/bin/env python3
"""Small, resumable local-model experiment; no training or remote inference."""
import argparse
import datetime
import decimal
import hashlib
import json
import math
import pathlib
import random
import re
import time
import urllib.parse
import urllib.request

DATA_REV = "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
DATA_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REV}/grade_school_math/data/train.jsonl"
NUMBER = r"[-+]?(?:\d[\d,]*)(?:\.\d+)?"


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_new(path, value):
    with pathlib.Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def prepare(args):
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(DATA_URL, timeout=30).read()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows.sort(key=lambda row: canonical(row["question"]))
    selected = rows[:args.size * 2]
    if len(selected) != args.size * 2 or args.size < 1:
        raise ValueError("invalid split size")
    if len({r["question"] for r in selected}) != len(selected):
        raise ValueError("duplicate questions")
    for split, subset in [("dev", selected[:args.size]), ("validation", selected[args.size:])]:
        cases = []
        for row in subset:
            answer = row["answer"].rsplit("####", 1)[1].strip()
            decimal.Decimal(answer.replace(",", ""))
            cases.append({"id": canonical(row["question"])[:20], "question": row["question"], "answer": answer})
        write_new(out / f"{split}.json", {"schema": "wsp.gsm8k-mini/v1", "source": DATA_URL,
                  "source_sha256": hashlib.sha256(raw).hexdigest(), "split": split,
                  "selection": "question SHA256 ascending; dev then validation", "cases": cases})
    print(json.dumps({"size_per_split": args.size, "source_sha256": hashlib.sha256(raw).hexdigest()}))


def final_number(text):
    # Never grade a number in unfinished reasoning as the final answer.
    if "<think>" in text and "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[-1]
    matches = re.findall(r"(?im)^\s*FINAL:\s*(" + NUMBER + r")\s*[.!]?\s*$", visible)
    if len(matches) != 1:
        return None
    return str(decimal.Decimal(matches[0].replace(",", "")).normalize())


def repeated_fourgram_fraction(text):
    words = text.split()
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def request(server, endpoint, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(server + endpoint, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def summary(records):
    correct = sum(r["correct"] for r in records)
    seconds = sum(r["wall_seconds"] for r in records)
    tokens = sum(r["response"].get("tokens_predicted", 0) for r in records)
    return {"n": len(records), "correct": correct, "accuracy": correct / len(records) if records else None,
            "wall_seconds": seconds, "generated_tokens": tokens,
            "seconds_per_correct": seconds / correct if correct else None,
            "tokens_per_correct": tokens / correct if correct else None,
            "limit_hits": sum(r["response"].get("stop_type") == "limit" or bool(r["response"].get("stopped_limit")) for r in records),
            "no_final_answer": sum(r["predicted"] is None for r in records)}


def run(args):
    if args.limit < 0 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("invalid run limit, token budget, or timeout")
    parsed = urllib.parse.urlparse(args.server)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("only an explicit loopback inference server is supported")
    if digest(args.model) != args.model_sha256:
        raise ValueError("model checksum mismatch")
    data = json.loads(pathlib.Path(args.data).read_text())
    cases = data["cases"][:args.limit] if args.limit else data["cases"]
    if not cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("empty or duplicate cases")
    props = request(args.server, "/props")
    loaded = props.get("model_path")
    if loaded is None or pathlib.Path(loaded).resolve() != pathlib.Path(args.model).resolve():
        raise ValueError(f"server model_path does not match requested checkpoint: {loaded}")
    settings = {"n_predict": args.max_tokens, "temperature": 0.0, "top_k": 1,
                "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "seed": args.seed, "cache_prompt": False, "stream": False}
    identity = {"schema": "wsp.local-run/v1", "data_sha256": digest(args.data),
                "case_ids": [c["id"] for c in cases], "split": data["split"],
                "settings": settings, "thinking": args.thinking, "backend_commit": args.backend_commit,
                "server_context": props.get("default_generation_settings", {}).get("n_ctx"),
                "model_sha256": args.model_sha256, "condition": args.condition,
                "bench_sha256": digest(__file__)}
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("resume manifest mismatch")
    else:
        write_new(manifest, {"identity": identity, "model_path": str(pathlib.Path(args.model).resolve()),
                  "server_props": props, "bench_sha256": digest(__file__),
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    rows_path = out / "results.jsonl"
    records = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if [r["id"] for r in records] != identity["case_ids"][:len(records)]:
        raise ValueError("results do not match manifest prefix")
    for case in cases[len(records):]:
        user = case["question"] + "\nGive your final answer on its own line as FINAL: <number>."
        assistant = "<think>\n" if args.thinking == "on" else "<think>\n</think>\n\n"
        prompt = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + assistant
        payload = {**settings, "prompt": prompt}
        start = time.monotonic()
        response = request(args.server, "/completion", payload, timeout=args.timeout)
        elapsed = time.monotonic() - start
        # The opening think tag lives in the prompt, not necessarily in response.content.
        content = response["content"]
        judged = ("<think>\n" + content) if args.thinking == "on" else content
        predicted = final_number(judged)
        expected = str(decimal.Decimal(case["answer"].replace(",", "")).normalize())
        record = {"id": case["id"], "prompt_sha256": canonical(prompt), "expected": expected,
                  "predicted": predicted, "correct": predicted == expected,
                  "wall_seconds": elapsed, "repeated_fourgram_fraction": repeated_fourgram_fraction(content),
                  "response": response}
        with rows_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        records.append(record)
        print(json.dumps({"completed": len(records), **summary(records)}), flush=True)
        (out / "summary.json").write_text(json.dumps(summary(records), indent=2) + "\n")


def compare(args):
    roots = [pathlib.Path(x) for x in (args.baseline, args.candidate)]
    manifests = [json.loads((p / "manifest.json").read_text())["identity"] for p in roots]
    for key in ("data_sha256", "case_ids", "split", "settings", "thinking", "backend_commit", "server_context"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"not a paired comparison: {key} differs")
    records = [[json.loads(line) for line in (p / "results.jsonl").read_text().splitlines()] for p in roots]
    if any([r["id"] for r in rows] != manifests[0]["case_ids"] for rows in records):
        raise ValueError("comparison requires complete matching runs")
    differences = [int(b["correct"]) - int(a["correct"]) for a, b in zip(*records)]
    rng = random.Random(797)
    boot = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(10000))
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    result = {"baseline": summary(records[0]), "candidate": summary(records[1]),
              "paired_accuracy_delta": sum(differences) / len(differences),
              "bootstrap_95_interval": [boot[250], boot[9749]], "wins": wins, "losses": losses,
              "mcnemar_exact_two_sided_p": p,
              "qualification": "development diagnostic; small n and selected candidates do not establish held-out improvement"}
    print(json.dumps(result, indent=2))
    if args.output:
        write_new(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=32)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    for flag in ("data", "model", "model-sha256", "backend-commit", "condition", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--server", default="http://127.0.0.1:17977")
    p.add_argument("--thinking", choices=("on", "off"), default="off")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=600)
    p.set_defaults(func=run)
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--output")
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
