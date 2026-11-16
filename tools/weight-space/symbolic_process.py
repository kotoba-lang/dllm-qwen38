"""Bounded subprocess wrapper for the separate synthetic symbolic scorer."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from symbolic_score import PROFILE, score


def unavailable(reason):
    return dict(profile=PROFILE, status='grader_unavailable', correct=None,
                reason=reason)


def score_isolated(gold, content, *, thinking_open=False, timeout_seconds=20):
    """Run parser/comparator on a child main thread with an execution deadline.

    Process creation/termination has OS overhead, so this is not a real-time
    scheduling guarantee. It bounds computation beyond the parser's own alarms.
    Raw expressions and child diagnostics never appear in the returned result.
    """
    if not isinstance(gold, str) or not gold or len(gold.encode()) > 4096:
        raise ValueError('invalid_reference_size')
    if not isinstance(content, str):
        raise ValueError('invalid_prediction_type')
    if len(content.encode()) > 65536:
        return dict(profile=PROFILE, status='prediction_too_long', correct=False)
    if not isinstance(thinking_open, bool):
        raise ValueError('invalid_thinking_state')
    if (not isinstance(timeout_seconds, (int, float)) or
            not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
        raise ValueError('invalid_timeout')
    payload=json.dumps(dict(gold=gold, content=content, thinking_open=thinking_open))
    deadline=time.monotonic()+timeout_seconds
    with subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as child:
        try:
            stdout, _=child.communicate(payload, timeout=max(0, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            return unavailable('process_timeout')
        if child.returncode:
            return unavailable('process_exit')
    try:
        result=json.loads(stdout)
        if (set(result)-{'profile','status','correct','error_type','reason'} or
                result['profile'] != PROFILE or
                not isinstance(result['status'], str) or
                not (result['correct'] is None or type(result['correct']) is bool)):
            raise ValueError('unexpected_worker_result')
    except (ValueError, TypeError, KeyError):
        return unavailable('invalid_process_result')
    return result


def worker():
    try:
        request=json.loads(sys.stdin.read(300000))
        result=score(request['gold'], request['content'],
                     thinking_open=request['thinking_open'])
    except Exception as exc:
        result=unavailable('worker_exception')
        result['error_type']=type(exc).__name__
    print(json.dumps(result))


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use score_isolated from the pinned symbolic environment.')
    worker()
