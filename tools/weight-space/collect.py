#!/usr/bin/env python3
"""Collect completed managed runs under an explicit comparison plan; no inference."""
import argparse
import contextlib
import io
import json
import math
import pathlib

import bench


def collect(plan_path):
    plan_path = pathlib.Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    roots, manifests, launches, records, runs, frozen = {}, {}, {}, {}, {}, {}
    for item in plan['conditions']:
        name = item['name']
        if name in roots:
            raise ValueError('duplicate condition name')
        root = (plan_path.parent / item['run']).resolve()
        paths = [root / f for f in ['manifest.json', 'launch.json', 'process-status.json', 'results.jsonl', 'summary.json']]
        hashes = {p.name: bench.digest(p) for p in paths}
        process = json.loads(paths[2].read_text())
        if process['state'] != 'completed' or not process.get('engine_stopped') or type(process.get('engine_exit_code')) is not int:
            raise ValueError(f'{name}: run not completed and stopped')
        identity = json.loads(paths[0].read_text())['identity']
        launch = json.loads(paths[1].read_text())['config']
        if (not launch.get('engine_sha256') or launch.get('model_sha256') != identity['model_sha256']
                or launch.get('backend_commit') != identity['backend_commit']):
            raise ValueError(f'{name}: launch/manifest provenance mismatch')
        data = pathlib.Path(launch['data'])
        if not data.is_absolute() or bench.digest(data) != identity['data_sha256']:
            raise ValueError(f'{name}: dataset checksum mismatch or nonabsolute path')
        frozen[data] = identity['data_sha256']
        rows = [json.loads(s) for s in paths[3].read_text().splitlines()]
        if not rows or [r['id'] for r in rows] != identity['case_ids']:
            raise ValueError(f'{name}: incomplete or reordered case list')
        actual, stored = bench.summary(rows), json.loads(paths[4].read_text())
        for key, value in actual.items():
            if isinstance(value, float):
                if not isinstance(stored.get(key), (int, float)) or not math.isclose(value, stored[key], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError(f'{name}: stored summary mismatch: {key}')
            elif stored.get(key) != value:
                raise ValueError(f'{name}: stored summary mismatch: {key}')
        if not identity.get('bench_sha256'):
            raise ValueError(f'{name}: missing scorer revision')
        roots[name], manifests[name], launches[name], records[name] = root, identity, launch, rows
        runs[name] = dict(identity=identity, summary=actual, artifact_sha256=hashes,
                          runtime={k:launch.get(k) for k in ['engine_sha256','engine_args','backend_commit']},
                          engine_exit_code=process['engine_exit_code'], engine_stopped=True)
        frozen.update({p: hashes[p.name] for p in paths})
    paired = []
    for left, right in plan['paired']:
        # The original two-run comparator lacks these provenance checks. Keep
        # collection stricter without changing a scorer during a running trial.
        for key in ['bench_sha256']:
            if manifests[left][key] != manifests[right][key]:
                raise ValueError(f'{left}/{right}: {key} differs')
        for key in ['engine_sha256', 'engine_args', 'backend_commit']:
            if launches[left].get(key) != launches[right].get(key):
                raise ValueError(f'{left}/{right}: launch {key} differs')
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            bench.compare(argparse.Namespace(baseline=str(roots[left]), candidate=str(roots[right]), output=None))
        paired.append(dict(baseline=left, candidate=right, result=json.loads(printed.getvalue())))
    no_ops = []
    for left, right in plan.get('no_op', []):
        if [left, right] not in plan['paired']:
            raise ValueError('no-op must also have a paired comparison')
        a, b = records[left], records[right]
        if manifests[left]['case_ids'] != manifests[right]['case_ids']:
            raise ValueError('no-op case lists differ')
        no_ops.append(dict(baseline=left, candidate=right,
            checkpoint_hash_equal=manifests[left]['model_sha256'] == manifests[right]['model_sha256'],
            exact_response_text_matches=sum(x['response']['content'] == y['response']['content'] for x, y in zip(a,b)),
            n=len(a)))
    if any(bench.digest(p) != expected for p, expected in frozen.items()):
        raise ValueError('run changed while collecting')
    return dict(schema='wsp.completed-run-bundle/v1', plan_sha256=bench.digest(plan_path),
        collector_sha256=bench.digest(__file__), comparison_code_sha256=bench.digest(bench.__file__),
        qualification=plan['qualification'], runs=runs, paired=paired, no_op=no_ops,
        limitations=['Stored grades and measured timing are aggregated; no new inference or regrading.',
                     'Completion records and hashes are checked; model files are not rehashed by this read-only collector.',
                     'Small or selected development subsets do not establish held-out improvement.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    a = p.parse_args()
    result = collect(a.plan)
    bench.write_new(a.output, result)
    print(json.dumps({name:run['summary'] for name,run in result['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
