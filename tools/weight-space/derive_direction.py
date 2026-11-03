#!/usr/bin/env python3
"""Derive one explicitly selected residual direction from saved development probes."""
import argparse
import json
import pathlib

import numpy as np

import bench
from capture_data import load_vectors


def derive(capture, output, layer, pairs, reset, minimum):
    if layer not in (31, 35, 39) or not np.isfinite(minimum) or not -1 <= minimum <= 1:
        raise ValueError('invalid frozen layer or cosine gate')
    indices = [i for pair in pairs for i in pair] + list(reset)
    if len(pairs) < 2 or any(len(pair) != 2 for pair in pairs) or len(set(indices)) != len(indices) or set(indices) != set(range(8)):
        raise ValueError('require three disjoint probe pairs and one reset pair covering eight inputs')
    vectors, accounting = load_vectors(capture)
    name = f'l_out-{layer}'
    reset_delta = float(np.max(np.abs(vectors[(reset[0], name)] - vectors[(reset[1], name)])))
    if reset_delta > 1e-5:
        raise ValueError('identical-prefix reset control failed')
    differences = [vectors[(late, name)] - vectors[(early, name)] for early, late in pairs]
    norms = [float(np.linalg.norm(v)) for v in differences]
    if not np.isfinite(norms).all() or min(norms) <= 1e-8:
        raise ValueError('no observable within-prompt contrast')
    unit = [v / norm for v, norm in zip(differences, norms)]
    cosines = [float(np.dot(unit[i], unit[j])) for i in range(len(unit)) for j in range(i+1, len(unit))]
    mean = float(np.mean(cosines))
    direction = np.mean(unit, axis=0)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError('mean direction vanished')
    direction /= norm
    output = pathlib.Path(output)
    output.mkdir()  # A changed declaration needs a fresh output directory.
    report = dict(layer=layer, pairs=pairs, reset_pair=reset, minimum_mean_pair_cosine=minimum,
                  pair_delta_norms=norms, pair_cosines=cosines, mean_pair_cosine=mean,
                  reset_control_max_abs=reset_delta, accounting=accounting,
                  source_sha256=bench.digest(__file__), capture_file_sha256={
                      p.name: bench.digest(p) for p in sorted(pathlib.Path(capture).iterdir()) if p.is_file()},
                  gate_passed=mean >= minimum,
                  qualification='Saved development-prefix association only; not proof of causal obstruction or performance gain')
    if report['gate_passed']:
        target = output / f'direction-{layer}.npy'
        np.save(target, direction)
        report['direction_sha256'] = bench.digest(target)
    (output / 'derivation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def pair(value):
    parsed = tuple(map(int, value.split(':')))
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError('use early:late input indexes')
    return parsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True, type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    p.add_argument('--layer', required=True, type=int)
    p.add_argument('--pairs', nargs=3, required=True, type=pair)
    p.add_argument('--reset', required=True, type=pair)
    p.add_argument('--min-mean-cosine', required=True, type=float)
    a = p.parse_args()
    result = derive(a.capture, a.output, a.layer, a.pairs, a.reset, a.min_mean_cosine)
    print(json.dumps({k:result[k] for k in ['layer','gate_passed','mean_pair_cosine','reset_control_max_abs']}, indent=2))


if __name__ == '__main__':
    main()
