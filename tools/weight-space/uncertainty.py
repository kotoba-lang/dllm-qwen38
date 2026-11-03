#!/usr/bin/env python3
"""Supplementary paired-delta bounds; does not alter frozen benchmark analyses."""
import argparse
import functools
import hashlib
import json
import math
import pathlib


@functools.lru_cache(maxsize=16)
def log_combinations(n):
    return [math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1) for k in range(n+1)]


def cdf(n, k, p):
    if p == 0: return 1.0
    if p == 1: return float(k == n)
    lp, lq = math.log(p), math.log1p(-p)
    coefficients = log_combinations(n)
    return math.fsum(math.exp(coefficients[i]+i*lp+(n-i)*lq) for i in range(k+1))


def upper_bound(k, n, tail):
    if k == n: return 1.0
    if k == 0: return -math.expm1(math.log(tail)/n)
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low+high)/2
        if cdf(n, k, middle) > tail: low = middle
        else: high = middle
    return (low+high)/2


def binomial_interval(k, n, confidence=0.95):
    if type(n) is not int or type(k) is not int or not 1 <= n <= 10000 or not 0 <= k <= n:
        raise ValueError('require integer counts0<=k<=n,1<=n<=10000')
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between0 and1')
    tail = (1-confidence)/2
    return [1-upper_bound(n-k,n,tail), upper_bound(k,n,tail)]


def paired_interval(n, wins, losses, confidence=0.95):
    if any(type(x) is not int for x in [n,wins,losses]) or min(wins,losses)<0 or wins+losses>n:
        raise ValueError('invalid paired counts')
    # Two marginal intervals cover jointly with probability at least confidence
    # by the union bound, without assuming independence of win/loss indicators.
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between0 and1')
    marginal_confidence = 1-(1-confidence)/2
    w = binomial_interval(wins,n,marginal_confidence)
    l = binomial_interval(losses,n,marginal_confidence)
    return dict(n=n,wins=wins,losses=losses,observed_delta=(wins-losses)/n,
                confidence=confidence,interval=[w[0]-l[1],w[1]-l[0]],
                marginal_confidence=marginal_confidence,win_interval=w,loss_interval=l)


def analyze(path, confidence=0.95):
    path = pathlib.Path(path)
    bundle = json.loads(path.read_text())
    if bundle.get('schema') != 'wsp.completed-run-bundle/v1':
        raise ValueError('requires a completed collector bundle')
    results = []
    for pair in bundle['paired']:
        source = pair['result']; n = source['baseline']['n']
        for name in [pair['baseline'],pair['candidate']]:
            ids=bundle['runs'][name]['identity']['case_ids']
            if len(ids)!=n or len(set(ids))!=n: raise ValueError('unique problem counts differ')
        if source['candidate']['n'] != n: raise ValueError('paired sizes differ')
        result=paired_interval(n,source['wins'],source['losses'],confidence)
        result.update(baseline=pair['baseline'],candidate=pair['candidate'],
                      frozen_bootstrap_interval=source['bootstrap_95_interval'],
                      frozen_mcnemar_p=source['mcnemar_exact_two_sided_p'])
        results.append(result)
    return dict(schema='wsp.supplementary-uncertainty/v1',
        source_bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        analysis_code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        comparisons=results,method='Binomial-tail inversion for win/loss marginals, each at1-alpha/2 confidence; subtract endpoints using Bonferroni joint coverage.',
        references=['https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm',
                    'https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm'],
        qualifications=['Supplementary sensitivity analysis; frozen bootstrap and McNemar outputs are retained, not replaced.',
                        'Coverage requires independent identically distributed problem pairs. Failure-selected development screens do not justify population inference.',
                        'This conservative interval is not a tight exact paired-risk-difference interval.',
                        'Confidence is per comparison, not adjusted across model, trial or benchmark families.',
                        'No new inference or regrading.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=pathlib.Path)
    p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--confidence',type=float,default=0.95)
    a=p.parse_args();result=analyze(a.bundle,a.confidence)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
