"""Fixed lexical train-derived split audit; never prints problem/answer text."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

PROFILE='nfkc-casefold-number-mask-token5-jaccard080-v1'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def features(text):
    plain=normalized(text)
    masked=re.sub(r'\d+(?:\.\d+)?','NUMBER',plain)
    tokens=re.findall(r'\w+|[^\w\s]',masked)
    shingles={tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))}
    return plain,masked,shingles


def compare(left,right):
    a,b=features(left),features(right)
    overlap=len(a[2]&b[2]);union=len(a[2]|b[2])
    similarity=overlap/union if union else 0
    flags=[]
    if a[0]==b[0]:flags.append('normalized_exact')
    if a[1]==b[1]:flags.append('number_masked_exact')
    if similarity>=0.8:flags.append('number_masked_token5_jaccard_ge_080')
    return flags,similarity


def load(path):
    data=Path(path).read_bytes();doc=json.loads(data)
    cases=doc['cases']
    if not cases or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('empty or duplicate case IDs')
    if any(not isinstance(c['question'],str) or not c['question'].strip() for c in cases):
        raise ValueError('invalid question')
    return cases,dict(file=Path(path).name,sha256=hashlib.sha256(data).hexdigest(),
                      split=doc['split'],n=len(cases))


def audit(dev_paths,validation_paths):
    inputs=[];groups=[]
    for label,paths in [('dev',dev_paths),('validation',validation_paths)]:
        rows=[]
        for index,path in enumerate(paths):
            cases,meta=load(path)
            if meta['split']!=label:raise ValueError('split label mismatch')
            meta.update(role=label,index=index);inputs.append(meta)
            rows.extend((index,c) for c in cases)
        groups.append(rows)
    flags=[];maximum=0;pair_count=0
    for (di,d),(vi,v) in itertools.product(*groups):
        pair_count+=1;reasons,similarity=compare(d['question'],v['question'])
        maximum=max(maximum,similarity)
        if reasons:
            flags.append(dict(dev_input=di,dev_id=d['id'],validation_input=vi,
                              validation_id=v['id'],reasons=reasons,jaccard=similarity))
    return dict(profile=PROFILE,inputs=inputs,cross_split_pair_count=pair_count,
                flagged_pair_count=len(flags),flagged_pairs=flags,
                maximum_number_masked_token5_jaccard=maximum,
                limitations=['A fixed lexical heuristic, not semantic equivalence or pretraining-contamination proof.',
                             'Only the supplied train-derived files are checked; final sets are not accessed.',
                             'No split changes or candidate tuning are authorized by this report.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev',nargs='+',required=True)
    parser.add_argument('--validation',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.dev,args.validation)
    with open(args.output,'x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:result[k] for k in ['profile','cross_split_pair_count','flagged_pair_count']}))


if __name__=='__main__':main()
