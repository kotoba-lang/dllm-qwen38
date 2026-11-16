"""Separate boxed-answer scoring prototype; never regrades frozen numeric runs."""
import functools
import importlib.metadata
import re
import threading

PROFILE = 'boxed-math-verify-0.9.0-round12-v1'
VERSIONS = {'math-verify':'0.9.0','latex2sympy2-extended':'1.11.0',
            'antlr4-python3-runtime':'4.13.2','sympy':'1.14.0','mpmath':'1.3.0'}


@functools.lru_cache(maxsize=1)
def check_versions():
    if any(importlib.metadata.version(name)!=version for name,version in VERSIONS.items()):
        raise RuntimeError('symbolic grader versions differ from lock')


def outside_thinking(text, thinking_open=False):
    depth=int(thinking_open);cursor=0;parts=[]
    for match in re.finditer(r'</?think>',text):
        if depth==0:parts.append(text[cursor:match.start()])
        if match.group()=='<think>':
            if depth:raise ValueError('nested_thinking')
            depth=1
        else:
            if not depth:raise ValueError('unmatched_thinking_close')
            depth=0
        cursor=match.end()
    if depth:raise ValueError('unfinished_thinking')
    parts.append(text[cursor:]);return ''.join(parts)


def final_box(text):
    cursor=0;last=None
    while (match:=re.search(r'\\boxed\s*\{',text[cursor:])) is not None:
        start=cursor+match.start();end=cursor+match.end();depth=1;i=end
        preceding=0;j=start-1
        while j>=0 and text[j]=='\\':preceding+=1;j-=1
        if preceding%2:cursor=end;continue
        while i<len(text) and depth:
            if text[i]=='\\' and i+1<len(text):i+=2;continue
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            i+=1
        if depth:raise ValueError('unfinished_box')
        last=text[start:i];cursor=i
    if last is None:raise ValueError('missing_box')
    return last


def score(gold, content, *, thinking_open=False):
    """Return tri-state correctness; grader errors are unavailable, not wrong answers.

    Requires the main thread (the pinned parser uses POSIX alarms). Each parse
    and comparison has its own 5-second limit; callers need a separate process
    deadline for a hard whole-case budget. No raw answers are returned.
    """
    check_versions()
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('symbolic grading requires main thread')
    if not gold or len(gold.encode())>4096:raise ValueError('invalid_reference_size')
    if len(content.encode())>65536:return dict(profile=PROFILE,status='prediction_too_long',correct=False)
    from math_verify import LatexExtractionConfig,parse,verify
    config=[LatexExtractionConfig(boxed_match_priority=0)]
    def parsed(text):
        return parse(text,extraction_config=config,fallback_mode='no_fallback',
                     extraction_mode='first_match',parsing_timeout=5,raise_on_error=True)
    try:reference=parsed(r'\boxed{'+gold+'}')
    except Exception as exc:raise ValueError('reference_parser_error') from exc
    if not reference:raise ValueError('reference_unparsed')
    try:answer=final_box(outside_thinking(content,thinking_open))
    except ValueError as exc:return dict(profile=PROFILE,status=str(exc),correct=False)
    try:
        prediction=parsed(answer)
        if not prediction:return dict(profile=PROFILE,status='prediction_unparsed',correct=False)
        correct=verify(reference,prediction,float_rounding=12,numeric_precision=30,
                       strict=True,allow_set_relation_comp=False,timeout_seconds=5,raise_on_error=True)
    except Exception as exc:return dict(profile=PROFILE,status='grader_unavailable',correct=None,error_type=type(exc).__name__)
    return dict(profile=PROFILE,status='scored',correct=bool(correct))
