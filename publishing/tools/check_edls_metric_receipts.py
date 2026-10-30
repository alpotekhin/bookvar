"""Execute the actual teaching receipt fence; no HTTP service or GPU timing."""
from pathlib import Path
from math import isclose, nan, inf
from types import SimpleNamespace
import ast
import contextlib
import io
import random
import re
import typing

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '06 Практика/14 Собрать минимальный inference engine.md'
text = chapter.read_text()
code = re.search(r'<!-- example: edls-metric-receipts -->\s*```python\n(.*?)\n```', text, re.S).group(1)
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, str(chapter), 'exec'), ns)
report, receipts = ns['request_report'], ns['receipts']
rows, totals = report(receipts, 0., 1.)
by_id = {r['id']: r for r in rows}
assert totals['offered'] == 5 and totals['completed'] == 3
assert totals['outcomes'] == dict(completed=3, cancelled=1, rejected=1, error=0)
assert totals['emitted_tokens'] == 7 and totals['completed_tokens'] == 6
assert totals['output_tokens_per_second'] == 7 and totals['completed_requests_per_second'] == 3
assert isclose(totals['mean_request_tpot'], .2)
assert isclose(totals['pooled_mean_itl'], .7 / 3)
assert isclose(by_id['A']['queue_wait'], .1)
assert isclose(by_id['A']['ttft'], .3) and isclose(by_id['A']['tpot'], .3)
assert all(isclose(x, y) for x, y in zip(by_id['A']['itls'], [.2, .4]))
assert isclose(by_id['B']['ttft'], .7) and by_id['B']['tpot'] is None
assert by_id['C']['ttft'] is None and by_id['C']['queue_wait'] is None
assert isclose(by_id['D']['ttft'], .75)
assert isclose(by_id['E']['tpot'], .1)
mean_slo = [r['id'] for r in rows if r['status'] == 'completed'
            and r['ttft'] is not None and r['ttft'] <= .4
            and (r['tpot'] is None or r['tpot'] <= .35)]
strict_slo = [r['id'] for r in rows if r['id'] in mean_slo
              and all(dt <= .35 for dt in r['itls'])]
assert mean_slo == ['A', 'E'] and strict_slo == ['E']

empty_rows, empty = report([], 10, 12)
assert empty_rows == [] and empty['output_tokens_per_second'] == 0
assert empty['mean_request_tpot'] is None and empty['pooled_mean_itl'] is None
zero = dict(id='zero', submitted=0., admitted=0., tokens=[], finished=0., status='completed')
assert report([zero], 0, 1)[0][0]['tpot'] is None
assert report([zero], 0, 1)[0][0]['ttft'] is None
assert report([{**zero, 'status': 'error'}], 0, 1)[1]['completed'] == 0

def rejected(records, start=0, end=1):
    try:
        report(records, start, end)
    except ValueError:
        return
    raise AssertionError('invalid receipt was accepted')

for bad in [dict(status='pending'), dict(submitted=-.1), dict(finished=2),
            dict(tokens=[.5, .4]), dict(tokens=[nan]), dict(finished=inf),
            dict(admitted=.8), dict(admitted=None), dict(status='rejected')]:
    rejected([{**receipts[0], **bad}])
rejected([receipts[0], receipts[0]])
for start, end in [(0, 0), (1, 0), (nan, 1), (0, inf)]:
    rejected([], start, end)

# Affine clock changes preserve counts and scale durations/rates as expected.
rng = random.Random(26)
for _ in range(200):
    scale, shift = rng.uniform(.1, 100), rng.uniform(100, 1000)
    shifted = [{**r, 'submitted': shift + scale*r['submitted'],
                'admitted': None if r['admitted'] is None else shift + scale*r['admitted'],
                'finished': shift + scale*r['finished'],
                'tokens': [shift + scale*t for t in r['tokens']]} for r in receipts]
    rr, tt = report(shifted, shift, shift + scale)
    assert tt['outcomes'] == totals['outcomes']
    assert isclose(tt['output_tokens_per_second'], 7 / scale)
    for actual, expected in zip(rr, rows):
        for metric in ['ttft', 'tpot', 'queue_wait', 'latency']:
            if expected[metric] is None:
                assert actual[metric] is None
            else:
                assert isclose(actual[metric], scale*expected[metric], abs_tol=1e-10)

# Only the reviewed pure config/baseline functions are executed from originals.
archive = ROOT / '05 Источники/Originals/edls-inference-2026-10-26'
benchmark = ast.parse((archive / 'test_benchmark.py').read_text())
mode = next(n for n in benchmark.body if isinstance(n, ast.FunctionDef) and n.name == 'get_mode_config')
source_ns = dict(Optional=typing.Optional, Dict=typing.Dict, Any=typing.Any)
exec(compile(ast.fix_missing_locations(ast.Module(body=[mode], type_ignores=[])), 'source mode', 'exec'), source_ns)
assert source_ns['get_mode_config']('hard_prefill') == dict(max_new_tokens=20, prompt_length=2000)
assert source_ns['get_mode_config']('hard_decode') == dict(max_new_tokens=2000, prompt_length=50)
assert source_ns['get_mode_config']('medium') == dict(max_new_tokens=200, prompt_length=200)
scheduler = ast.parse((archive / 'scheduler_manager.py').read_text())
policy = next(n for n in ast.walk(scheduler) if isinstance(n, ast.FunctionDef) and n.name == '_decide_prefill_batch_size')
exec(compile(ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])), 'source policy', 'exec'), source_ns)
baseline = source_ns['_decide_prefill_batch_size']
assert baseline(SimpleNamespace(active_requests=[])) == 1
assert baseline(SimpleNamespace(active_requests=[SimpleNamespace(is_finished=False)])) == 0
assert baseline(SimpleNamespace(active_requests=[SimpleNamespace(is_finished=True)])) == 1

print('PASS actual receipt fence; TTFT/ITL/TPOT, undefined values, outcomes, goodput, '
      'invalid receipts and 200 affine-clock cases; reviewed source modes/baseline')
print('Assigned timestamps only; no HTTP run, policy performance result or GPU benchmark')
