"""Run practice 14's actual CPU fences; verify its source proxies separately."""
import ast
from collections import Counter
import contextlib
from fractions import Fraction
from hashlib import sha256
import io
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / '06 Практика/14 Собрать минимальный inference engine.md'
text = path.read_text()
blocks = [(name, code) for name, code in re.findall(
    r'<!-- example: ([^>]+) -->\n```python\n(.*?)\n```', text, re.S)
    if name.startswith('inference-practice-')]
assert len(blocks) == 5
namespaces, outputs = {}, {}
for name, code in blocks:
    ast.parse(code)
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(code, f'{path.name}:{name}', 'exec'), ns)
    namespaces[name], outputs[name] = ns, out.getvalue().splitlines()

assert outputs['inference-practice-cost'] == [
    'baseline/day 50000', 'crossover/days 40',
    'optimized crossover/days 44.44', 'annual saving 1825000']
assert Fraction(50_000_000, 86400) * 86400 * Fraction(1, 1000) == 50_000
assert Fraction(2_000_000, 45_000) == Fraction(400, 9)
assert 50_000_000 * 365 == 18_250_000_000
assert outputs['inference-practice-memory'] == [
    '4096 1342177280 27', '8192 2684354560 13', '16 64.47 True', '32 85.95 False']
memory = namespaces['inference-practice-memory']
for context, maximum in [(4096, 27), (8192, 13)]:
    cost = 2 * 80 * 8 * 128 * 2 * context
    assert 35_000_000_000 + maximum * cost + 8_000_000_000 <= 80_000_000_000
    assert 35_000_000_000 + (maximum + 1) * cost + 8_000_000_000 > 80_000_000_000

assert outputs['inference-practice-scheduler'] == [
    'refill False ticks 7', "{'A': 2, 'B': 5, 'C': 6, 'D': 7}",
    'refill True ticks 5', "{'A': 2, 'B': 5, 'C': 3, 'D': 5}"]
schedule = namespaces['inference-practice-scheduler']['schedule']
jobs = namespaces['inference-practice-scheduler']['jobs']
traces = [schedule(jobs, 2, refill)[0] for refill in [False, True]]
assert traces[0] == [(0, ('A', 'B')), (1, ('A', 'B')), (2, ('B',)),
                     (3, ('B',)), (4, ('B',)), (5, ('C', 'D')), (6, ('D',))]
assert traces[1] == [(0, ('A', 'B')), (1, ('A', 'B')), (2, ('B', 'C')),
                     (3, ('B', 'D')), (4, ('B', 'D'))]
assert Fraction(10, 14) == Fraction(5, 7)
assert Fraction(10, 10) == 1
rng = random.Random(14)
for _ in range(150):
    fixture = [(f'r{i}', rng.randrange(20), rng.randrange(1, 12))
               for i in range(rng.randrange(1, 15))]
    slots = rng.randrange(1, 5)
    for refill in [False, True]:
        trace, emitted = schedule(fixture, slots, refill)
        counts = Counter(name for _, names in trace for name in names)
        for name, arrival, length in fixture:
            assert len(emitted[name]) == counts[name] == length
            assert emitted[name][0] >= arrival + 1
            assert emitted[name] == sorted(set(emitted[name]))
            assert all(b == a + 1 for a, b in zip(emitted[name], emitted[name][1:]))
        assert all(len(names) <= slots and len(names) == len(set(names)) for _, names in trace)
for bad_jobs, slots in [([('a', 0, 1)], 0), ([('a', 0, 1)], 1.5),
                        ([('a', 0, 1), ('a', 1, 1)], 2),
                        ([('a', -1, 2)], 1), ([('a', 0, 0)], 1),
                        ([('a', 0, .5)], 1)]:
    try:
        schedule(bad_jobs, slots, True)
    except ValueError:
        pass
    else:
        raise AssertionError((bad_jobs, slots))
assert schedule([], 2, True) == ([], {})
assert schedule([('a', 10, 1)], 2, True)[1] == {'a': [11]}

assert outputs['inference-practice-feasibility'] == [
    "A 2 288 ['cost']", "B 2 144 ['memory', 'p99']",
    "C 2 144 ['quality']", "D 3 216 ['pass']"]
assert outputs['inference-practice-cold-start'] == [
    'queued requests 6000', 'seconds to drain after ready 75.0']
assert Fraction(100, 8 * 5) == Fraction(5, 2)

# Execute only reviewed pure source helpers, not Marimo/bootstrap/widgets.
source = ROOT / '05 Источники/Originals/inference-practice-2026-10-26/harvard-lab10.py'
tree = ast.parse(source.read_text())
selected = {'v2_10_schedule_label', 'v2_10_policy_multiplier',
            'v2_10_scheduling_rows', 'v2_10_serving_latency_ms',
            'v2_10_binding_constraint', 'v2_10_phase_amounts'}
nodes = {node.name: node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name in selected}
assert set(nodes) == selected
module = ast.fix_missing_locations(ast.Module(body=list(nodes.values()), type_ignores=[]))
source_ns = {'math': math}
exec(compile(module, str(source), 'exec'), source_ns)
assert source_ns['v2_10_policy_multiplier']('cloud_fleet', 'continuous') == 3.0
for values, expected in [((False, False, False, False), 'memory/state'),
                         ((True, False, True, True), 'SLO/deadline'),
                         ((True, True, True, False), 'quality guardrail'),
                         ((True, True, False, True), 'recurring cost')]:
    assert source_ns['v2_10_binding_constraint'](*values) == expected
rows, winner = source_ns['v2_10_scheduling_rows']('cloud_fleet', 8, 1024, 4096, .8, 100)
assert len(rows) == 4 and winner in {r['policy'] for r in rows}
assert math.isinf(source_ns['v2_10_serving_latency_ms']('cloud_fleet', 100, 'static', .8, True))
phase = source_ns['v2_10_phase_amounts'](
    SimpleNamespace(track_id='cloud_fleet', model_params_b=7,
                    memory_bandwidth_gbs=2000, slo_ms=100), 1024, 10, 2)
assert phase['tpot_ms'] == 7 and phase['decode_ms'] == 70
assert phase['prefill_ms'] == 10.015 and phase['binding_phase'] == 'decode'

prose = re.sub(r'```[^\n]*\n.*?\n```', '', text, flags=re.S)
for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', prose):
    target = match.group(1).split('#')[0].removeprefix('02 Areas/ML & DL/')
    assert (ROOT / target).exists() or (ROOT / (target + '.md')).exists(), target
for row in ['| 16 | 56.47 | 64.47 | yes, in this accounting |',
            '| 32 | 77.95 | 85.95 | no |', '| D | 3 | 216 | none |']:
    assert row in text
print('PASS five actual CPU fences, 300 scheduler fixtures, invalid inputs, '
      'memory boundaries, source proxy helpers and worked tables; no GPU benchmark')
print('chapter_sha256=' + sha256(path.read_bytes()).hexdigest())
