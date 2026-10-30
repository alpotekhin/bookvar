"""Execute the actual practice fences on CPU; no GPU benchmarking or cloud API."""
import ast
import contextlib
from hashlib import sha256
import io
import math
from pathlib import Path
import re

import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / '06 Практика/13 Оптимизировать один Transformer step.md'
body = path.read_text()
blocks = re.findall(r'```python\n(.*?)\n```', body, re.S)
assert len(blocks) == 4
namespaces, outputs = [], []
for index, block in enumerate(blocks):
    ast.parse(block)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(block, f'{path.name}:python-{index}', 'exec'), namespace)
    namespaces.append(namespace)
    outputs.append(output.getvalue().splitlines())

assert outputs[0] == [
    'Mean difference, ms: -4.500',
    '95% CI, ms: [-6.907, -2.093]',
    'p-value: 0.001350',
]
ns = namespaces[0]
from statistics import mean, variance
a, b = ns['candidate'], ns['baseline']
va, vb = variance(a) / len(a), variance(b) / len(b)
se = math.sqrt(va + vb)
df = (va + vb)**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
delta = mean(a) - mean(b)
critical = t.ppf(.975, df)
assert math.isclose(float(ns['interval'].low), delta - critical * se)
assert math.isclose(float(ns['interval'].high), delta + critical * se)
assert math.isclose(float(ns['result'].df), df)

assert outputs[1] == [
    'CI half-width: 5.88 ms',
    'Approximate 80%-power shift: 8.40 ms',
    'Power at the half-width: 50.00%',
    'Planned observations per variant: 23',
]
plan = namespaces[1]
z, zp = plan['z_cut'], plan['z_power']
for n in (22, 23):
    attained = (z + zp) * 6 * math.sqrt(2/n)
    assert (attained <= 5) == (n == 23)
assert .50 < plan['power_at_half_width'] < .501

assert outputs[2] == [
    '5 19800 95.96% 4.21% 4900',
    '6 23760 79.97% 25.05% 5880',
]
capacity = namespaces[2]
assert capacity['required_units'] == 6
assert (5 * capacity['rate'] / capacity['demand'] - 1) < .25
assert (6 * capacity['rate'] / capacity['demand'] - 1) >= .25
assert math.isclose(1 / (1 + .25), .8)

assert outputs[3] == [
    "2.0 ['targeted', 'precision']", "1.0 ['targeted']",
    '0.5 hold: no feasible option',
]
feasible = namespaces[3]['feasible']
boundary = {'p99': 180, 'headroom': .25, 'cost': 6615, 'quality': 1.0}
assert feasible(boundary, 1.0)
for key, value in [('p99', 180.01), ('headroom', .2499), ('cost', 6616), ('quality', 1.01)]:
    assert not feasible(dict(boundary, **{key: value}), 1.0), key

# Match the rounded fixture rows to the original reviewed source helpers.
source_checker = ROOT / 'publishing/tools/check_harvard_performance_lab.py'
source_ns = {'__file__': str(source_checker)}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(source_checker.read_text(), str(source_checker), 'exec'), source_ns)
source_rows = source_ns['rows']
for actual, original in zip(namespaces[3]['rows'], source_rows, strict=True):
    assert actual['name'] == original['id']
    for field, source_field, digits in [('p99', 'p99_ms', 2), ('headroom', 'headroom', 4),
                                      ('cost', 'daily_cost', 0), ('quality', 'quality_loss_pct', 1)]:
        assert actual[field] == round(original[source_field], digits)

def table_rows(header):
    lines = body.split(header + '\n', 1)[1].split('\n\n', 1)[0].splitlines()[1:]
    return [[cell.strip() for cell in line.strip('|').split('|')] for line in lines]

capacity_table = table_rows('| Units | Assumed capacity, tokens/s | Utilization | Headroom | Cost/day, USD |')
assert capacity_table == [line.split() for line in outputs[2]]
candidate_table = table_rows('| Candidate | Assumed p99, ms | Headroom | Cost/day, USD | Assumed quality loss | Failed constraints |')
for actual, original in zip(candidate_table, source_rows, strict=True):
    assert float(actual[1]) == round(original['p99_ms'], 2)
    assert actual[2] == f"{100 * original['headroom']:.2f}%"
    assert float(actual[3]) == original['daily_cost']
    assert actual[4] == f"{original['quality_loss_pct']:.1f}%"
    expected_failures = ', '.join('latency' if v == 'p99' else v for v in original['failures']) or 'none'
    assert actual[5] == expected_failures

for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', body):
    target = match.group(1).split('#')[0].removeprefix('02 Areas/ML & DL/')
    if target:
        assert (ROOT / target).exists() or (ROOT / (target + '.md')).exists(), target
print('PASS four actual CPU Markdown examples; Welch interval, power/sample size, '
      'capacity bounds, constraint boundaries and source-derived fixtures')
print('SciPy', scipy.__version__, '; no CUDA or performance claim')
print('chapter_sha256=' + sha256(path.read_bytes()).hexdigest())
