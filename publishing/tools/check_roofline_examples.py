"""Check the chapter's cost arithmetic on CPU; never benchmark CUDA."""
import ast
import contextlib
import hashlib
import io
import math
from fractions import Fraction as Q
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parents[2]
path = root / "00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md"
body = path.read_text()
blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
assert len(blocks) == 6
for index, block in enumerate(blocks):
    ast.parse(block, filename=f"{path.name}:python-{index}")
cpu = [block for block in blocks if "def lower_bound_ms" in block]
assert len(cpu) == 1
ns = {}
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(compile(cpu[0], path.name, "exec"), ns)
assert output.getvalue().splitlines() == ["[2.4, 2.4]", "4.8", "3.0"]

# Independent rational arithmetic, in GFLOP/GB and milliseconds.
for f in (1, 12, 72, 288, 360):
    for q in (Q(1, 10), Q(3, 10), Q(9, 10), Q(36, 10), Q(39, 10)):
        expected = max(Q(f, 120), q / Q(3, 2))
        actual = ns["lower_bound_ms"](f * 10**9, float(q) * 10**9)
        assert math.isclose(actual, float(expected)), (f, q, actual, expected)
assert math.isclose(ns["sequential_ms"], 4.8)
assert math.isclose(ns["pooled_ms"], 3.0)

assert Q(1, 1) / (Q(8, 10) + Q(2, 10) / 3) == Q(15, 13)
times = [20] * 980 + [500] * 20
assert statistics.mean(times) == 29.6
assert statistics.median(times) == 20
assert sorted(times)[math.ceil(.99 * len(times)) - 1] == 500
assert sum(t > 50 for t in times) / len(times) == .02
assert Q(10 - 7, 10) == Q(3, 10) and Q(10, 7) > Q(14, 10)

f = Q(72)
for traffic, intensity, memory_ms, roof in (
    (Q(36, 10), 20, Q(24, 10), 30),
    (Q(9, 10), 80, Q(6, 10), 120),
):
    assert f / traffic == intensity
    assert traffic / Q(3, 2) == memory_ms
    assert min(120, Q(3, 2) * intensity) == roof
assert Q(24, 30) == Q(8, 10) and Q(72, 120) == Q(6, 10)

# Same work, different traffic at HBM and L2.
assert f / Q(9, 10) == 80 and f / Q(36, 10) == 20
assert max(f / 120, Q(9, 10) / Q(3, 2), Q(36, 10) / Q(45, 10)) == Q(8, 10)
assert f / Q(8, 10) == 90

n = 4096
assert 2 * n**3 == 137438953472
assert 2 * 3 * n**2 == 100663296 == 96 * 1024**2
for m, rounded in ((1, 1.00), (128, 120.47), (4096, 1365.33)):
    intensity = Q(2 * m * n * n, 2 * (m * n + n * n + m * n))
    assert round(float(intensity), 2) == rounded
blocks_1792 = math.ceil(1792 / 256) * math.ceil(1792 / 128)
blocks_1793 = math.ceil(1793 / 256) * math.ceil(1793 / 128)
assert (blocks_1792, blocks_1793) == (98, 120)
assert [math.ceil(g / 108) for g in (98, 120)] == [1, 2]
assert [math.ceil(g / 216) for g in (98, 120)] == [1, 1]
assert round(100 * ((1793 / 1792)**3 - 1), 2) == .17
assert Q(3, 2) / 2 == Q(3, 4)

figures = re.findall(r"!\[\[02 Areas/ML & DL/([^\]]+)\]\]", body)
assert len(figures) == 9 and all((root / p).is_file() for p in figures)
assert body.count('id="cs336-systems-roofline"') == 1

# Execute the actual new Markdown examples, not a second implementation.
fleet_block = next(block for block in blocks if 'def dense_training_mfu' in block)
fleet_ns = {}
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(compile(fleet_block, path.name + ':fleet', 'exec'), fleet_ns)
assert output.getvalue().splitlines() == [
    'MFU: 65.00% -> 48.00%',
    'Relative decrease: 26.15%',
    'Percentage-point decrease: 17.00',
]

def table_rows(header):
    rows = body.split(header + '\n', 1)[1].split('\n\n', 1)[0].splitlines()[1:]
    return [[cell.strip() for cell in row.strip('|').split('|')] for row in rows]

fleet_rows = table_rows('| Quantity | Local baseline | Larger job |')
for col in (1, 2):
    devices, tokens, ms = [Q(row[col]) for row in fleet_rows[:3]]
    work, budget = [Q(row[col]) for row in fleet_rows[3:5]]
    mfu = Q(fleet_rows[5][col].rstrip('%')) / 100
    exact_work = 6 * 70 * Q(tokens, 1) / 10**6  # 70 billion parameters -> PFLOP.
    exact_budget = devices * 989 * ms / 10**6
    assert work == exact_work and budget == exact_budget
    actual = fleet_ns['dense_training_mfu'](70e9, float(tokens), int(devices), float(ms)/1000, 989e12)
    assert math.isclose(actual, float(work / budget))
    assert abs(actual - float(mfu)) < .00005
assert Q(35446, 2204) / 16 > 1  # Nearly, not exactly, fixed work per GPU.

progress_block = next(block for block in blocks if 'productive_fraction = preserved / elapsed' in block)
progress_ns = {}
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(compile(progress_block, path.name + ':progress', 'exec'), progress_ns)
assert output.getvalue().splitlines() == [
    'Productive time: 75.00%', 'Preserved-work utilization: 37.50%'
]
progress_rows = table_rows('| State | Seconds | What remains after recovery |')
seconds = [int(row[1]) for row in progress_rows]
assert seconds == [2700, 300, 120, 180, 300, 3600]
assert sum(seconds[:-1]) == seconds[-1] == progress_ns['elapsed']
assert Q(seconds[0], seconds[-1]) == Q(3, 4)
assert Q(seconds[0] + seconds[1], seconds[-1]) == Q(5, 6)
assert Q(1, 2) * Q(seconds[0], seconds[-1]) == Q(3, 8)

# Exact arithmetic for the numerator/denominator and aggregation examples.
assert (Q(600, 1000), Q(800, 1000)) == (Q(3, 5), Q(4, 5))
assert (Q(80, 100) * 100 + Q(20, 100) * 400) / (100 + 400) == Q(8, 25)
assert (Q(8, 10) * 10 + Q(2, 10) * 90) / 100 == Q(13, 50)
for n0 in (1, 8, 32):
    for factor in (1, 2, 4, 16):
        for t0 in (Q(1, 10), Q(1), Q(3, 2)):
            for tn in (Q(1, 20), Q(1, 5), Q(1)):
                n = n0 * factor
                local_mfu = Q(600, 1) / (t0 * n0 * 1000)
                fleet_mfu = Q(600, 1) / (tn * n * 1000)
                assert fleet_mfu / local_mfu == (t0 / tn) / Q(n, n0)
print('PASS actual fleet/progress CPU blocks, both chapter tables, weighted aggregates, '
      'and 108 exact strong-scaling identity cases; no cluster benchmark')
print("PASS actual CPU cost block, 25 rational cross-checks, all worked numbers; "
      "6 Python blocks parsed, 3 CPU blocks executed, 3 CUDA blocks not executed")
print("chapter_sha256=" + hashlib.sha256(path.read_bytes()).hexdigest())
