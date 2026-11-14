"""Validate actual observation-format examples; do not run a timing benchmark."""
import ast
import hashlib
import inspect
import itertools
from pathlib import Path
import re
import runpy
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'publishing/tools'))
CHAPTER = ROOT / '00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md'
body = CHAPTER.read_text()
blocks = re.findall(r'^(~~~|```)python\n([\s\S]*?)^\1\s*$', body, re.M)
assert len(blocks) == 7
for i, (_, block) in enumerate(blocks):
    ast.parse(block, filename=f'{CHAPTER}:program-{i+1}')
# All previous CPU arithmetic/table cases remain enforced on exact restored bytes.
runpy.run_path(str(ROOT / 'publishing/tools/check_roofline_examples.py'))
block = next(s for _, s in blocks if 'def group_measurement(' in s)
ns = {}
exec(compile(block, str(CHAPTER)+':measurement-tail', 'exec'), ns)
make, quantile = ns['group_measurement'], ns['nearest_rank']
assert ns['uniform_measurement'].number_per_run == 100
assert ns['uniform_measurement'].num_threads == 1
assert ns['uniform_measurement'].median == .01
assert ns['bursty_measurement'].median == .01

# Check actual runtime property behavior against the pinned implementation AST,
# independently of the hand-calculated demonstration.
from torch.utils.benchmark import Measurement
source = ROOT / '05 Источники/Originals/roofline-timing-2026-11-04/common.py'
if source.exists():
    raw = source.read_bytes()
else:
    from urllib.request import urlopen
    with urlopen('https://raw.githubusercontent.com/pytorch/pytorch/'
                 '5c4886908584029761b579af026dcfb627c84070/'
                 'torch/utils/benchmark/utils/common.py', timeout=30) as response:
        raw = response.read()
assert hashlib.sha256(raw).hexdigest() == '61a4652909e5aa046ae7bdf10a91c9862fe094eaf162cb4db5af330c4cda1357'
cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'Measurement')
original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'times')
runtime = ast.parse(textwrap.dedent(inspect.getsource(Measurement.times.fget))).body[0]
assert [ast.dump(n) for n in runtime.body] == [ast.dump(n) for n in original.body]

# Different durations and group counts; same amortized observations.
checked = 0
for size, repeats in itertools.product([2, 5, 10, 50, 100], [1, 2, 10, 100]):
    slow_count = max(1, size // 50)
    slow = (10 * size - (size - slow_count)) // slow_count
    assumed = [[1] * (size - slow_count) + [slow] * slow_count for _ in range(repeats)]
    flat = [t for g in assumed for t in g]
    m = make(assumed)
    assert m.number_per_run == size and len(m.raw_times) == repeats
    assert m.times == [.01] * repeats and m.iqr == 0
    assert m.meets_confidence(.1)
    assert quantile(flat, .99) == slow > 10
    # CDF/order-statistic condition checked without calling the example helper.
    for p in [.5, .9, .99, 1.]:
        value = quantile(flat, p)
        rank = __import__('math').ceil(p * len(flat))
        assert sum(x < value for x in flat) < rank <= sum(x <= value for x in flat)
    for n, raw_time in enumerate(m.raw_times):
        assert abs(raw_time / m.number_per_run - m.times[n]) < 1e-12
    checked += 1
for args in [([], .99), ([1], 0), ([1], 1.1)]:
    try:
        quantile(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(args)
for groups in [[], [[]], [[1], [1, 2]]]:
    try:
        make(groups)
    except ValueError:
        pass
    else:
        raise AssertionError(groups)
assert (1000 * 2 + 12) / 1000 == 2.012
print(f'PASS actual Measurement example,{checked} grouped-tail cases,source property AST,invalid-input controls')
print('7 Python blocks parsed,4 CPU blocks executed,3 CUDA blocks not executed; no timing experiment')
print('chapter_sha256='+hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

