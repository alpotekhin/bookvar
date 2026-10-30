"""Run every actual 55a CPU fence and independently verify MBU examples.

No profiler, GPU, or inference benchmark is run by this arithmetic check.
"""
import ast
import contextlib
from fractions import Fraction
from hashlib import sha256
import io
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / '00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md'
text = path.read_text()
blocks = re.findall(r'<!-- example: ([^>]+) -->\n```python\n(.*?)\n```', text, re.S)
assert len(blocks) == 5
namespaces, outputs = {}, {}
for name, code in blocks:
    ns = {}
    ast.parse(code)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(code, f'{path.name}:{name}', 'exec'), ns)
    namespaces[name], outputs[name] = ns, out.getvalue().splitlines()

assert outputs['inference-mbu-batch'] == ['1 71.43 50.00%', '4 200.00 35.00%']
assert outputs['inference-mbu-moe'] == ['(0, 0) 3 GiB', '(0, 1) 4 GiB']
assert outputs['inference-mbu-ranks'] == ['12 ms 58.33%']

# Independent rational arithmetic: no binary-float fixture copied from a printout.
W = 2 * 10**12
for B, t, expected in [(1, Fraction(14, 1000), Fraction(1, 2)),
                        (4, Fraction(20, 1000), Fraction(7, 20))]:
    R = B / t
    assert Fraction(14 * 10**9, 1) / (t * W) == expected
    assert R * 14 * 10**9 / (B * W) == expected
    if B == 4:
        assert R * 14 * 10**9 / W == Fraction(7, 5)  # deliberately wrong accounting
assert Fraction(27 * 10**9, 1) / (Fraction(14, 1000) * W) == Fraction(27, 28)

# All 16 top-1 routing pairs and their permutation: distinct expert union, shared once.
ns = namespaces['inference-mbu-moe']
for a in range(4):
    for b in range(4):
        used = ns['shared_bytes'] + sum(ns['expert_bytes'][i] for i in {a, b})
        assert used == (3 if a == b else 4) * 2**30

rank = namespaces['inference-mbu-ranks']
for m, peak, expected in zip(rank['rank_bytes'], rank['rank_peaks'], [Fraction(1), Fraction(1, 6)]):
    assert Fraction(m, 1) / (Fraction(12, 1000) * peak) == expected
assert Fraction(sum(rank['rank_bytes']), 1) / (Fraction(12, 1000) * sum(rank['rank_peaks'])) == Fraction(7, 12)

# Previous code is still executable; MHA/GQA fixtures remain arithmetically intact.
budget = namespaces['inference-llama-budget']['decode_budget']
assert budget(1)[0] == 26_869_760_000
assert round(budget(64, kv_heads=8)[0] / 1e9, 2) == 33.41
assert round(budget(256, kv_heads=8)[1] / 1e9, 2) == 65.30
for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', text):
    target = match.group(1).split('#')[0].removeprefix('02 Areas/ML & DL/')
    if target:
        assert any(candidate.exists() for candidate in (
            ROOT / target, ROOT / (target + '.md'), path.parent / (target + '.md')
        )), target
for row in ['| 1 | 14 | 71,43 | 50,00% |', '| 4 | 20 | 200,00 | 35,00% |',
            '| 0, 0 | {0} | 3 GiB |', '| 0, 1 | {0, 1} | 4 GiB |']:
    assert row in text, row
print('PASS all five actual chapter CPU fences; fixed-batch identity, 16 MoE routes, '
      'physical/model traffic distinction and uneven-rank bound; no GPU benchmark')
print('chapter_sha256=' + sha256(path.read_bytes()).hexdigest())
