"""Check the actual S8 reduction trace and original GIF; no CUDA execution."""
import hashlib
import math
from pathlib import Path
import random
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md'
text = chapter.read_text()
section = text.split('<!-- BEGIN REVIEW: local reduction tree 2026-10-27 -->')[1].split(
    '<!-- END REVIEW: local reduction tree 2026-10-27 -->')[0]
blocks = re.findall(r'```python\n(.*?)\n```', section, re.S)
assert len(blocks) == 1
scope = {}
exec(compile(blocks[0], str(chapter) + ':reduction-trace', 'exec'), scope)
trace = scope['reduction_trace']
rng = random.Random(27)
for n in range(1, 1026):
    values = [rng.randrange(-100, 101) for _ in range(n)]
    levels, additions = trace(values)
    assert levels[0] == values and levels[-1] == [sum(values)]
    assert additions == n - 1
    assert len(levels) - 1 == (n - 1).bit_length()
    assert all(len(b) == (len(a) + 1) // 2 for a, b in zip(levels, levels[1:]))
    assert all(sum(level) == sum(values) for level in levels)
try:
    trace([])
except ValueError:
    pass
else:
    raise AssertionError('empty input contract was not enforced')

# Each stage is an ideal dependency layer, not a CUDA duration.
stage_work = [4, 2, 1]
assert [sum(math.ceil(work / p) for work in stage_work) for p in (1, 2, 4)] == [7, 4, 3]
assert trace([3, 1, 4, 1, 5])[1] == 4
assert trace([3, 1, 4, 1, 5, 0, 0, 0])[1] == 7
assert 2 + 3 + (5 + 1) == 11 and (2 + 3) + 5 == 10
assert trace([13, 13, 6, 7]) == ([[13, 13, 6, 7], [26, 13], [39]], 3)

# Model only the full-participation register exchange in the copied helper.
lanes = list(range(1, 33))
lane_additions = 0
for offset in (16, 8, 4, 2, 1):
    before = lanes.copy()
    lanes = [value + before[i + offset if i + offset < 32 else i]
             for i, value in enumerate(before)]
    lane_additions += len(lanes)
assert lanes[0] == 528 and lanes[-1] == 1024
assert lane_additions == 160 and trace(list(range(1, 33)))[1] == 31
assert not all(value == 528 for value in lanes)

directory = ROOT / '05 Источники/Originals/reduction-primitives-2026-10-27'
gif = directory / 'binomial-tree.gif'
assert hashlib.sha256(gif.read_bytes()).hexdigest() == 'ca079d087b8e9a7d4f61cf91d3d64323eb1ebeb5c8992b4c0d8c3f790de1ee8a'
with Image.open(gif) as image:
    assert image.size == (901, 512) and image.n_frames == 4
    for frame in range(4):
        image.seek(frame)
        image.convert('RGBA').load()
        assert image.info['duration'] == 1500
assert gif.read_bytes() == (ROOT / '00 Учебник/Assets/Figures/curated/reduction-primitives-2026-10-27/binomial-tree.gif').read_bytes()
print('PASS actual reduction_trace fence; 1025 signed-integer arrays, every partial sum, odd carry, work/depth, empty-input rejection')
print('PASS ideal 1/2/4-worker schedules, padding cost, stale-read counterexample, four-lane accumulator example')
print('PASS full-warp CPU exchange model and all four original GIF frames; no CUDA/Triton compilation, execution or timing')
print('chapter_sha256=' + hashlib.sha256(chapter.read_bytes()).hexdigest())
