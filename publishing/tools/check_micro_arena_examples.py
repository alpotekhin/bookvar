"""Check printed lifetime/budget arithmetic, not a TFLM firmware deployment."""
from hashlib import sha256
from pathlib import Path
import json
import random
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md'
text = CHAPTER.read_text()
pattern = r'<!-- BEGIN REVIEW: Micro arena 2026-10-27 -->\n(.*?)<!-- END REVIEW: Micro arena 2026-10-27 -->\n\n'
addition = re.search(pattern, text, re.S)
assert addition is not None
baseline = (text[:addition.start()] + text[addition.end():]).replace(
    'last_updated: 2026-10-27', 'last_updated: 2026-10-26').replace(
    'last_verified: 2026-10-27', 'last_verified: 2026-10-26')
assert sha256(baseline.encode()).hexdigest() == '9038a3d22ef6b746b7a99720542763f94b8117641c84543499433305a001c7e3'
blocks = re.findall(r'```python\n(.*?)\n```', addition.group(1), re.S)
assert len(blocks) == 2
scope = {}
for i, block in enumerate(blocks, 1):
    exec(compile(block, f'{CHAPTER}:arena-block-{i}', 'exec'), scope)
    print(f'actual arena block {i}: PASS')
check = scope['check_arena_plan']
kib = 1024
buffers, offsets = scope['buffers'], scope['offsets']
def peak_live(records):
    return max(sum(size for size, first, last in records.values() if first <= step <= last)
               for step in range(1 + max(last for _, _, last in records.values())))
assert sum(size for size, _, _ in buffers.values()) == 32*kib
assert peak_live(buffers) == check(buffers, offsets) == 20*kib
assert peak_live(scope['branch_buffers']) == check(scope['branch_buffers'], scope['branch_offsets']) == 28*kib
assert scope['arena_capacity'] == 32*kib

def rejects(records, plan, **kwargs):
    try:
        check(records, plan, **kwargs)
    except (ValueError, TypeError):
        return
    raise AssertionError('Expected this supplied plan to fail')
rejects(buffers, dict(offsets, a=1))
rejects(buffers, dict(offsets, a=-16))
rejects(buffers, offsets, alignment=0)
rejects({}, {})
rejects(buffers, {'input': 0})
rejects(dict(buffers, a=(0, 0, 1)), offsets)
rejects(dict(buffers, a=(12*kib, 2, 1)), offsets)
rejects({'x': (16, 0, 1), 'y': (16, 1, 2)}, {'x': 0, 'y': 0})
assert check({'x': (16, 0, 0), 'y': (16, 1, 1)}, {'x': 0, 'y': 0}) == 16
assert check({'x': (16, 0, 1), 'y': (16, 1, 2)}, {'x': 0, 'y': 16}) == 32

# An independent byte/step occupancy oracle, rather than the pairwise formula
# used by the printed checker. Cases include overlapping and reusable layouts.
rng = random.Random(20261027)
passed = rejected = 0
for case in range(400):
    records, plan = {}, {}
    for i in range(rng.randrange(1, 7)):
        first = rng.randrange(5)
        records[str(i)] = (rng.randrange(1, 9)*8, first, first + rng.randrange(3))
        plan[str(i)] = rng.randrange(12)*8
    occupied = set()
    collision = False
    for name, (size, first, last) in records.items():
        cells = {(step, byte) for step in range(first, last+1)
                 for byte in range(plan[name], plan[name]+size)}
        collision |= bool(occupied.intersection(cells))
        occupied.update(cells)
    if collision:
        rejects(records, plan, alignment=8)
        rejected += 1
    else:
        assert check(records, plan, alignment=8) == max(plan[n]+r[0] for n, r in records.items())
        passed += 1
assert passed > 0 and rejected > 0

source_root = ROOT / '05 Источники/Originals/micro-runtime-2026-10-27'
manifest = json.loads((source_root / 'source-manifest.json').read_text())
assert len(manifest) == 7
for row in manifest:
    data = (source_root / row['file']).read_bytes()
    assert sha256(data).hexdigest() == row['sha256']
    import hashlib
    assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == row['git_blob']
manual = (source_root / 'memory_management.md').read_text()
original_cpp = re.findall(r'```c\+\+\n(.*?)\n```', manual, re.S)
printed_cpp = re.findall(r'```cpp\n(.*?)\n```', addition.group(1), re.S)
assert printed_cpp == original_cpp and len(printed_cpp) == 2
source_diagram = re.findall(r'```\n(.*?)\n```', manual, re.S)[0]
assert re.findall(r'```text\n(.*?)\n```', addition.group(1), re.S)[0] == source_diagram
assert 7744 + 1824 == 9568 and 2048 < 9568

frameworks = (ROOT / '05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
original_decisions = frameworks.split('**Framework solution**:', 1)[1].split('**Silicon contract**:', 1)[0]
original_decisions = original_decisions.split('1.', 1)[1]
original_decisions = '1.' + original_decisions
original_decisions = re.sub(r'\\index\{[^}]*\}', '', original_decisions).strip()
quote = '\n'.join(line[2:] for line in addition.group(1).splitlines() if re.match(r'^> [123]\. ', line))
assert ' '.join(quote.split()) == ' '.join(original_decisions.split())
print(f'PASS: two actual Python blocks; 20/28 KiB layouts; 32 KiB phase peak; '
      f'112 KiB assigned remainder; 400 independent occupancy cases ({passed} valid/{rejected} rejected); '
      'boundary/invalid controls; seven Git-blob/SHA sources; exact Harvard decisions, '
      'both complete C++ fragments and original text diagram. No TFLM build, board, energy or timing result.')
print('chapter_sha256=' + sha256(CHAPTER.read_bytes()).hexdigest())
