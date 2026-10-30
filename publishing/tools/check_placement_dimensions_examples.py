"""Check the exact visible 3D placement program; no processes or GPU work."""
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / '00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md'
text = path.read_text()
# Restore only the separately recorded navigation revision for the historical
# prose-preservation assertion. Keep its old full SHA as the acceptance target.
nav = json.loads((ROOT / 'docs/audit/2026-10-29-reader-route-navigation.json').read_text())
row = next((r for r in nav['files'] if r['source'] == path.relative_to(ROOT).as_posix()), None)
if row is not None:
    current = sha256(text.encode()).hexdigest()
    if current == row['after']:
        for change in reversed(row['changes']):
            assert text.count(change['new']) == 1
            text = text.replace(change['new'], change['old'], 1)
        assert sha256(text.encode()).hexdigest() == row['before']
    else:
        assert current == row['before'], 'Unreviewed distributed chapter revision'
match = re.search(r'<!-- BEGIN REVIEW: placement dimensions 2026-10-28 -->\n(.*?)<!-- END REVIEW: placement dimensions 2026-10-28 -->\n\n', text, re.S)
assert match
baseline = text[:match.start()] + text[match.end():]
baseline = baseline.replace('last_updated: 2026-10-28', 'last_updated: 2026-10-23')
baseline = baseline.replace('last_verified: 2026-10-28', 'last_verified: 2026-10-23')
assert sha256(baseline.encode()).hexdigest() == 'fe1a7b12bc09cabd25750a0f14ca805217c05f72359549d3cbd611e3906a02a9'
blocks = re.findall(r'```python\n(.*?)\n```', match.group(1), re.S)
assert len(blocks) == 1
scope = {}
exec(compile(blocks[0], str(path) + ':placement', 'exec'), scope)
placement, groups = scope['placement'], scope['placement_groups']
for coordinate, rank in placement.items():
    p, r, t = coordinate
    assert (rank // 8, rank % 8 // 4, rank % 4) == coordinate
    result = groups(*coordinate)
    for key, vary in [('tp', 2), ('dp', 1), ('pp', 0)]:
        expected = {other_rank for other, other_rank in placement.items()
                    if all(other[i] == coordinate[i] for i in range(3) if i != vary)}
        assert set(result[key]) == expected
        assert len(result[key]) == (4, 2, 4)[vary]
    for first, second in combinations(result.values(), 2):
        assert set(first) & set(second) == {rank}

# Each group family partitions the complete placement, not just rank 0.
for axis, count, width in [('tp', 8, 4), ('dp', 16, 2), ('pp', 8, 4)]:
    unique = {tuple(groups(*coordinate)[axis]) for coordinate in placement}
    assert len(unique) == count and all(len(group) == width for group in unique)
    assert sorted(rank for group in unique for rank in group) == list(range(32))
assert [placement[p, r, 0] for r in range(2) for p in range(4)] == [0, 8, 16, 24, 4, 12, 20, 28]
assert scope['microbatch'] * scope['accumulation'] * 2 == 30
assert scope['microbatch'] * scope['accumulation'] * 32 == 480

asset = ROOT / '00 Учебник/Assets/Figures/curated/frameworks-native-2026-10-27/device-placement-dimensions.svg'
original = ROOT / 'Assets/Sources/Harvard ML Systems/vol1/frameworks/2989ddaa72253cc48b5b80d50c512a87ca09ff9b.svg'
assert asset.read_bytes() == original.read_bytes()
assert sha256(asset.read_bytes()).hexdigest() == '6c3e1e350e5653c517ac8b82ec4a7184eceb7573ae8320b56c4ca222861c6b12'
svg = ET.fromstring(asset.read_bytes())
assert svg.attrib['viewBox'] == '0 0 274.96933 168.46268'
labels = {element.attrib.get('aria-label') for element in svg.iter()}
assert {'GPU0', 'GPU4', 'GPU8', 'GPU12', 'GPU16', 'GPU20', 'GPU24', 'GPU28',
        'Acrosslayergroups', 'Acrossreplicas', 'Withintensors'} <= labels
assert not any(element.tag.split('}')[-1] in {'script', 'foreignObject'} for element in svg.iter())
print('PASS exact printed placement fence; all 32 coordinate inverses and all TP/DP/PP group memberships/intersections;')
print('three complete group-family partitions, eight visible GPU labels and batch 30 versus wrong 480.')
print('PASS whole prior 917-line chapter preserved, full unchanged SVG/viewBox/labels; no process, collective, GPU or timing run.')
print('chapter_sha256=' + sha256(path.read_bytes()).hexdigest())
