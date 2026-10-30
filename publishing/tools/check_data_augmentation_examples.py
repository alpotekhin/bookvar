"""Run the reader's nine CPU programs and independently check their mechanisms.

Requires the existing local PyTorch environment; downloads nothing. Source
excerpts are archival and deliberately excluded from executable examples.
No recognition benchmark, training-dataset download or filesystem writes.
"""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from itertools import product
import json
from math import isclose
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований.md"
text = CHAPTER.read_text()
programs = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
assert len(programs) == 9
scope, output = {}, StringIO()
for index, program in enumerate(programs, 1):
    with redirect_stdout(output):
        exec(compile(program, f"{CHAPTER}:program-{index}", "exec"), scope)

# Exhaustive half-open boxes in two nonsquare images. Reflection must agree
# with a pixel mask, including empty boxes and boxes touching either boundary.
box_cases = 0
for height, width in [(3, 5), (4, 6)]:
    for y1, y2, x1, x2 in product(range(height + 1), range(height + 1),
                                  range(width + 1), range(width + 1)):
        if y1 > y2 or x1 > x2:
            continue
        mask = torch.zeros(height, width, dtype=torch.bool)
        mask[y1:y2, x1:x2] = True
        expected = torch.zeros_like(mask)
        expected[y1:y2, width-x2:width-x1] = True
        assert torch.equal(expected, mask.flip(-1))
        box_cases += 1

# The mean squared loss gradient uses every view, not just the mean prediction.
objective_cases = 0
for a, b, target in product([0., 1., 3.], [-2., 2.], [0., 2.]):
    views = torch.tensor([[a, b], [b, a]], dtype=torch.float64)
    w = torch.tensor([.25, -.5], dtype=torch.float64, requires_grad=True)
    predictions = views @ w
    loss = (predictions-target).square().mean()
    gradient, = torch.autograd.grad(loss, w)
    explicit = sum(2 * (float(w.detach() @ v)-target) * v for v in views)/2
    torch.testing.assert_close(gradient, explicit)
    gap = loss.detach() - (predictions.mean().detach()-target).square()
    torch.testing.assert_close(gap, predictions.detach().var(unbiased=False))
    objective_cases += 1

# Soft-target CE equals the weighted hard-label losses for the SAME logits.
mixup_cases = 0
generator = torch.Generator().manual_seed(29)
for classes, lam in product([2, 3, 7], [0., .1, .25, .5, .9, 1.]):
    for a, b in product(range(classes), repeat=2):
        z = torch.randn(classes, generator=generator, dtype=torch.float64, requires_grad=True)
        q = lam*F.one_hot(torch.tensor(a), classes)+(1-lam)*F.one_hot(torch.tensor(b), classes)
        loss = -(q*z.log_softmax(-1)).sum()
        weighted = lam*F.cross_entropy(z[None], torch.tensor([a]))+(1-lam)*F.cross_entropy(z[None], torch.tensor([b]))
        torch.testing.assert_close(loss, weighted)
        gradient, = torch.autograd.grad(loss, z)
        torch.testing.assert_close(gradient, z.softmax(-1)-q)
        mixup_cases += 1

# Count written pixels independently of the helper's lambda arithmetic.
# Non-square, odd/even dimensions, edge/corner/centre, lambda 0 and 1, and
# both swapped and self pairings are checked. The parents must not mutate.
cutmix_cases = 0
cutmix = scope["cutmix_batch"]
for height, width, nominal in product([3, 4, 5], [2, 6, 7], [0., .1, .5, .9, 1.]):
    images = torch.stack([torch.zeros(2, height, width), torch.ones(2, height, width)]).double()
    parents = images.clone()
    labels = torch.tensor([0, 1])
    for cy, cx in product([0, height//2, height-1], [0, width//2, width-1]):
        mixed, targets, actual, box = cutmix(images, labels, torch.tensor([1, 0]), nominal, (cy, cx), 2)
        x1, y1, x2, y2 = box
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height
        written = (mixed[0, 0] == 1).sum().item()
        assert written == (x2-x1)*(y2-y1)
        assert isclose(actual, 1-written/(height*width))
        torch.testing.assert_close(targets[0], torch.tensor([actual, 1-actual], dtype=images.dtype))
        torch.testing.assert_close(targets.sum(-1), torch.ones(2, dtype=images.dtype))
        assert torch.equal(mixed[1], 1-mixed[0])
        assert torch.equal(images, parents)
        same, same_targets, _, _ = cutmix(images, labels, torch.arange(2), nominal, (cy, cx), 2)
        assert torch.equal(same, images)
        torch.testing.assert_close(same_targets, torch.eye(2, dtype=images.dtype))
        cutmix_cases += 1

# Enumerate the sub-policy Bernoulli decisions, including their mixture weight.
policy_probability = 0.
mass = 0.
for p, q in [(.8, .5), (.3, .1)]:
    for first, second in product([False, True], repeat=2):
        probability = .5*(p if first else 1-p)*(q if second else 1-q)
        mass += probability
        if not first and not second:
            policy_probability += probability
assert isclose(mass, 1.) and isclose(policy_probability, .365)
assert len(scope["possible_sequences"]) == 196

parents_train = {r[:2] for r in scope["train"]}
parents_test = {r[:2] for r in scope["test"]}
assert parents_train.isdisjoint(parents_test)
assert len(parents_train) == 400 and len(parents_test) == 100

# Softmax(mean logits) is normalized geometric averaging of probabilities.
tta_cases = 0
for views, classes in product([1, 2, 5], [2, 3, 7]):
    z = torch.randn(views, classes, generator=generator, dtype=torch.float64)
    p = z.softmax(-1)
    actual = z.mean(0).softmax(-1)
    geometric = p.prod(0).pow(1/views)
    torch.testing.assert_close(actual, geometric/geometric.sum())
    tta_cases += 1

rows = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/assets.json").read_text())
assert len(rows) == 2
for row in rows:
    data = (ROOT/row["asset"]).read_bytes()
    assert data == (ROOT/row["source_asset"]).read_bytes()
    assert sha256(data).hexdigest() == row["sha256"] == row["source_asset_sha256"]
    for file_key, hash_key in [("source_page_snapshot", "source_page_sha256"),
                                ("license_evidence", "license_evidence_sha256")]:
        assert sha256((ROOT/row[file_key]).read_bytes()).hexdigest() == row[hash_key]
    assert (ROOT/row["public_license_copy"]).read_bytes() == (ROOT/row["license_evidence"]).read_bytes()
    if row["asset"].endswith(".png"):
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert list(struct.unpack(">II", data[16:24])) == [649, 447]
        position, compressed = 8, bytearray()
        while position < len(data):
            length = struct.unpack(">I", data[position:position+4])[0]
            kind = data[position+4:position+8]
            payload = data[position+8:position+8+length]
            crc = struct.unpack(">I", data[position+8+length:position+12+length])[0]
            assert zlib.crc32(kind+payload) & 0xffffffff == crc
            if kind == b"IDAT":
                compressed.extend(payload)
            position += length+12
            if kind == b"IEND":
                break
        assert position == len(data) and zlib.decompress(compressed)
    else:
        svg = ET.fromstring(data)
        assert [float(v) for v in svg.attrib["viewBox"].split()] == [0, 0, 465.600016, 237.203471]
        assert len(svg.findall(".//{http://www.w3.org/2000/svg}image")) == 8
        assert not re.search(rb'(?:href|src)="https?://', data)

print(output.getvalue(), end="")
print(f"PASS: 9 programs, {box_cases} boxes, {objective_cases} objectives, "
      f"{mixup_cases} Mixup, {cutmix_cases} CutMix, 8 policy outcomes, {tta_cases} TTA cases")
print("chapter SHA256", sha256(CHAPTER.read_bytes()).hexdigest())
