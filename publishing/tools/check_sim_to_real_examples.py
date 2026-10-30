"""Run the nine sim-to-real CPU programs and independently test their arithmetic.

Requires existing PyTorch. No network, package install, Blender, datasets,
rendering, model-quality benchmark, or filesystem writes.
"""
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
from io import StringIO
from itertools import product
import json
from math import ceil, isclose, log, log1p
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции.md'
text = CHAPTER.read_text()
programs = re.findall(r'^```python\n(.*?)^```\s*$', text, re.M | re.S)
assert len(programs) == 9
output, scope = StringIO(), {}
for index, program in enumerate(programs, 1):
    with redirect_stdout(output):
        exec(compile(program, f'{CHAPTER}:program-{index}', 'exec'), scope)

generator = torch.Generator().manual_seed(49)
semantic_cases = 0
for objects, classes in product([2, 5, 11], [2, 3, 7]):
    categories = torch.randint(classes, (objects,), generator=generator)
    instances = torch.randint(objects, (7, 9), generator=generator)
    semantic = categories[instances]
    for category in range(classes):
        expected = sum(int(categories[int(i)]) == category for i in instances.flatten())
        assert (semantic == category).sum().item() == expected
    semantic_cases += 1

# Project, then reconstruct a camera point at the same depth. The equivalent
# world-to-camera matrix also agrees with subtracting the camera centre.
projection_cases = 0
for depth, focal, shift in product([.5, 1., 5.], [50., 100., 800.], [-2., 0., 3.]):
    point = torch.tensor([.7, -.2, depth], dtype=torch.float64)
    center = torch.tensor([shift, 0., 0.], dtype=torch.float64)
    world = point + center
    extrinsic = torch.cat([torch.eye(3, dtype=torch.float64), -center[:, None]], dim=1)
    actual = extrinsic @ torch.cat([world, torch.ones(1, dtype=world.dtype)])
    torch.testing.assert_close(actual, point)
    principal = torch.tensor([320., 240.], dtype=point.dtype)
    pixel = focal*actual[:2]/actual[2] + principal
    reconstructed = torch.cat([(pixel-principal)*depth/focal, torch.tensor([depth])])
    torch.testing.assert_close(reconstructed, point)
    projection_cases += 1

material_cases = 0
for count, replace in product(range(1, 13), [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)]):
    explicit = sum(replace/count for candidate in range(count) if candidate != 0)
    assert explicit == replace*(count-1)/count
    assert 0 <= explicit <= replace
    material_cases += 1

coverage_cases = 0
for p, confidence in product([.0001, .001, .01, .2], [.5, .9, .95, .999]):
    n = ceil(log(1-confidence)/log1p(-p))
    assert (1-p)**n <= 1-confidence+1e-14
    assert (1-p)**(n-1) > 1-confidence-1e-14
    coverage_cases += 1

risk_cases = 0
for a, b, first, second in product([Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)],
                                  [Fraction(1, 5), Fraction(4, 5)],
                                  [Fraction(0), Fraction(1, 4), Fraction(1)],
                                  [Fraction(0), Fraction(1, 3), Fraction(1)]):
    ps, pt, losses = [a, 1-a], [b, 1-b], [first, second]
    weighted = sum(s*(t/s)*loss for s, t, loss in zip(ps, pt, losses))
    assert weighted == sum(t*loss for t, loss in zip(pt, losses))
    risk_cases += 1

# Test the actual chapter GRL inside a two-head graph, including lambda0 and
# unequal source/target counts. Plain domain-head gradients must not reverse.
Reverse = scope['Reverse']
gradient_cases = 0
for batch, features, strength in product([3, 5, 9], [1, 3], [0., .1, .7, 1., 2.]):
    torch.manual_seed(11)
    encoder = nn.Linear(2, features).double()
    class_head = nn.Linear(features, 2).double()
    domain_head = nn.Linear(features, 1).double()
    inputs = torch.randn(batch, 2, generator=generator, dtype=torch.float64)
    encoded = encoder(inputs)
    domains = torch.tensor([0., 0.] + [1.]*(batch-2), dtype=torch.float64)
    ly = F.cross_entropy(class_head(encoded[:2]), torch.tensor([0, 1]))
    ld = F.binary_cross_entropy_with_logits(domain_head(encoded).flatten(), domains)
    class_encoder, = torch.autograd.grad(ly, encoder.weight, retain_graph=True)
    domain_encoder, = torch.autograd.grad(ld, encoder.weight, retain_graph=True)
    plain_domain_head, = torch.autograd.grad(ld, domain_head.weight, retain_graph=True)
    plain_class_head, = torch.autograd.grad(ly, class_head.weight, retain_graph=True)
    reversed_loss = F.binary_cross_entropy_with_logits(
        domain_head(Reverse.apply(encoded, strength)).flatten(), domains)
    actual_encoder, actual_domain, actual_class = torch.autograd.grad(
        ly+reversed_loss, (encoder.weight, domain_head.weight, class_head.weight))
    torch.testing.assert_close(actual_encoder, class_encoder-strength*domain_encoder)
    torch.testing.assert_close(actual_domain, plain_domain_head)
    torch.testing.assert_close(actual_class, plain_class_head)
    gradient_cases += 1

# Evaluate the actual population functions on translated/scaled distributions.
moments, kernel_mean = scope['moments'], scope['kernel_mean']
metric_cases = 0
base_s, base_t = scope['source'], scope['target']
for shift, scale in product([-10., 0., 5.], [.1, 1., 4.]):
    left = [(shift+scale*x, p) for x, p in base_s]
    right = [(shift+scale*x, p) for x, p in base_t]
    m1, v1 = moments(left)
    m2, v2 = moments(right)
    assert isclose(m1, m2, abs_tol=1e-12) and isclose(v1, v2, abs_tol=1e-12)
    mmd = kernel_mean(left, left)+kernel_mean(right, right)-2*kernel_mean(left, right)
    assert mmd > 0
    assert isclose(kernel_mean(left, right), kernel_mean(right, left), abs_tol=1e-12)
    same = kernel_mean(left, left)+kernel_mean(left, left)-2*kernel_mean(left, left)
    assert same == 0
    metric_cases += 1

rows = json.loads((ROOT/'00 Учебник/Assets/Figures/curated/sim-to-real-2026-10-28/assets.json').read_text())
assert len(rows) == 7
for row in rows:
    for file_key, hash_key in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                               ('source_page_snapshot', 'source_page_sha256'),
                               ('license_evidence', 'license_evidence_sha256')]:
        assert sha256((ROOT/row[file_key]).read_bytes()).hexdigest() == row[hash_key]
    if row['derivation'] == 'source-jpeg-copy':
        assert (ROOT/row['asset']).read_bytes() == (ROOT/row['source_asset']).read_bytes()
    assert (ROOT/row['public_license_copy']).read_bytes() == (ROOT/row['license_evidence']).read_bytes()

print(output.getvalue(), end='')
print(f'PASS: 9 programs; {semantic_cases} ID maps, {projection_cases} projections, '
      f'{material_cases} material cases, {coverage_cases} coverage, {risk_cases} risks, '
      f'{gradient_cases} GRL graphs, {metric_cases} metric pairs; 7 source figures')
print('chapter SHA256', sha256(CHAPTER.read_bytes()).hexdigest())
