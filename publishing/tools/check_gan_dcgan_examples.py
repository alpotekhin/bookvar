"""Execute the actual reader code on CPU and check source bodies/gradients.

No dataset download, GPU benchmark, plot regeneration, or image-quality claim.
The Gaussian experiment does train for its declared 20 tiny CPU epochs.
"""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор.md'
text = CHAPTER.read_text()
blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
assert len(blocks) == 14
scope = {}
output = StringIO()
with redirect_stdout(output):
    for i, block in enumerate(blocks):
        exec(compile(block, f'{CHAPTER}:program-{i+1}', 'exec'), scope)

packet = ROOT / '05 Источники/Originals/gan-dcgan-2026-10-28/d2l'
def pytorch_blocks(name):
    raw = (packet / name).read_text()
    code = re.findall(r'```\{\.python \.input\}\n(.*?)\n```', raw, re.S)
    return [re.sub(r'^#@(?:tab|save)[^\n]*\n', '', b, flags=re.M)
            for b in code if re.match(r'#@tab [^\n]*(?:pytorch|all)', b)]

gan, dcgan = pytorch_blocks('gan.md'), pytorch_blocks('dcgan.md')
original_bodies = [gan[i] for i in [4, 5, 6, 7]] + [dcgan[i] for i in [4, 7, 10, 12]]
for body in original_bodies:
    assert '\n```python\n' + body.strip() + '\n```' in text

# Independent finite differences use the same deterministic scalar losses.
gradient_cases = 0
for value in [-12., -6., -1., 0., 1., 6., 12.]:
    a = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    for fun, expected in [(lambda x: -F.softplus(x), -a.sigmoid()),
                          (lambda x: F.softplus(-x), a.sigmoid()-1)]:
        derivative, = torch.autograd.grad(fun(a), a)
        step = 1e-5
        numeric = (fun(a.detach()+step)-fun(a.detach()-step))/(2*step)
        torch.testing.assert_close(derivative, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(derivative, numeric, atol=1e-9, rtol=1e-7)
        gradient_cases += 1

# Pointwise discriminator optimum against a dense independent candidate grid.
optimal_cases = 0
grid = torch.linspace(.0001, .9999, 9999, dtype=torch.float64)
for p_mass in [.01, .2, .5, .8, .99]:
    for g_mass in [.01, .2, .5, .8, .99]:
        optimum = p_mass / (p_mass + g_mass)
        value = p_mass * torch.tensor(optimum, dtype=torch.float64).log()
        value += g_mass * torch.tensor(1-optimum, dtype=torch.float64).log()
        sampled = p_mass*grid.log() + g_mass*torch.log1p(-grid)
        assert value.item() >= sampled.max().item() - 1e-7
        optimal_cases += 1

assets = json.loads((ROOT / '00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/assets.json').read_text())
assert len(assets) == 4
for row in assets:
    data = (ROOT / row['asset']).read_bytes()
    assert data == (ROOT / row['source_asset']).read_bytes()
    assert sha256(data).hexdigest() == row['sha256']
    assert sha256((ROOT / row['source_page_snapshot']).read_bytes()).hexdigest() == row['source_page_sha256']
    assert sha256((ROOT / row['license_evidence']).read_bytes()).hexdigest() == row['license_evidence_sha256']

print(output.getvalue())
print(json.dumps({'chapter_sha256': sha256(CHAPTER.read_bytes()).hexdigest(),
                  'actual_programs': len(blocks), 'unchanged_source_bodies': len(original_bodies),
                  'finite_difference_cases': gradient_cases,
                  'optimal_discriminator_cases': optimal_cases,
                  'gaussian_epochs': len(scope['gaussian_history']),
                  'gaussian_mean': scope['generated'].mean(0).tolist(),
                  'gaussian_covariance': torch.cov(scope['generated'].T).tolist(),
                  'new_unchanged_original_figures': len(assets)}, indent=2))
