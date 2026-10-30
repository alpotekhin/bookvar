"""Execute the chapter's three memory examples and the pinned CPU projector.

This checks algebra, state shapes and source preservation, not a GPU memory
benchmark, GaLoRE convergence, bitsandbytes or the full Adafactor recipe.
"""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import importlib.util
import math
import re

import torch
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md'
SOURCE = ROOT / '05 Источники/Originals/galore-2026-10-28'
text = CHAPTER.read_text()
text = before_topic_consolidation(
    text, CHAPTER, stop_before_journal='2026-10-29-retention-additions.json')
pattern = (r'<!-- BEGIN REVIEW: optimizer memory 2026-10-28 -->\n(.*?)'
           r'<!-- END REVIEW: optimizer memory 2026-10-28 -->\n\n')
addition, = re.findall(pattern, text, re.S)
prior = re.sub(pattern, '', text, flags=re.S)
prior = prior.replace('last_updated: 2026-10-28', 'last_updated: 2026-10-21', 1)
prior = prior.replace('source_language: mixed\n', '', 1)
assert hashlib.sha256(prior.encode()).hexdigest() == 'ffd865468db1e8ea07c64ec84f74a31cd065935f4b8aeb09a0f1f5b65670e43b'
blocks = re.findall(r'```python\n(.*?)```', addition, re.S)
assert len(blocks) == 3
namespace = {}
for i, code in enumerate(blocks, 1):
    with redirect_stdout(StringIO()) as output:
        exec(compile(code, f'{CHAPTER}:memory-example-{i}', 'exec'), namespace)
    print(f'actual memory fence {i}:', output.getvalue().strip() or 'PASS')

# Direct derivation: marginals preserve their total and factor exactly for rank one.
cases = 0
for m, n in ((2, 2), (3, 5), (5, 3), (1, 7)):
    generator = torch.Generator().manual_seed(m * 100 + n)
    row = torch.zeros(m, dtype=torch.float64)
    col = torch.zeros(n, dtype=torch.float64)
    full = torch.zeros((m, n), dtype=torch.float64)
    for step in range(1, 5):
        squares = torch.rand((m, n), generator=generator, dtype=torch.float64) + .1
        full = .7 * full + .3 * squares
        row = .7 * row + .3 * squares.sum(1)
        col = .7 * col + .3 * squares.sum(0)
        torch.testing.assert_close(row, full.sum(1), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(col, full.sum(0), rtol=1e-14, atol=1e-14)
        estimate = torch.outer(row, col) / row.sum()
        torch.testing.assert_close(estimate.sum(1), row)
        torch.testing.assert_close(estimate.sum(0), col)
        cases += 1
    exact = torch.outer(torch.arange(1., m + 1), torch.arange(1., n + 1))
    recovered = torch.outer(exact.sum(1), exact.sum(0)) / exact.sum()
    torch.testing.assert_close(recovered, exact)

# Execute the complete, unmodified primary projector without importing its package.
projector_path = SOURCE / 'galore_projector.py'
assert hashlib.sha256(projector_path.read_bytes()).hexdigest() == 'c820230b3d64bff01efda1d1e7b4ad909fe799a53737efdb57b4d31d6fb76d48'
spec = importlib.util.spec_from_file_location('bookvar_original_galore_projector', projector_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Projector = module.GaLoreProjector
for m, n in ((2, 2), (7, 3), (3, 7), (8, 4), (4, 8)):
    for rank in (1, min(m, n)):
        G = torch.randn((m, n), generator=torch.Generator().manual_seed(m * n + rank))
        projector = Projector(rank, update_proj_gap=2, scale=.25, proj_type='std')
        compact = projector.project(G, 0)
        reconstructed = projector.project_back(compact)
        U, s, Vh = torch.linalg.svd(G, full_matrices=False)
        best_rank = (U[:, :rank] * s[:rank]) @ Vh[:rank, :]
        torch.testing.assert_close(reconstructed, .25 * best_rank, rtol=3e-5, atol=3e-6)
        expected_shape = (m, rank) if m >= n else (rank, n)
        assert tuple(compact.shape) == expected_shape
        assert projector.ortho_matrix.numel() == min(m, n) * rank
        # A real Adam on these coordinates allocates two compact state tensors.
        proxy = torch.nn.Parameter(torch.zeros_like(compact))
        optimizer = torch.optim.Adam([proxy], lr=.01)
        proxy.grad = compact.clone()
        optimizer.step()
        moments = optimizer.state[proxy]
        assert moments['exp_avg'].numel() + moments['exp_avg_sq'].numel() == 2 * max(m, n) * rank
        assert moments['exp_avg'].dtype == compact.dtype
        cases += 1

# Refresh happens at zero-based iterations 0,2,...; step 1 must reuse the old basis.
projector = Projector(1, update_proj_gap=2, scale=1., proj_type='std')
G1, G2 = torch.diag(torch.tensor([3., 1.])), torch.diag(torch.tensor([1., 3.]))
first = projector.project_back(projector.project(G1, 0))
basis_before = projector.ortho_matrix.clone()
stale = projector.project_back(projector.project(G2, 1))
assert torch.equal(projector.ortho_matrix, basis_before)
fresh = projector.project_back(projector.project(G2, 2))
torch.testing.assert_close(first, torch.diag(torch.tensor([3., 0.])))
torch.testing.assert_close(stale, torch.diag(torch.tensor([1., 0.])))
torch.testing.assert_close(fresh, torch.diag(torch.tensor([0., 3.])))

# Rotating a full-rank Adam normalization is not equivalent to full Adam.
G = torch.tensor([[3., 1.], [1., 3.]], dtype=torch.float64)
Q = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / math.sqrt(2)
zero = torch.zeros_like(G)
rotated, _, _ = namespace['projected_adam'](G, Q, zero, zero, 1)
plain = -.1 * G / (G.abs() + 1e-8)
assert not torch.allclose(rotated, plain)
torch.testing.assert_close((G @ Q) @ Q.T, G, rtol=1e-14, atol=1e-14)

assert namespace['full_moments'] == 134217728
assert namespace['compact_moments'] + namespace['basis'] == 6291456
assert (2 * 4096**2) / 1024**2 == 32  # Ideal one-byte moments only.
assert namespace['weights_and_grad'] + namespace['full_moments'] == 256 * 1024**2
print(f'PASS: 3 actual fences; {cases} marginal/projection/state cases; actual pinned '
      'projector reuse/refresh; full-rank adaptive counterexample; byte counts. '
      f'Torch {torch.__version__}, CPU; no timing or peak GPU memory measurement.')
print('chapter SHA256:', hashlib.sha256(text.encode()).hexdigest())
