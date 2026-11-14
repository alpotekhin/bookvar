#!/usr/bin/env python3
"""CPU gradient equivalence, including pinned upstream RNG replay. No benchmark."""
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning.md'
FIXTURES = ROOT / 'publishing/fixtures/gradcache-906f0383'


def objective(q, d):
    return F.cross_entropy(q @ d.T, torch.arange(len(q)))


def reference(model, q, d, chunk):
    model.zero_grad(set_to_none=True)
    # Retain all chunk graphs. This reference preserves the same RNG call order
    # as the cached path while still using the full candidate matrix.
    loss = objective(torch.cat([model(x) for x in q.split(chunk)]),
                     torch.cat([model(x) for x in d.split(chunk)]))
    loss.backward()
    return loss.detach(), [p.grad.detach().clone() for p in model.parameters()]


def check_figure_candidates(cached):
    # Figure 1(a,b): four current pairs, two encoder chunks of two pairs.
    identity = torch.eye(4, dtype=torch.float64)
    encoder = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.copy_(identity)
    full_loss, full_gradient = reference(encoder, identity, identity, 2)
    cached_loss, cached_gradient = cached(encoder, identity, identity, 2)
    torch.testing.assert_close(cached_loss, full_loss)
    torch.testing.assert_close(cached_gradient[0], full_gradient[0])
    assert math.isclose(float(full_loss), math.log1p(3 / math.e), abs_tol=1e-12)

    encoder.zero_grad(set_to_none=True)
    local_loss = torch.stack([objective(encoder(x), encoder(x)) for x in identity.split(2)]).mean()
    local_loss.backward()
    assert math.isclose(local_loss.item(), math.log1p(1 / math.e), abs_tol=1e-12)
    assert math.isclose(full_gradient[0][0, 2].item(), 0.5 / (math.e + 3), abs_tol=1e-12)
    assert torch.count_nonzero(encoder.weight.grad[:2, 2:]) == 0

    # Figure 1(c): scores still depend on both sides, but the old side is detached.
    identity = torch.eye(6, dtype=torch.float64)
    old_q, old_d = (identity[:4].clone().requires_grad_() for _ in range(2))
    new_q, new_d = (identity[4:].clone().requires_grad_() for _ in range(2))
    loss = objective(torch.cat([old_q.detach(), new_q]), torch.cat([old_d.detach(), new_d]))
    loss.backward()
    assert math.isclose(loss.item(), math.log1p(5 / math.e), abs_tol=1e-12)
    assert old_q.grad is None and old_d.grad is None
    expected = 1 / (6 * (math.e + 5))
    assert math.isclose(new_q.grad[0, 0].item(), expected, abs_tol=1e-12)
    assert math.isclose(new_d.grad[0, 0].item(), expected, abs_tol=1e-12)


def main():
    torch.set_num_threads(1)
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    block, = [b for b in blocks if 'def cached_encoder_grad' in b]
    namespace = {}
    exec(compile(block, str(CHAPTER) + ':GradCache', 'exec'), namespace)
    cached = namespace['cached_encoder_grad']
    check_figure_candidates(cached)
    package = types.ModuleType('grad_cache')
    package.__path__ = [str(FIXTURES)]
    sys.modules['grad_cache'] = package
    GradCache = importlib.import_module('grad_cache.grad_cache').GradCache
    rng = torch.Generator().manual_seed(20261106)
    counts = {'chapter_example': 1, 'figure_candidate_cases': 3, 'deterministic_gradient_pairs': 0,
              'upstream_dropout_replay_pairs': 0, 'missing_replay_negative_controls': 0}
    for batch in [2, 5]:
        q = torch.randn(batch, 3, generator=rng, dtype=torch.float64)
        d = torch.randn(batch + 1, 3, generator=rng, dtype=torch.float64)
        for chunk in [1, 2, 3, 7]:
            for dropout in [0., .35]:
                torch.manual_seed(123)
                model = nn.Sequential(nn.Linear(3, 4, dtype=torch.float64), nn.Tanh(),
                                      nn.Dropout(dropout), nn.Linear(4, 3, dtype=torch.float64))
                expected_model, actual_model = deepcopy(model), deepcopy(model)
                torch.manual_seed(456)
                expected_loss, expected_grad = reference(expected_model, q, d, chunk)
                torch.manual_seed(456)
                if dropout:
                    gc = GradCache([actual_model, actual_model], chunk, objective)
                    actual_loss = gc.cache_step(q, d)
                    actual_grad = [p.grad for p in actual_model.parameters()]
                    counts['upstream_dropout_replay_pairs'] += 1
                else:
                    actual_loss, actual_grad = cached(actual_model, q, d, chunk)
                    counts['deterministic_gradient_pairs'] += 1
                assert torch.allclose(actual_loss, expected_loss, atol=1e-12, rtol=1e-12)
                for actual, expected in zip(actual_grad, expected_grad, strict=True):
                    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-11)
                if dropout:
                    torch.manual_seed(456)
                    _, wrong = cached(deepcopy(model), q, d, chunk)
                    assert any(not torch.allclose(a, e, atol=1e-9, rtol=1e-9)
                               for a, e in zip(wrong, expected_grad, strict=True))
                    counts['missing_replay_negative_controls'] += 1
    source = json.loads((FIXTURES / 'sources.json').read_text())
    for row in source['files']:
        data = (FIXTURES / row['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    print(json.dumps({'chapter_sha256': hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      'checks': counts, 'upstream_commit': source['commit'],
                      'scope': 'CPU derivatives, not retrieval quality, GPU memory or throughput'}, indent=2))


if __name__ == '__main__':
    main()
