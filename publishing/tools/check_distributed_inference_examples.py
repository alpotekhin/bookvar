#!/usr/bin/env python3
"""Execute chapter 58a examples and compare CPU algebra with dense references.

No GPU, process group, NCCL, serving run or performance benchmark is executed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md'
SOURCES = ROOT / '05 Источники/Originals/distributed-inference-2026-10-20'


def dense_attention(q, k, v):
    result = np.empty_like(q)
    for b in range(len(q)):
        for h in range(q.shape[1]):
            logits = k[b] @ q[b, h] / np.sqrt(q.shape[-1])
            weights = np.exp(logits - logits.max())
            result[b, h] = weights @ v[b] / weights.sum()
    return result


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 8, len(blocks)
    namespace = {'__name__': '__chapter_examples__'}
    for i, block in enumerate(blocks):
        exec(compile(block, f'{CHAPTER}:python-block-{i+1}', 'exec'), namespace)
    rng = np.random.default_rng(20261020)
    mlp = namespace['tp_mlp']
    norm = namespace['layer_norm']
    merge = namespace['merge_attention']
    dispatch = namespace['expert_dispatch']
    pipeline = namespace['pipeline_finish']
    attention = namespace['local_attention']
    cases = dict(tp=0, swiglu=0, sp=0, softmax_merge=0, ep=0, pipeline=0, kv_permutation=0)

    for _ in range(200):
        tokens, d, f = 8, 6, 16
        x, up = rng.normal(size=(tokens, d)), rng.normal(size=(d, f))
        b, down, out_bias = rng.normal(size=f), rng.normal(size=(f, d)), rng.normal(size=d)
        gate = rng.normal(size=(d, f))
        dense = np.maximum(x @ up + b, 0) @ down + out_bias
        for degree in [1, 2, 4, 8]:
            _, partials, actual = mlp(x, up, b, down, out_bias, degree)
            np.testing.assert_allclose(actual, dense, atol=1e-12)
            cases['tp'] += 1
            full = norm(x)
            gathered = np.concatenate([norm(a) for a in np.split(x, degree)])
            _, parts, _ = mlp(gathered, up, b, down, out_bias, degree)
            local = np.split(np.sum(parts, axis=0), degree)
            np.testing.assert_allclose(np.concatenate([a + out_bias for a in local]),
                                       np.maximum(full @ up + b, 0) @ down + out_bias, atol=1e-12)
            cases['sp'] += 1
            def silu(value):
                return value / (1 + np.exp(-value))
            expected = (silu(x @ gate) * (x @ up)) @ down
            contributions = [
                (silu(x @ g) * (x @ u)) @ w
                for g, u, w in zip(np.split(gate, degree, axis=1),
                                   np.split(up, degree, axis=1), np.split(down, degree))
            ]
            np.testing.assert_allclose(np.sum(contributions, axis=0), expected, atol=1e-12)
            cases['swiglu'] += 1

    # Negative controls: nonlinear reduction order and output-bias duplication.
    assert max(-2 + 3, 0) != max(-2, 0) + max(3, 0)
    _, parts, y = mlp(x, up, b, down, out_bias, 2)
    assert not np.allclose(sum(a + out_bias for a in parts), y)

    for _ in range(500):
        length, width, shards = int(rng.integers(1, 25)), 5, int(rng.integers(1, 9))
        scores = rng.normal(size=length) * 40 + rng.choice([-1000., 0., 1000.])
        scores[rng.random(length) < .3] = -np.inf
        scores[0] = 1000.  # At least one unmasked key.
        values = rng.normal(size=(length, width))
        weights = np.exp(scores - scores.max())
        reference = weights @ values / weights.sum()
        actual = merge(np.array_split(scores, shards), np.array_split(values, shards))
        np.testing.assert_allclose(actual, reference, atol=1e-12)
        cases['softmax_merge'] += 1
    np.testing.assert_allclose(merge([np.array([-np.inf]), np.array([0.])],
                                    [np.array([[999.]]), np.array([[7.]])]), [7.])
    try:
        merge([np.array([-np.inf])], [np.array([[1.]])])
    except ValueError:
        pass
    else:
        raise AssertionError('fully masked query must be handled explicitly')

    for _ in range(300):
        tokens, experts, ranks, width = 13, 8, 4, 5
        x = rng.normal(size=(tokens, width))
        origins = rng.integers(ranks, size=tokens)
        owners = rng.permutation(np.repeat(np.arange(ranks), experts // ranks))
        choices = np.array([rng.choice(experts, size=3, replace=False) for _ in x])
        gates = rng.random(size=choices.shape)
        gates /= gates.sum(axis=1, keepdims=True)
        matrices = rng.normal(size=(experts, width, width))
        result, counts = dispatch(x, origins, choices, gates, owners, matrices)
        reference = np.empty_like(x)
        expected_counts = np.zeros((ranks, ranks), dtype=int)
        for t in range(tokens):
            combined_matrix = sum(g * matrices[e] for e, g in zip(choices[t], gates[t]))
            reference[t] = x[t] @ combined_matrix
            for e in choices[t]:
                expected_counts[origins[t], owners[e]] += 1
        np.testing.assert_allclose(result, reference, atol=1e-12)
        np.testing.assert_array_equal(counts, expected_counts)
        assert counts.sum() == tokens * 3
        cases['ep'] += 1

    for stages in range(1, 9):
        for batches in range(1, 13):
            tau = 3.5
            final = pipeline([tau] * stages, batches)[-1]
            np.testing.assert_allclose(final, [(stages + b) * tau for b in range(batches)])
            cases['pipeline'] += 1
    assert pipeline([2., 6., 3., 5.], 4)[-1] == [16., 22., 28., 34.]

    for _ in range(200):
        q = rng.normal(size=(2, 4, 3))
        k, v = rng.normal(size=(2, 7, 3)), rng.normal(size=(2, 7, 3))
        head_parts = np.split(q, 2, axis=1)
        requests = [np.concatenate([part[b:b+1] for part in head_parts], axis=1) for b in range(2)]
        local = [attention(part, k[b:b+1], v[b:b+1]) for b, part in enumerate(requests)]
        returned = [np.concatenate([o[:, h:h+2] for o in local]) for h in [0, 2]]
        np.testing.assert_allclose(np.concatenate(returned, axis=1), dense_attention(q, k, v), atol=1e-12)
        cases['kv_permutation'] += 1

    ring_seconds = lambda batch: 14 * 2e-6 + 1.75 * (batch * 8192 * 2) / 50e9
    np.testing.assert_allclose(ring_seconds(1) * 1e6, 28.57344)
    np.testing.assert_allclose(ring_seconds(8192) * 1e3, 4.72562048)
    np.testing.assert_allclose(ring_seconds(1) * 160 * 1e3, 4.5717504)
    np.testing.assert_allclose(ring_seconds(8192) * 160 * 1e3, 756.0992768)
    bound = namespace['kv_memory_bound']
    for tp, dp in [(4, 4), (8, 2), (16, 1)]:
        local_kv, per_replica, total = bound(tp, dp)
        available = 80_000_000_000 - 140_000_000_000 / tp - 8_000_000_000
        assert per_replica * local_kv <= available < (per_replica + 1) * local_kv
        assert total == per_replica * dp

    manifest = json.loads((SOURCES / 'references.json').read_text())
    for source in manifest['sources']:
        assert hashlib.sha256((SOURCES / source['file']).read_bytes()).hexdigest() == source['sha256']
    embeds = [p.removeprefix('02 Areas/ML & DL/') for p in re.findall(r'!\[\[([^\]]+)\]\]', chapter)]
    assert set(embeds) == {a['asset'] for a in manifest['figures']}
    assert len(embeds) == 11
    for asset in manifest['figures']:
        assert hashlib.sha256((ROOT / asset['asset']).read_bytes()).hexdigest() == asset['sha256']
    original = ROOT / '00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/pipeline-parallel-routing.svg'
    panels = ROOT / '00 Учебник/Assets/Figures/curated/distributed-inference-review-2026-10/harvard-pipeline-panels.svg'
    assert panels.read_text() == original.read_text().replace('viewBox="0 0 680 480"', 'viewBox="0 0 680 365"')
    for marker in ['id="граница-этой-главы"', 'id="небольшой-пример-чтения-конфигурации"',
                   'id="allgather-из-shards-в-полную-копию"', '## Sequence parallelism и context parallelism — не синонимы']:
        assert marker in chapter, marker
    print(json.dumps({'chapter_blocks': len(blocks), 'randomized_checks': cases,
                      'source_snapshots': len(manifest['sources']), 'figures': len(embeds),
                      'source_svg_viewport_crop': 1,
                      'scope': 'CPU algebra only; no measured GPU or serving performance'}, indent=2))


if __name__ == '__main__':
    main()
