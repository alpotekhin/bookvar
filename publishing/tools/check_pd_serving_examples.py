#!/usr/bin/env python3
"""Chapter 58a2: CPU references and source fidelity, not a serving benchmark."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md'
SOURCES = ROOT / '05 Источники/Originals/pd-serving-2026-10-20'
ASSETS = ROOT / '00 Учебник/Assets/Figures/curated/pd-serving-review-2026-10'


def dense_attention_layer(x, weights):
    """Per-query full-prefix reference, without a cache or mask array."""
    wq, wk, wv = weights
    result = []
    for i, row in enumerate(x):
        scores = (x[:i + 1] @ wk) @ (row @ wq) / np.sqrt(x.shape[-1])
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        result.append(row + probabilities @ (x[:i + 1] @ wv))
    return np.asarray(result)


def expect_assertion(fn, *args):
    try:
        fn(*args)
    except AssertionError:
        return
    raise AssertionError(f'{fn.__name__} accepted invalid input')


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 10, len(blocks)
    ns = {'__name__': '__chapter_examples__'}
    for i, block in enumerate(blocks):
        exec(compile(block, f'{CHAPTER}:python-block-{i + 1}', 'exec'), ns)
    rng = np.random.default_rng(20261020)
    cases = dict(latencies=0, kv_layout=0, layer_transfer=0, token_policy=0,
                 routing=0, reservation=0, queue=0, cpp=0, handoff=0)

    latency = ns['token_latencies']
    for _ in range(200):
        times = np.cumsum(rng.uniform(0, 100, size=int(rng.integers(1, 50))))
        result = latency(times.tolist())
        np.testing.assert_allclose(result['ttft'] + sum(result['itl']), result['e2e'])
        assert len(result['itl']) == len(times) - 1
        if len(times) > 1:
            np.testing.assert_allclose(result['tpot'], (times[-1] - times[0]) / (len(times) - 1))
        else:
            assert result['tpot'] is None
        cases['latencies'] += 1
    for invalid in [[], [-1.], [10., 5.]]:
        expect_assertion(latency, invalid)

    size = ns['kv_bytes']
    assert size(32, 8192, 8, 128, 2) == 2**30
    assert size(32, 6144, 8, 128, 2) / 2**30 == .75
    for _ in range(200):
        layers, length, heads, width = [int(rng.integers(1, m)) for m in [5, 15, 9, 6]]
        logical = rng.normal(size=(2, layers, length, heads, width)).astype(np.float32)
        assert logical.nbytes == size(layers, length, heads, width, 4)
        degrees = [d for d in range(1, heads + 1) if heads % d == 0]
        for p_tp in degrees:
            parts = np.split(logical, p_tp, axis=3)
            wires = [p.transpose(1, 0, 3, 2, 4).copy() for p in parts]
            rebuilt = np.concatenate([w.transpose(1, 0, 3, 2, 4) for w in wires], axis=3)
            np.testing.assert_array_equal(rebuilt, logical)
            for d_tp in degrees:
                recipients = np.split(rebuilt, d_tp, axis=3)
                assert sum(v.nbytes for v in recipients) == logical.nbytes
                np.testing.assert_array_equal(np.concatenate(recipients, axis=3), logical)
                cases['kv_layout'] += 1
    sample = np.arange(2 * 2 * 4 * 4 * 2).reshape(2, 2, 4, 4, 2)
    wire = sample.transpose(1, 0, 3, 2, 4).copy()
    assert wire.nbytes == sample.nbytes and not np.array_equal(wire.reshape(sample.shape), sample)

    finish = ns['layer_transfer_finish']
    for _ in range(500):
        n = int(rng.integers(1, 50))
        compute, transfer = rng.uniform(0, 50, n), rng.uniform(0, 50, n)
        c_end, ready = finish(compute, transfer)
        releases = np.cumsum(compute)
        for j, actual in enumerate(ready):
            # Closed form of a single-channel FIFO dependency graph.
            reference = max(releases[i] + sum(transfer[i:j + 1]) for i in range(j + 1))
            np.testing.assert_allclose(actual, reference)
        assert ready[-1] >= c_end - 1e-10
        assert ready[-1] <= sum(compute) + sum(transfer) + 1e-10
        cases['layer_transfer'] += 1
    assert finish([20.], [10.]) == (20., [30.])
    assert finish([20.] * 4, [10.] * 4)[1][-1] > max(80., 40.)
    expect_assertion(finish, [], [])
    expect_assertion(finish, [1.], [1., 2.])

    policy = ns['first_two_tokens']
    for _ in range(300):
        a, transfer, queue, decode = rng.uniform(0, 100, 4)
        p, d = [policy(a, transfer, queue, decode, name) for name in ['p-first', 'd-ready']]
        np.testing.assert_allclose(p['ttst'], d['ttst'])
        np.testing.assert_allclose(d['ttft'] - p['ttft'], p['first_itl'] - d['first_itl'])
        np.testing.assert_allclose(d['first_itl'], decode)
        for result in [p, d]:
            assert all(v >= -1e-12 for v in result.values())
        cases['token_policy'] += 1

    route = ns['prefill_candidates']
    for _ in range(200):
        tokens, bytes_per_token = 128, 256
        workers = [dict(name=str(i), prefix=int(rng.integers(tokens + 1)),
                        queue=float(rng.uniform(0, 10))) for i in range(4)]
        bandwidth, cost = float(rng.uniform(1, 10000)), float(rng.uniform(.01, .2))
        plans = route(tokens, workers, bytes_per_token, bandwidth, cost)
        reference = []
        prefix = max(w['prefix'] for w in workers)
        for w in workers:
            reference.append((w['queue'] + (tokens - w['prefix']) * cost, w['name'], 'local'))
            if prefix > w['prefix']:
                reference.append((w['queue'] + (prefix - w['prefix']) * bytes_per_token / bandwidth
                                  + (tokens - prefix) * cost, w['name'], 'fetch'))
        assert sorted(plans) == sorted(reference)
        cases['routing'] += 1
    assert ns['fast'][1:] == ('B', 'fetch') and ns['slow'][1:] == ('A', 'local')

    reserve = ns['reserve_in_order']
    for _ in range(500):
        capacity = int(rng.integers(1, 1000))
        active = int(rng.integers(capacity + 1))
        reserved = int(rng.integers(capacity - active + 1))
        incoming = rng.integers(0, 100, 20).tolist()
        accepted, used = reserve(capacity, active, reserved, incoming)
        free = capacity - active - reserved
        for demand, admitted in zip(incoming, accepted):
            assert admitted == (demand <= free)
            if admitted:
                free -= demand
        assert used == capacity - free <= capacity
        cases['reservation'] += 1
    assert ns['wrong_used'] + 4096 > 16384

    wait = ns['md1_wait']
    for _ in range(300):
        service = float(rng.uniform(.01, 1))
        utilization = float(rng.uniform(0, .999))
        rate = utilization / service
        np.testing.assert_allclose(wait(rate, service),
                                   service * utilization / (2 * (1 - utilization)))
        assert wait(rate, service) >= 0
        cases['queue'] += 1
    assert wait(0., .2) == 0.
    assert np.isinf(wait(5., .2)) and np.isinf(wait(6., .2))
    assert r'\lambda(E[O]-1)' in chapter
    assert ns['d_instances'] == 2 and ns['p_instances'] == 5
    assert 10 * (200 - 1) == 1990  # First output is already selected on P.

    chunk, empty = ns['causal_attention_chunk'], ns['empty_kv']
    for _ in range(400):
        n, width = int(rng.integers(2, 16)), int(rng.integers(2, 7))
        x = rng.normal(size=(n, width))
        weights = [[rng.normal(size=(width, width)) * .2 for _ in range(3)] for _ in range(2)]
        full0 = dense_attention_layer(x, weights[0])
        reference = dense_attention_layer(full0, weights[1])
        boundaries = sorted(set([0, n, *rng.integers(1, n, 4).tolist()]))
        stage0, stage1 = empty(width), empty(width)
        parts0 = [chunk(x[a:b], weights[0], stage0) for a, b in zip(boundaries, boundaries[1:])]
        parts1 = [chunk(part, weights[1], stage1) for part in parts0]
        np.testing.assert_allclose(np.concatenate(parts1), reference, atol=1e-12)
        np.testing.assert_allclose(stage0['k'], x @ weights[0][1], atol=1e-12)
        np.testing.assert_allclose(stage1['k'], full0 @ weights[1][1], atol=1e-12)
        cases['cpp'] += 1

        # Feed the pending first output once, at the first uncached position.
        next_x = rng.normal(size=(1, width))
        caches = [{k: v.copy() for k, v in stage.items()} for stage in [stage0, stage1]]
        new0 = chunk(next_x, weights[0], caches[0])
        new1 = chunk(new0, weights[1], caches[1])
        extended = np.concatenate([x, next_x])
        full_extended = dense_attention_layer(dense_attention_layer(extended, weights[0]), weights[1])
        np.testing.assert_allclose(new1, full_extended[-1:], atol=1e-12)
        assert len(caches[0]['k']) == n + 1
        wrong_caches = [{k: v.copy() for k, v in stage.items()} for stage in [stage0, stage1]]
        wrong0 = chunk(x[-1:], weights[0], wrong_caches[0])
        wrong1 = chunk(wrong0, weights[1], wrong_caches[1])
        assert not np.allclose(wrong1, new1)
        cases['handoff'] += 1

    ttft_ok, itl_ok = np.ones(20, dtype=bool), np.ones(20, dtype=bool)
    ttft_ok[0], itl_ok[1] = False, False
    assert ttft_ok.mean() == .95 and itl_ok.mean() == .95
    assert (ttft_ok & itl_ok).mean() == .9
    np.testing.assert_allclose(14 * .95 / 16, .83125)
    assert '| Раздельный пул | 16 | 14 | 95% | 13.3 | 0.83125 |' in chapter

    manifest = json.loads((SOURCES / 'references.json').read_text())
    for source in manifest['sources']:
        assert hashlib.sha256((SOURCES / source['file']).read_bytes()).hexdigest() == source['sha256']
    for excerpt in manifest['excerpts']:
        raw = (SOURCES / excerpt['source']).read_text()
        raw = raw[raw.index(excerpt['start']):]
        raw = raw[:raw.index(excerpt['end_exclusive'])]
        marker = excerpt['id']
        actual = chapter.split(f'<!-- source-excerpt: {marker}:start -->')[1]
        actual = actual.split(f'<!-- source-excerpt: {marker}:end -->')[0]
        actual = '\n'.join(re.sub(r'^> ?', '', line) for line in actual.strip().splitlines())
        assert ' '.join(actual.split()) == ' '.join(raw.split())
    embeds = [p.removeprefix('02 Areas/ML & DL/') for p in re.findall(r'!\[\[([^\]]+)\]\]', chapter)]
    assert set(embeds) == {a['asset'] for a in manifest['figures']} and len(embeds) == 6
    for asset in manifest['figures']:
        assert hashlib.sha256((ROOT / asset['asset']).read_bytes()).hexdigest() == asset['sha256']

    records = json.loads((ASSETS / 'assets.json').read_text())
    with tempfile.TemporaryDirectory(prefix='bookvar-pd-figures-') as temporary:
        rendered = {}
        for record in records:
            pdf = ROOT / record['source_asset']
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == record['source_asset_sha256']
            page = record['source_page']
            if page not in rendered:
                target = Path(temporary) / f'page-{page}'
                subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
                                '-scale-to', str(record['render_scale_to']), '-png', str(pdf), str(target)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                rendered[page] = Image.open(target.with_suffix('.png')).convert('RGB')
            source = rendered[page]
            assert list(source.size) == record['render_page_size_px']
            expected = np.asarray(source.crop(record['crop_box_px']))
            actual = np.asarray(Image.open(ROOT / record['asset']).convert('RGB'))
            np.testing.assert_array_equal(actual, expected)

    for marker in [
        'интерференция-в-общем-gpu-pool', 'что-именно-меняет-disaggregation',
        'kv-transfer-как-отдельная-стадия-запроса',
        'routing-вычислительная-нагрузка-встречается-с-locality',
        'independent-scaling--не-независимое-управление',
        'distserve-splitwise-и-mooncake-три-исследовательских-акцента',
        'как-идеи-реализованы-в-современных-стеках', 'когда-разделение-выгодно',
    ]:
        assert f'id="{marker}"' in chapter
    assert '[[58a Распределённый inference и disaggregated serving|' in chapter
    assert '[[58b Benchmarking, SLO и эксплуатация inference|' in chapter
    print(json.dumps(dict(chapter_blocks=len(blocks), cpu_cases=cases,
                          source_snapshots=len(manifest['sources']),
                          exact_source_excerpts=len(manifest['excerpts']),
                          figure_hashes=len(embeds), reproduced_pdf_crops=len(records),
                          scope='CPU references and source fidelity only; no measured serving performance'),
                     indent=2))


if __name__ == '__main__':
    main()

