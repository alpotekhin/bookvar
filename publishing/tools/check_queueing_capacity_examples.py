#!/usr/bin/env python3
"""Chapter 58c: CPU teaching calculations and original source fidelity only."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md'
SOURCES = ROOT / '05 Источники/Originals/inference-queueing-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(fn, *args):
    try:
        fn(*args)
    except (AssertionError, ValueError, TypeError):
        return
    raise AssertionError(f'{fn.__name__} accepted invalid input')


def occupancy_area(arrivals, departures):
    events = sorted({0., *arrivals, *departures})
    area = 0.
    for left, right in zip(events, events[1:]):
        midpoint = (left + right) / 2
        area += sum(a <= midpoint < d for a, d in zip(arrivals, departures)) * (right - left)
    return area


def main():
    # Reuse the exact-journal restoration chain: new explanations, math layout
    # and navigation have independent checks; historical assertions stay intact.
    text = subprocess.check_output([
        'node', '--input-type=module', '-e',
        "import {readBeforeReaderRouteNavigation} from './publishing/tools/reader-route-baseline.mjs';"
        "process.stdout.write(readBeforeReaderRouteNavigation(process.argv[1]));",
        CHAPTER.relative_to(ROOT).as_posix(),
    ], cwd=ROOT, text=True)
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    assert len(blocks) == 10, len(blocks)
    namespace = {'__name__': '__chapter_examples__'}
    for index, block in enumerate(blocks):
        exec(compile(block, f'{CHAPTER}:block-{index + 1}', 'exec'), namespace)
    rng = np.random.default_rng(20261020)
    counts = {key: 0 for key in [
        'finite_queue_area', 'service_variability', 'batch_formation',
        'replica_routing', 'kv_reservations', 'workload_flows',
        'replica_reserve', 'burst_recovery', 'deadline_budget', 'retries'
    ]}

    fcfs = namespace['fcfs_schedule']
    for _ in range(300):
        n = int(rng.integers(1, 35))
        arrivals = np.cumsum(rng.uniform(0, 1, n))
        service = rng.uniform(.01, 2, n)
        starts, finishes = fcfs(arrivals, service)
        current = 0.
        expected = []
        for i in range(n):
            while current < arrivals[i]:
                current = arrivals[i]
            expected.append(current)
            current += service[i]
        np.testing.assert_allclose(starts, expected)
        np.testing.assert_allclose(finishes - starts, service)
        area = occupancy_area(arrivals, finishes)
        wait_area = occupancy_area(arrivals, starts)
        np.testing.assert_allclose(area, np.sum(finishes - arrivals))
        np.testing.assert_allclose(wait_area, np.sum(starts - arrivals))
        np.testing.assert_allclose(area - wait_area, service.sum())
        horizon = finishes[-1]
        np.testing.assert_allclose(area / horizon, n / horizon * np.mean(finishes - arrivals))
        counts['finite_queue_area'] += 1
    for args in [([], []), ([1, 0], [1, 1]), ([0], [0]), ([0], [-1]), ([0], [float('nan')])]:
        rejects(fcfs, *args)

    mg1 = namespace['mg1_means']
    for _ in range(300):
        values = rng.uniform(.001, 3, int(rng.integers(2, 8)))
        weights = rng.dirichlet(np.ones(len(values)))
        mean = float(sum(p * s for p, s in zip(weights, values)))
        second = float(sum(p * s * s for p, s in zip(weights, values)))
        rate = float(rng.uniform(.01, .98) / mean)
        rho, wait, total = mg1(rate, mean, second)
        # Independent residual-work fixed-point identity from the slide proof.
        residual = rate * second / 2
        np.testing.assert_allclose(wait, residual + rho * wait)
        np.testing.assert_allclose(total, wait + mean)
        mm = mg1(rate, mean, 2 * mean ** 2)
        dd = mg1(rate, mean, mean ** 2)
        np.testing.assert_allclose(mm[1], 2 * dd[1])
        np.testing.assert_allclose(mm[2], 1 / (1 / mean - rate))
        assert total >= mean
        counts['service_variability'] += 1
    assert namespace['mix_mean'] == .1
    np.testing.assert_allclose(mg1(8, .1, .0829)[1], 1.658)
    assert mg1(8, .1, .0829)[1] > mg1(8, .1, .02)[1]
    for args in [(10, .1, .02), (11, .1, .02), (1, .1, .001), (-1, .1, .02)]:
        rejects(mg1, *args)

    formation = namespace['batch_formation']
    for _ in range(300):
        b = int(rng.integers(1, 35))
        arrivals = np.cumsum(rng.uniform(0, .5, b * int(rng.integers(1, 8))))
        expected = []
        for i in range(0, len(arrivals), b):
            launch = max(arrivals[i:i + b])
            expected.extend(launch - a for a in arrivals[i:i + b])
        np.testing.assert_allclose(formation(arrivals, b), expected)
        counts['batch_formation'] += 1
    rejects(formation, [0, 1, 2], 2)
    rejects(formation, [1, 0], 2)

    route = namespace['route_jobs']
    for _ in range(300):
        c = int(rng.integers(1, 6))
        n = int(rng.integers(1, 30))
        arrivals = np.cumsum(rng.uniform(0, .5, n))
        service = rng.uniform(.1, 2, n)
        initial = rng.uniform(0, 10, c)
        starts, replicas = route(arrivals, service, initial)
        free = dict(enumerate(initial))
        expected = []
        for arrival, duration in zip(arrivals, service):
            r = sorted(free, key=lambda j: (free[j], j))[0]
            start = max(arrival, free[r])
            expected.append((start, r))
            free[r] = start + duration
        np.testing.assert_allclose(starts, [a for a, r in expected])
        np.testing.assert_array_equal(replicas, [r for a, r in expected])
        assigned = rng.integers(0, c, n)
        a_starts, _ = route(arrivals, service, initial, assigned)
        for r in range(c):
            indices = np.flatnonzero(assigned == r)
            r_free = initial[r]
            for index in indices:
                np.testing.assert_allclose(a_starts[index], max(arrivals[index], r_free))
                r_free = a_starts[index] + service[index]
        counts['replica_routing'] += 1

    blocks_for = namespace['kv_blocks']
    slots = namespace['new_request_slots']
    for _ in range(300):
        b = int(rng.integers(1, 33))
        n = int(rng.integers(1, 12))
        current = rng.integers(0, 500, n).tolist()
        remaining = rng.integers(0, 500, n).tolist()
        promised = [c + r for c, r in zip(current, remaining)]
        # Enumerate occupied page identifiers rather than repeat ceil arithmetic.
        expected_blocks = sum(len({position // b for position in range(length)}) for length in promised)
        assert blocks_for(promised, b) == expected_blocks
        pool_blocks = int(rng.integers(0, 5000))
        bytes_per_position = 128
        prompt, output = map(int, rng.integers(1, 300, 2))
        one_blocks = len({p // b for p in range(prompt + output)})
        available = max(0, pool_blocks - expected_blocks)
        expected_slots = len(range(one_blocks, available + 1, one_blocks))
        actual = slots(pool_blocks * b * bytes_per_position, bytes_per_position,
                       current, remaining, prompt, output, b)
        assert actual == expected_slots
        counts['kv_reservations'] += 1
    assert blocks_for([17, 17], 16) == 4
    assert blocks_for([17, 17], 16) != math.ceil(34 / 16)
    rejects(blocks_for, [-1], 16)
    rejects(blocks_for, [1.5], 16)

    flows = namespace['workload_flows']
    for _ in range(300):
        n = int(rng.integers(1, 8))
        weights = rng.dirichlet(np.ones(n))
        inputs = rng.integers(1, 8193, n)
        outputs = rng.integers(1, 1025, n)
        rate = float(rng.uniform(0, 100))
        kv = int(rng.integers(0, 2 ** 30))
        actual = flows(rate, weights, inputs, outputs, kv)
        np.testing.assert_allclose(actual['input_tokens_s'], sum(rate * p * s for p, s in zip(weights, inputs)))
        np.testing.assert_allclose(actual['output_tokens_s'], sum(rate * p * o for p, o in zip(weights, outputs)))
        np.testing.assert_allclose(actual['decode_sequence_steps_s'], actual['output_tokens_s'] - rate)
        np.testing.assert_allclose(actual['kv_bytes_s'], rate * kv)
        counts['workload_flows'] += 1
    assert 5 * 2 ** 30 / (256 * 2 ** 20) == 20
    assert 3 * 400 / 300 == math.ceil(1200 / 300) == 4
    assert 5 * 2 * 3.2 == 32
    assert math.isclose(22 * 3600 * .97, 76824)
    assert round(32 / 76824, 6) == .000417

    replicas = namespace['replica_count']
    for _ in range(300):
        peak = float(rng.uniform(.1, 1000))
        capacity = float(rng.uniform(.1, 30))
        use = float(rng.uniform(.05, 1))
        lost = int(rng.integers(0, 5))
        actual = replicas(peak, capacity, use, lost)
        candidate = lost
        while (candidate - lost) * capacity * use < peak:
            candidate += 1
        assert actual == candidate
        assert (actual - lost) * capacity * use >= peak - 1e-10
        assert (actual - lost - 1) * capacity * use < peak
        counts['replica_reserve'] += 1
    for args in [(0, 6), (22, 0), (22, 6, 0), (22, 6, 1.1)]:
        rejects(replicas, *args)

    fluid = namespace['fluid_step']
    for _ in range(300):
        initial, offered, capacity = map(float, rng.uniform(0, 100, 3))
        duration = float(rng.uniform(.1, 100))
        final, served = fluid(initial, offered, capacity, duration)
        np.testing.assert_allclose(final + served, initial + offered * duration)
        assert final >= 0 and served >= 0
        assert served <= capacity * duration + 1e-8
        split_final, split_served = initial, 0.
        for _ in range(40):
            split_final, processed = fluid(split_final, offered, capacity, duration / 40)
            split_served += processed
        np.testing.assert_allclose([split_final, split_served], [final, served], atol=1e-8)
        counts['burst_recovery'] += 1
    assert (30 - 24) * 60 == 360
    assert 360 / (36 - 30) == 60
    assert 360 / 24 == 15 and 360 / 36 == 10

    fits = namespace['fits_first_token_deadline']
    for _ in range(300):
        stages = rng.uniform(0, 2, 4)
        deadline = float(rng.uniform(.1, 8))
        remaining = deadline
        for stage in stages:
            remaining -= stage
        assert bool(fits(deadline, *stages)) == bool(remaining >= 0)
        counts['deadline_budget'] += 1
    rejects(fits, 1, -1, 0, 0, 0)

    retry = namespace['retry_load']
    for _ in range(300):
        p = float(rng.uniform(0, 1))
        k = int(rng.integers(1, 12))
        attempts, success = retry(p, k)
        # Enumerate mutually exclusive outcomes: first success on attempt i, or all fail.
        reference_attempts = sum(i * p ** (i - 1) * (1 - p) for i in range(1, k + 1)) + k * p ** k
        reference_success = sum(p ** (i - 1) * (1 - p) for i in range(1, k + 1))
        np.testing.assert_allclose([attempts, success], [reference_attempts, reference_success])
        counts['retries'] += 1
    assert 3 ** 3 == 27 and 4 ** 3 == 64
    rejects(retry, 1.1, 3)

    manifest = json.loads((SOURCES / 'references.json').read_text())
    assert len(manifest['files']) == 10
    for entry in manifest['files']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    assert len(assets) == 5
    crops, copies = 0, 0
    with tempfile.TemporaryDirectory(prefix='bookvar-queueing-crops-') as tmp:
        rendered = {}
        for entry in assets:
            target = ROOT / entry['asset']
            assert digest(target) == entry['sha256']
            assert digest(ROOT / entry['source_asset']) == entry['source_asset_sha256']
            evidence = (ROOT / entry['license_evidence']).read_text()
            assert digest(ROOT / entry['license_evidence']) == entry['license_evidence_sha256']
            assert entry['license_url'] in evidence
            assert entry['asset'] in text
            if entry['derivation'] == 'source-raster-copy':
                assert target.read_bytes() == (ROOT / entry['source_asset']).read_bytes()
                assert entry['source_image_url'].removeprefix('https://sre.google') in evidence
                assert entry['license'] == 'CC BY-NC-ND 4.0'
                assert 'crop_box_px' not in entry
                copies += 1
            else:
                page = entry['source_page']
                if page not in rendered:
                    output = Path(tmp) / f'page-{page}'
                    subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page),
                                    '-scale-to', '6000', '-singlefile', '-png',
                                    str(ROOT / entry['source_asset']), str(output)], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    rendered[page] = Image.open(output.with_suffix('.png')).convert('RGB')
                original = rendered[page]
                assert list(original.size) == entry['render_page_size_px']
                crop = original.crop(entry['crop_box_px'])
                actual = Image.open(target).convert('RGB')
                assert crop.size == actual.size and crop.tobytes() == actual.tobytes()
                crops += 1
    for alias in [
        'littles-law-среднее-число-запросов-в-системе',
        'почему-возникает-utilization-knee', 'static-и-continuous-batching',
        'worked-capacity-plan', 'token-weighted-demand',
        'tail-latency-admission-и-autoscaling', 'capacity-experiment'
    ]:
        assert f'id="{alias}"' in text
    for removed in ['queuing-hockey-stick.svg', 'ttft-tpot-timeline.svg', 'batching-strategies.svg']:
        assert removed not in text
    assert '[[58b Benchmarking, SLO и эксплуатация inference|' in text
    assert '19 Deployment, Reliability и MLOps/01 ML workflow|' in text
    print(json.dumps(dict(chapter_blocks=len(blocks), cpu_cases=counts,
                          sources=len(manifest['files']), figures=len(assets),
                          reproduced_pdf_crops=crops, unchanged_originals=copies,
                          scope='CPU/source checks only; no GPU benchmark or autoscaler run'), indent=2))


if __name__ == '__main__':
    main()
