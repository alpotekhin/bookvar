#!/usr/bin/env python3
"""MLOps teaching arithmetic/source fidelity; no server or release validation."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from serving_operations_baseline import before_serving_operations
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from check_ml_workflow_examples import SourceText, normalized

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
SOURCES = ROOT / '05 Источники/Originals/mlops-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/mlops-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(function, *args):
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError, OverflowError):
        return
    raise AssertionError(f'{function.__name__} accepted invalid input')


def main():
    chapter = before_serving_operations(CHAPTER.read_text(), CHAPTER)
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 16
    ns = {'__name__': '__chapter_examples__'}
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{index}', 'exec'), ns)
    rng = random.Random(20261020)
    cases = dict.fromkeys(['release_evidence', 'canary_counts', 'rare_detection',
                           'psi', 'burn_budget', 'dependency_closure', 'platform_economics'], 0)

    gate = ns['release_blockers']
    required = ns['required']
    for _ in range(200):
        results, expected = {}, []
        for name in required:
            state = rng.choice(['missing', 'failed', 'different-candidate', 'passed'])
            if state != 'missing':
                results[name] = {'subject': 'other' if state == 'different-candidate' else 'candidate',
                                 'passed': state != 'failed'}
            if state != 'passed':
                expected.append((name, state))
        assert gate('candidate', results, required) == expected
        cases['release_evidence'] += 1
    for args in [('', {}, ['data']), ('candidate', {}, []), ('candidate', {}, ['data', 'data'])]:
        rejects(gate, *args)
    assert gate('candidate', {'data': {'subject': 'candidate', 'passed': 1}}, ['data']) == [('data', 'failed')]

    impact = ns['canary_impact']
    for _ in range(200):
        # Integer populations independently count candidate and old-version errors.
        n, candidate_percent = 10_000, rng.randrange(101)
        old_percent, new_percent = rng.randrange(101), rng.randrange(101)
        candidate_n = candidate_percent * 100
        old_n = n - candidate_n
        old_bad = old_n * old_percent // 100
        new_bad = candidate_n * new_percent // 100
        result = impact(100, 100, candidate_percent / 100, old_percent / 100, new_percent / 100)
        np.testing.assert_allclose(result['all_errors'], old_bad + new_bad)
        np.testing.assert_allclose(result['additional_errors'], old_bad + new_bad - n * old_percent / 100)
        np.testing.assert_allclose(result['candidate_requests'], candidate_n)
        np.testing.assert_allclose(result['overall_error_ratio'], (old_bad + new_bad) / n)
        cases['canary_counts'] += 1
    for args in [(-1, 1, .1, 0, .2), (1, -1, .1, 0, .2),
                 (1, 1, 1.1, 0, .2), (math.inf, 1, .1, 0, .2),
                 (1, math.nan, .1, 0, .2)]:
        rejects(impact, *args)

    detection = ns['requests_for_detection']
    for _ in range(200):
        p, confidence = rng.uniform(.0002, .8), rng.uniform(.5, .999)
        n = detection(p, confidence)
        with localcontext() as ctx:
            ctx.prec = 60
            failure = Decimal(1) - Decimal(str(p))
            target = Decimal(1) - Decimal(str(confidence))
            assert failure ** n <= target
            assert n == 1 or failure ** (n - 1) > target
        cases['rare_detection'] += 1
    assert detection(1) == 1 and detection(.001) == 2995
    for args in [(0,), (-.1,), (1.1,), (.1, 0), (.1, 1), (math.nan,)]:
        rejects(detection, *args)

    psi = ns['population_stability']
    for _ in range(200):
        length = rng.randrange(2, 15)
        p = np.array([rng.uniform(.1, 10) for _ in range(length)])
        q = np.array([rng.uniform(.1, 10) for _ in range(length)])
        p, q = p / p.sum(), q / q.sum()
        # PSI equals the sum of the two directed discrete KL divergences.
        expected = sum(float(a) * math.log(float(a / b)) +
                       float(b) * math.log(float(b / a)) for a, b in zip(p, q))
        np.testing.assert_allclose(psi(p, q), expected, atol=1e-12)
        np.testing.assert_allclose(psi(q, p), expected, atol=1e-12)
        assert psi(p, p) == 0 and psi(p, q) >= 0
        cases['psi'] += 1
    for args in [([], []), ([1, 0], [.5, .5]), ([.5, .5], [1]),
                 ([.4, .4], [.5, .5]), ([math.nan, .5], [.5, .5])]:
        rejects(psi, *args)
    # Marginal distributions can miss a change in feature dependence.
    old_pairs, new_pairs = [(0, 0), (1, 1)], [(0, 1), (1, 0)]
    for axis in [0, 1]:
        assert sorted(row[axis] for row in old_pairs) == sorted(row[axis] for row in new_pairs)
    assert old_pairs != new_pairs

    burn = ns['budget_burn']
    for _ in range(200):
        # A fixed request rate converts the time formula into explicit counts.
        slo = rng.choice([.9, .99, .999])
        rate, period = rng.randrange(1, 1000), rng.randrange(24, 1000)
        window, error = rng.uniform(0, period), rng.random()
        b, spent = burn(slo, error, window, period)
        errors = rate * 3600 * window * error
        allowed_errors = rate * 3600 * period * (1 - slo)
        np.testing.assert_allclose(spent, errors / allowed_errors)
        np.testing.assert_allclose(b, error / (1 - slo))
        cases['burn_budget'] += 1
    for args in [(1, .1, 1, 720), (.999, -1, 1, 720), (.999, .1, 2, 1),
                 (.999, .1, 0, 0), (.999, .1, 1, math.inf)]:
        rejects(burn, *args)
    # Two windows: long-only persistence after recovery must not page.
    threshold = .0144
    assert not (.02 > threshold and .001 > threshold)
    assert .02 > threshold and .03 > threshold

    downstream = ns['downstream_consumers']
    for _ in range(200):
        nodes = [str(i) for i in range(rng.randrange(2, 12))]
        graph = {node: [other for other in nodes if rng.random() < .15] for node in nodes}
        start = rng.choice(nodes)
        reachable = {start}
        for _ in nodes:
            reachable |= {target for source in list(reachable) for target in graph[source]}
        assert downstream(graph, start) == reachable - {start}
        cases['dependency_closure'] += 1
    points = np.array([-.8, .2, .7])
    np.testing.assert_allclose(abs(points[:, None] - points),
                               abs((-points)[:, None] + points))
    assert 2 * .8 - 1 > 0 > 2 * (-.8) - 1

    economics = ns['platform_economics']
    for _ in range(200):
        teams, hours, hourly = rng.randrange(100), rng.randrange(100), rng.randrange(200)
        cost = rng.randrange(1, 500_000)
        benefit, ratio, roi = economics(teams, hours, hourly, cost)
        assert benefit == sum(hours * hourly for _ in range(teams))
        np.testing.assert_allclose(ratio - roi, 1)
        np.testing.assert_allclose(roi * cost, benefit - cost)
        cases['platform_economics'] += 1
    for args in [(-1, 1, 1, 1), (.5, 1, 1, 1), (1, -1, 1, 1),
                 (1, 1, 1, 0), (1, 1, math.inf, 1)]:
        rejects(economics, *args)
    np.testing.assert_allclose(.95 * .94 + .05 * .62, .924)
    np.testing.assert_allclose([810 / 900, 900 / 1000, 95 / 100], [.9, .9, .95])

    manifest = json.loads((SOURCES / 'references.json').read_text())
    assert len(manifest['files']) == 15
    for entry in manifest['files']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    dimensions = {}
    assert len(assets) == 7
    for entry in assets:
        target, source = ROOT / entry['asset'], ROOT / entry['source_asset']
        assert target.read_bytes() == source.read_bytes()
        assert digest(target) == entry['sha256'] == entry['source_asset_sha256']
        assert digest(ROOT / entry['license_evidence']) == entry['license_evidence_sha256']
        assert entry['asset'] in chapter
        if target.suffix == '.svg':
            svg = ET.parse(target).getroot()
            assert svg.tag.endswith('svg')
            assert not any(el.tag.endswith(('script', 'foreignObject')) for el in svg.iter())
            dimensions[target.name] = svg.attrib.get('viewBox')
        else:
            with Image.open(target) as picture:
                picture.load()
                dimensions[target.name] = picture.size
    for excerpt in manifest['excerpts']:
        parser = SourceText()
        parser.feed((ROOT / excerpt['source']).read_text())
        original = normalized(''.join(parser.parts))
        found = re.search(r'<!-- source-excerpt: ' + re.escape(excerpt['id']) +
                          r' -->\n(.*?)\n<!-- /source-excerpt -->', chapter, re.S)
        assert found
        quote = '\n'.join(re.sub(r'^> ?(?:- )?', '', line) for line in found[1].splitlines())
        assert normalized(quote) in original
    cloud = (SOURCES / 'google-mlops.html').read_text()
    for anchor in ['mlops_level_0_manual_process', 'mlops_level_1_ml_pipeline_automation',
                   'mlops_level_2_cicd_pipeline_automation', 'metadata_management']:
        assert f'id="{anchor}"' in cloud
    assert 'CACE' in chapter and 'CACHE' not in chapter
    assert '← [[01 ML workflow]] · Далее: [[03 Edge и federated deployment]]' in chapter
    assert 'Assets/Sources/Harvard ML Systems/' not in chapter
    legacy_anchors = [
        'что-именно-требуется-воспроизвести', 'три-интерфейса-на-которых-ломается-система',
        'интерфейс-данных-и-модели', 'интерфейс-модели-и-инфраструктуры',
        'интерфейс-эксплуатации-и-наблюдения', 'технический-долг-ml-систем',
        'ci-ct-и-cd', 'безопасный-выпуск', 'наблюдаемость-сервиса-данных-модели-и-продукта',
        'состояние-сервиса', 'состояние-данных', 'состояние-модели', 'состояние-продукта',
        'диагностика-и-инцидент', 'от-одной-модели-к-платформе',
        'mlops-maturity--свойства-а-не-список-покупок', 'практика-и-первоисточники',
    ]
    built = (ROOT / 'site/dist/textbook/deployment/mlops/index.html').read_text()
    for anchor in legacy_anchors:
        assert f'id="{anchor}"' in built, anchor
    print(json.dumps({
        'chapter_sha256': digest(CHAPTER), 'executed_blocks': len(blocks),
        'independent_cases': cases, 'source_hashes': 15,
        'original_figures': dimensions, 'exact_english_excerpts': len(manifest['excerpts']),
        'retained_built_heading_anchors': len(legacy_anchors),
        'scope': 'Teaching arithmetic/source identity only; no model, server, rollout or product validation'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
