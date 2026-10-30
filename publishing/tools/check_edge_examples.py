#!/usr/bin/env python3
"""Edge/FL worked examples and source identity. Not a fleet or privacy validation."""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image
from pypdf import PdfReader

from check_ml_workflow_examples import SourceText, normalized

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md'
SOURCES = ROOT / '05 Источники/Originals/edge-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/edge-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(function, *args):
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError, OverflowError):
        return
    raise AssertionError(f'{function.__name__} accepted invalid input')


def main():
    chapter = CHAPTER.read_text()
    chapter = re.sub(r'<!-- BEGIN REVIEW: Micro arena 2026-10-27 -->\n.*?'
                     r'<!-- END REVIEW: Micro arena 2026-10-27 -->\n\n', '', chapter, flags=re.S)
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 5
    ns = {'__name__': '__chapter_examples__'}
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{index}', 'exec'), ns)
    rng = random.Random(20261020)
    cases = dict.fromkeys(['weighted_mean', 'local_steps', 'linear_gradients',
                           'mask_cancellation', 'clipping_sensitivity', 'metric_weighting'], 0)

    average = ns['average_clients']
    for _ in range(200):
        count, width = rng.randrange(1, 8), rng.randrange(1, 8)
        weights = [rng.randrange(1, 1000) for _ in range(count)]
        models = [[rng.randrange(-100, 101) for _ in range(width)] for _ in range(count)]
        expected = [float(sum(Fraction(n * row[j], sum(weights))
                              for n, row in zip(weights, models))) for j in range(width)]
        np.testing.assert_allclose(average(models, weights), expected, atol=1e-12)
        base = np.array([rng.uniform(-2, 2) for _ in range(width)])
        np.testing.assert_allclose(average(np.array(models) - base, weights) + base,
                                   expected, atol=1e-12)
        cases['weighted_mean'] += 1
    for args in [([], []), ([[1, 2]], []), ([[1], [1, 2]], [1, 2]),
                 ([[1]], [0]), ([[1]], [-1]), ([[np.nan]], [1]),
                 ([[1]], [np.inf])]:
        rejects(average, *args)

    local = ns['local_steps']
    for _ in range(200):
        start, target = rng.uniform(-10, 10), rng.uniform(-10, 10)
        curvature, rate, steps = rng.uniform(.1, 4), .1, rng.randrange(1, 20)
        closed_form = target + (start - target) * (1 - rate * curvature) ** steps
        np.testing.assert_allclose(local(start, curvature, target, steps, rate),
                                   closed_form, atol=1e-12)
        cases['local_steps'] += 1
    assert abs(.25 * local(2, 1, 0, 2000) + .75 * local(2, 4, 4, 2000) - 3) < 1e-12
    assert abs(3 - 48 / 13) > .6
    assert 2 * ((100 + 32 - 1) // 32) == 8

    # Check every coordinate against finite differences, independently of text formulas.
    for _ in range(200):
        a = np.array([rng.uniform(-2, 2) for _ in range(3)])
        w = np.array([[rng.uniform(-2, 2) for _ in range(2)] for _ in range(3)])
        b = np.array([rng.uniform(-2, 2) for _ in range(2)])
        g = np.array([rng.uniform(-2, 2) for _ in range(2)])
        params = [a, w, b]
        expected = [g @ w.T, np.outer(a, g), g]
        epsilon = 1e-5
        for parameter, derivative in zip(params, expected):
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old + epsilon
                plus = float((a @ w + b) @ g)
                parameter[index] = old - epsilon
                minus = float((a @ w + b) @ g)
                parameter[index] = old
                np.testing.assert_allclose((plus - minus) / (2 * epsilon),
                                           derivative[index], atol=1e-8)
        cases['linear_gradients'] += 1
    a, w, b, g = np.array([2, -1]), np.array([[1, 3], [2, 4]]), np.array([.5, -.5]), np.array([.1, -.2])
    np.testing.assert_allclose(a @ w + b, [.5, 1.5])
    np.testing.assert_allclose(g @ w.T, [-.5, -.6])
    np.testing.assert_allclose(np.outer(a, g), [[.2, -.4], [-.1, .2]])

    for _ in range(200):
        n, modulus = rng.randrange(2, 9), 104729
        updates = [rng.randrange(-100, 101) for _ in range(n)]
        masked = updates.copy()
        for i in range(n):
            for j in range(i + 1, n):
                mask = rng.randrange(modulus)
                masked[i] += mask
                masked[j] -= mask
        masked = [value % modulus for value in masked]
        assert sum(masked) % modulus == sum(updates) % modulus
        cases['mask_cancellation'] += 1

    clip = ns['clip_update']
    for _ in range(200):
        width, bound = rng.randrange(1, 12), rng.uniform(.1, 10)
        a = np.array([rng.uniform(-100, 100) for _ in range(width)])
        b = np.array([rng.uniform(-100, 100) for _ in range(width)])
        ca, cb = clip(a, bound), clip(b, bound)
        assert np.linalg.norm(ca) <= bound + 1e-12
        assert np.linalg.norm(ca - cb) <= 2 * bound + 1e-12
        if np.linalg.norm(a) > bound:
            np.testing.assert_allclose(ca / np.linalg.norm(ca), a / np.linalg.norm(a))
        else:
            np.testing.assert_array_equal(a, ca)
        # Replacement sensitivity for an equal-weight mean with fixed m.
        m = rng.randrange(1, 20)
        assert np.linalg.norm((ca - cb) / m) <= 2 * bound / m + 1e-12
        cases['clipping_sensitivity'] += 1
    for args in [([1, 2], 0), ([1], -1), ([np.nan], 1), ([1], np.inf), ([[1]], 1)]:
        rejects(clip, *args)
    np.testing.assert_allclose(clip([1e308, 1e308], 2), np.array([1, 1]) * np.sqrt(2))
    assert Decimal('.2') * 10 == Decimal('2')
    assert Decimal('.000001') * 10 == Decimal('.00001')

    for _ in range(200):
        n = rng.randrange(2, 8)
        totals = [rng.randrange(1, 100) for _ in range(n)]
        correct = [rng.randrange(t + 1) for t in totals]
        # Explicit object-level outcomes independently establish micro accuracy.
        outcomes = [result for good, total in zip(correct, totals)
                    for result in [1] * good + [0] * (total - good)]
        assert abs(np.mean(outcomes) - sum(correct) / sum(totals)) < 1e-12
        expected_macro = sum(Fraction(good, total) for good, total in zip(correct, totals)) / n
        assert abs(np.mean(np.array(correct) / totals) - float(expected_macro)) < 1e-12
        cases['metric_weighting'] += 1
    assert 4 * 10_000_000 == 40_000_000
    assert 16 * 10_000_000 == 160_000_000
    # Added Harvard accounting uses two FP16 arrays and two FP32 moments.
    assert (2 + 2 + 4 + 4) * 10_000_000 == 120_000_000
    assert Fraction(120_000_000 + 32_000_000, 20_000_000) == Fraction(76, 10)
    assert Fraction(120_000_000 + 32_000_000 + 40_000_000, 20_000_000) == Fraction(96, 10)
    assert 1 + 1 + 4 == 6 > 5
    assert (128 * 10 + 10) * 16 == 20_640
    assert 1000 * 128 * 4 == 512_000
    assert 2 * 100 * 40_000_000 * 100 == 800_000_000_000
    assert 2 * 100 * (100_000 * 4) * 100 == 8_000_000_000
    assert 120 / 200 == .6 and 120 / 10_000 == .012

    manifest = json.loads((SOURCES / 'references.json').read_text())
    assert len(manifest['files']) == 16
    for entry in manifest['files']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    assert len(assets) == 6
    dimensions, reproduced = {}, []
    for entry in assets:
        target, source = ROOT / entry['asset'], ROOT / entry['source_asset']
        assert digest(target) == entry['sha256']
        assert digest(source) == entry['source_asset_sha256']
        assert digest(ROOT / entry['license_evidence']) == entry['license_evidence_sha256']
        assert entry['asset'] in chapter
        if entry['derivation'] == 'pdf-page-render-crop':
            pages = PdfReader(source).pages
            page = pages[entry['source_page'] - 1]
            scale = 6000 / max(float(page.mediabox.width), float(page.mediabox.height))
            import math
            full = [math.ceil(float(page.mediabox.width) * scale),
                    math.ceil(float(page.mediabox.height) * scale)]
            assert full == entry['render_page_size_px']
            x0, y0, x1, y1 = entry['crop_box_px']
            assert 0 <= x0 < x1 <= full[0] and 0 <= y0 < y1 <= full[1]
            with Image.open(target) as im:
                im.load()
                assert im.size == (x1 - x0, y1 - y0)
                dimensions[target.name] = im.size
            with tempfile.TemporaryDirectory() as tmp:
                prefix = str(Path(tmp) / 'figure')
                subprocess.run(['pdftoppm', '-f', str(entry['source_page']), '-l',
                                str(entry['source_page']), '-scale-to', '6000',
                                '-x', str(x0), '-y', str(y0), '-W', str(x1-x0),
                                '-H', str(y1-y0), '-singlefile', '-png', str(source), prefix],
                               check=True, capture_output=True)
                assert digest(Path(prefix + '.png')) == entry['sha256']
            reproduced.append(target.name)
        else:
            assert target.read_bytes() == source.read_bytes()
            if target.suffix == '.png':
                with Image.open(target) as im:
                    im.load()
                    dimensions[target.name] = im.size
            else:
                import xml.etree.ElementTree as ET
                svg = ET.parse(target).getroot()
                assert not any(el.tag.endswith(('script', 'foreignObject')) for el in svg.iter())
                dimensions[target.name] = svg.attrib['viewBox']
    for excerpt in manifest['excerpts']:
        parser = SourceText()
        parser.feed((ROOT / excerpt['source']).read_text())
        original = normalized(''.join(parser.parts))
        found = re.search(r'<!-- source-excerpt: ' + re.escape(excerpt['id']) +
                          r' -->\n(.*?)\n<!-- /source-excerpt -->', chapter, re.S)
        assert found
        quote = '\n'.join(re.sub(r'^> ?', '', line) for line in found[1].splitlines())
        assert normalized(quote) in original
    assert 'creativecommons.org/licenses/by/4.0' in (SOURCES / 'tinytl-arxiv.html').read_text()
    assert 'creativecommons.org/licenses/by/4.0' in (SOURCES / 'pmlr-license.html').read_text()
    assert 'Assets/Sources/Harvard ML Systems/' not in chapter
    anchors = ['сначала-выбрать-где-находится-вычисление-и-состояние',
               'ресурсная-модель-периферийного-устройства', 'ограничить-изменяемое-состояние',
               'федеративное-обучение-как-протокол', 'система-важнее-средней-формулы',
               'приватность-данные-не-ушли-информация-могла-уйти',
               'версионирование-распределённой-адаптации', 'distributed-validation-и-rollback',
               'разные-правила-для-разных-классов-устройств', 'production-checklist',
               'практика-и-первоисточники']
    built = (ROOT / 'site/dist/textbook/deployment/edge-federated/index.html').read_text()
    assert 'BOOKVARPROTECTED' not in built
    assert '[[02 Areas/ML &amp; DL/' not in built
    assert built.count('class="expressive-code"') == 5
    for anchor in anchors:
        assert f'id="{anchor}"' in built, anchor
    assert '← [[02 MLOps]] · Далее:' in chapter
    print(json.dumps({
        'chapter_sha256': digest(CHAPTER), 'executed_blocks': len(blocks),
        'independent_cases': cases, 'source_hashes': len(manifest['files']),
        'original_figures': dimensions, 'reproduced_pdf_crops': reproduced,
        'exact_english_excerpts': len(manifest['excerpts']),
        'retained_built_anchors': len(anchors),
        'scope': 'Arithmetic, derivatives and source identity only; not Android, FL deployment, crypto or DP-accountant validation'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
