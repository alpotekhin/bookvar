#!/usr/bin/env python3
"""ML workflow: actual chapter examples, source quotes and original figures.

CPU teaching checks only. Does not execute downloaded course programs,
train a model, validate a server, or estimate a production effect.
"""
from __future__ import annotations

import copy
import hashlib
from html.parser import HTMLParser
import json
import random
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md'
SOURCES = ROOT / '05 Источники/Originals/ml-workflow-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(function, *args):
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError, OverflowError):
        return
    raise AssertionError(f'{function.__name__} accepted invalid input')


class SourceText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'li', 'h1', 'h2', 'h3', 'h4', 'div', 'section', 'br'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip -= 1
        if tag in ('p', 'li', 'h1', 'h2', 'h3', 'h4', 'div', 'section'):
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def normalized(text):
    return ' '.join(text.split())


def main():
    text = CHAPTER.read_text()
    old_examples = re.sub(r'<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->.*?'
                          r'<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->\n\n', '', text, flags=re.S)
    old_examples = re.sub(r'<!-- BEGIN REVIEW: model export 2026-10-27 -->.*?'
                          r'<!-- END REVIEW: model export 2026-10-27 -->\n\n', '', old_examples, flags=re.S)
    blocks = re.findall(r'^```python\n(.*?)^```', old_examples, re.M | re.S)
    assert len(blocks) == 7, len(blocks)
    ns = {'__name__': '__chapter_examples__'}
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{index}', 'exec'), ns)
    rng = random.Random(20261020)
    counts = {name: 0 for name in ['triage', 'point_in_time', 'release', 'selection_weights']}

    metrics = ns['triage_metrics']
    for _ in range(200):
        # Construct individual true/predicted labels independently of the formula.
        pairs = [(rng.randrange(2), rng.randrange(2)) for _ in range(50)]
        pairs.append((1, 0))
        confusion = [sum(y == truth and p == pred for y, p in pairs)
                     for truth, pred in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        missed, false = rng.randrange(10), rng.randrange(10)
        result = metrics(*confusion, missed, false)
        np.testing.assert_allclose(result['accuracy'],
                                   sum(y == p for y, p in pairs) / len(pairs))
        np.testing.assert_allclose(result['recall'],
                                   sum(y == p == 1 for y, p in pairs) / sum(y for y, _ in pairs))
        assert result['engineer_load'] == sum(p for _, p in pairs)
        assert result['cost'] == sum(missed if (y, p) == (1, 0) else
                                     false if (y, p) == (0, 1) else 0 for y, p in pairs)
        counts['triage'] += 1
    for args in [(0, 0, 0, 0), (0, 0, 1, 1), (-1, 1, 1, 1), (.5, 1, 1, 1),
                 (1, 1, 1, 1, -1), (1, 1, 1, 1, 1, -1)]:
        rejects(metrics, *args)
    assert [ns[name]['cost'] for name in ['a', 'b', 'c']] == [1450., 1200., 1200.]

    as_of = ns['feature_as_of']
    for _ in range(200):
        rows = [{'state_at': state, 'available_at': rng.randrange(25), 'value': state * 7}
                for state in range(20)]
        rng.shuffle(rows)
        time = rng.randrange(25)
        expected = sorted((r['state_at'], r['available_at'], r['value'])
                          for r in rows if max(r['state_at'], r['available_at']) <= time)
        if expected:
            assert as_of(rows, time) == expected[-1][2]
        else:
            rejects(as_of, rows, time)
        counts['point_in_time'] += 1
    assert as_of(ns['history'], 6) == 2
    assert as_of(ns['history'], 11) == 4
    rejects(as_of, ns['history'], 5)
    rejects(as_of, [], 20)
    # A late revision of an old state cannot supersede a more recent state.
    assert as_of([{'state_at': 5, 'available_at': 12, 'value': 2},
                  {'state_at': 9, 'available_at': 10, 'value': 3}], 12) == 3

    identify = ns['release_id']
    for index in range(200):
        release = {**ns['release'], 'runtime_config': {'threads': index + 1, 'precision': 'bf16'}}
        shuffled = list(release.items())
        rng.shuffle(shuffled)
        assert identify(dict(shuffled)) == identify(release)
        changed = copy.deepcopy(release)
        changed['runtime_config']['precision'] = 'fp32'
        assert identify(changed) != identify(release)
        for key in ns['release']:
            changed = {**release, key: str(release[key]) + '-changed'}
            assert identify(changed) != identify(release)
        assert re.fullmatch('[0-9a-f]{64}', identify(release))
        counts['release'] += 1
    rejects(identify, {'threshold': float('nan')})
    rejects(identify, {'threshold': float('inf')})

    weight = ns['weighted_confusion']
    for _ in range(200):
        # Expanded finite counts check the arithmetic, not estimator variance.
        strata, expected = [], dict.fromkeys(['tp', 'fn', 'fp', 'tn'], 0)
        for _ in range(rng.randrange(1, 6)):
            multiplier = rng.randrange(1, 20)
            row = {key: rng.randrange(20) for key in expected}
            for key in expected:
                expected[key] += len([key] * row[key] * multiplier)
            strata.append({**row, 'inclusion_probability': 1 / multiplier})
        result = weight(strata)
        for key in expected:
            np.testing.assert_allclose(result[key], expected[key])
        counts['selection_weights'] += 1
    for probability in [0, -1, 1.1, float('nan')]:
        rejects(weight, [{'tp': 1, 'inclusion_probability': probability}])
    rejects(weight, [{'tp': -1, 'inclusion_probability': 1}])
    np.testing.assert_allclose(ns['weighted_recall'], 1 / 3)
    assert ns['weighted'] == {'tp': 90., 'fn': 180., 'fp': 10., 'tn': 720.}

    manifest = json.loads((SOURCES / 'references.json').read_text())
    assert len(manifest['files']) == 11
    for entry in [*manifest['files'], manifest['course_license']]:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    assert len(assets) == len(manifest['figures']) == 6
    pipelines = (SOURCES / 'google-pipelines.html').read_text()
    assert 'https://creativecommons.org/licenses/by/4.0/' in pipelines
    dimensions = {}
    for entry in assets:
        output, source = ROOT / entry['asset'], ROOT / entry['source_asset']
        assert output.read_bytes() == source.read_bytes()
        assert digest(output) == entry['sha256'] == entry['source_asset_sha256']
        assert digest(ROOT / entry['license_evidence']) == entry['license_evidence_sha256']
        assert entry['source_image_url'].removeprefix('https://developers.google.com') in pipelines
        assert entry['license'] == 'CC BY 4.0'
        assert entry['asset'] in text
        raw = output.read_bytes()
        assert raw[:8] == b'\x89PNG\r\n\x1a\n' and raw[12:16] == b'IHDR'
        dimensions[output.name] = struct.unpack('>II', raw[16:24])

    parser = SourceText()
    parser.feed((SOURCES / 'google-rules-of-ml.html').read_text())
    original = normalized(''.join(parser.parts))
    for excerpt in manifest['excerpts']:
        match = re.search(r'<!-- source-excerpt: ' + re.escape(excerpt['id']) +
                          r' -->\n(.*?)\n<!-- /source-excerpt -->', text, re.S)
        assert match, excerpt['id']
        quote = '\n'.join(re.sub(r'^> ?(?:- )?', '', line) for line in match[1].splitlines())
        assert normalized(quote) in original, excerpt['id']
    assert len(manifest['excerpts']) == 2
    trace = (SOURCES / 'edls-train-model.py').read_text()
    assert 'pretrained=True' in trace and 'torch.jit.trace' in trace
    assert '.backward(' not in trace and 'torch.optim' not in trace
    anchors = [
        'два-параллельных-контура', '1-постановка-задачи-как-системная-спецификация',
        '2-сбор-данных-начинается-с-условий-эксплуатации',
        '3-разработка-модели--работа-с-несколькими-бюджетами',
        '4-оценивание-валидация-и-готовность-к-развёртыванию', '5-от-функции-к-сервису',
        '6-обратная-связь-из-эксплуатации', 'цена-поздно-обнаруженного-ограничения',
        'минимальный-контракт-между-стадиями',
    ]
    for anchor in anchors:
        assert f'<a id="{anchor}"></a>' in text
    assert 'Далее: [[02 MLOps]]' in text and '58c Queueing и capacity planning|' in text
    assert 'exponential' not in text and 'ml-workflow-review-2026-10/' in text
    print(json.dumps({
        'chapter_sha256': digest(CHAPTER), 'executed_blocks': len(blocks),
        'independent_cases': counts, 'source_hashes': len(manifest['files']),
        'original_figures': dimensions, 'exact_original_excerpts': 2,
        'retained_heading_anchors': len(anchors),
        'scope': 'CPU examples and source fidelity only; no training or serving validation'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
