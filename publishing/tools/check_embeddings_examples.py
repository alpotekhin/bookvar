#!/usr/bin/env python3
"""Synthetic embedding arithmetic and frozen-source checks; no model benchmark."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning.md'
SOURCES = ROOT / '05 Источники/Originals/embeddings-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/embeddings-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('invalid input was accepted')


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 3
    ns = {}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{i}', 'exec'), ns)
        exec(compile(block, f'{CHAPTER}:standalone-{i}', 'exec'), {})
    cases = {'standalone_code_blocks': len(blocks)}
    pool, unit, ce, metric = [ns[name] for name in ['mean_pool', 'normalize_rows', 'contrastive_row', 'metrics']]
    rng = np.random.default_rng(20261020)
    for _ in range(80):
        n, d = [int(x) for x in rng.integers(2, 20, size=2)]
        states, mask = rng.normal(size=(n, d)), rng.integers(0, 2, size=n)
        mask[0] = 1
        assert np.allclose(pool(states, mask), states[mask.astype(bool)].mean(axis=0))
        q, docs = unit(rng.normal(size=(1, d))), unit(rng.normal(size=(n, d)))
        dot = (q @ docs.T)[0]
        distance = ((docs - q) ** 2).sum(axis=1)
        assert np.allclose(distance, 2 - 2 * dot)
        assert np.array_equal(np.argsort(-dot), np.argsort(distance))
    cases['masked_pooling_and_unit_metric_identity'] = 80
    for function, args in [
        (pool, ([[1., 2.]], [0])), (pool, ([[1., 2.]], [2])),
        (pool, ([[1., np.inf]], [1])), (pool, ([[1., 2.]], [1, 1])),
        (unit, ([[0., 0.]],)), (unit, ([[np.nan, 1.]],)),
        (ce, ([1., 2.], [])), (ce, ([1., 2.], [0, 0])),
        (ce, ([1., 2.], [2])), (ce, ([1., 2.], [.5])),
        (ce, ([np.inf, 1.], [0])), (ce, ([], [0])),
        (metric, (['a'], {}, 1)), (metric, (['a', 'a'], {'a': 1}, 2)),
        (metric, (['a'], {'a': -1}, 1)), (metric, (['a'], {'a': 1}, 0))
    ]:
        rejects(function, *args)
    cases['invalid_input_boundaries'] = 16

    # Finite differences independently verify both one-positive CE and set-positive log loss.
    for _ in range(80):
        scores = rng.normal(size=5)
        for positive in [[0], [0, 2]]:
            loss, prob, grad = ce(scores, positive)
            assert np.isclose(prob.sum(), 1) and np.isclose(grad.sum(), 0)
            assert np.isclose(loss, -np.log(prob[positive].sum()))
            assert np.isclose(ce(scores + 100, positive)[0], loss)
            numerical = []
            for i in range(len(scores)):
                delta = np.eye(len(scores))[i] * 1e-5
                numerical.append((ce(scores + delta, positive)[0] - ce(scores - delta, positive)[0]) / 2e-5)
            assert np.allclose(grad, numerical, atol=2e-9, rtol=2e-7)
    cases['softmax_gradients_and_shift_invariance'] = 160
    assert math.isclose(ce([2, 1.9, 0], [0])[0], .7130344617594373)
    assert math.isclose(ce([4, 2, 0], [0])[0], .1429316284998996)
    assert ce([0], [0])[0] == 0
    assert math.isclose(ce([0, 0], [0])[0], math.log(2))
    concentrated = np.array([10., -10., -20.])
    set_loss = ce(concentrated, [0, 1])[0]
    prob = ce(concentrated, [0, 1])[1]
    assert set_loss < 1e-10 and -np.log(prob[:2]).mean() > 9.9
    cases['temperature_accumulation_and_positive_objective'] = 5

    assert math.isclose((.6 - 2) ** 2, 1.96)
    assert math.isclose((105 - 103), (5 - 3))
    assert math.isclose(max(0, .2 + .5 - .6), .1)
    assert max(0, .2 + .3 - .6) == 0
    assert math.isclose(max(0, 1 - .6) ** 2, .16)
    assert math.isclose((unit([[100, 1]]) @ unit([[100, -1]]).T)[0, 0], 9999 / 10001)
    full = unit([[1, 0, 10], [0, 1, 10]])
    prefix = unit(np.array([[1, 0, 10], [0, 1, 10]])[:, :2])
    assert math.isclose(full[0] @ full[1], 100 / 101) and prefix[0] @ prefix[1] == 0
    assert not np.isclose(np.linalg.norm(full[0, :2]), 1)
    assert 1_000_000 * 768 * 4 == 3_072_000_000
    assert math.isclose(3_072_000_000 / 2**30, 2.86102294921875)
    assert 768 / 128 == 6 and math.isclose(64 / 768, 1 / 12)
    rotation = np.array([[0., -1.], [1., 0.]])
    x = np.array([1., 0.])
    assert (rotation @ x) @ (rotation @ x) == 1 and (rotation @ x) @ x == 0
    points = np.array([[2., 1.], [2., -1.], [-2., 1.], [-2., -1.]])
    assert np.allclose(np.cov(points, rowvar=False, bias=True), np.diag([4, 1]))
    assert np.allclose(np.cov(points / [2, 1], rowvar=False, bias=True), np.eye(2))
    cases['loss_margins_anisotropy_whitening_dimensions_and_rotation'] = 15

    ids, qrels = ns['ids'].tolist(), ns['qrels']
    ideal = 7 + 3 / math.log2(3) + .5
    for ranked in itertools.permutations(ids):
        r = metric(ranked, qrels, 3)
        gains = [2 ** qrels.get(doc, 0) - 1 for doc in ranked[:3]]
        expected_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        first = next((i + 1 for i, doc in enumerate(ranked[:3]) if doc in qrels), None)
        assert math.isclose(r['recall'], sum(doc in qrels for doc in ranked[:3]) / 3)
        assert math.isclose(r['ndcg'], expected_dcg / ideal)
        assert r['rr'] == (1 / first if first else 0)
        assert 0 <= r['ndcg'] <= 1
    cases['rank_metrics_all_five_document_permutations'] = math.factorial(5)
    # Index overlap has no access to qrels, unlike relevance recall.
    exact, approximate = {'d1', 'd2'}, {'d1', 'd2'}
    assert len(exact & approximate) / 2 == 1
    assert len(exact & set(qrels)) / len(qrels) == 1 / 3
    cases['ann_overlap_is_not_relevance_recall'] = 1

    references = json.loads((SOURCES / 'references.json').read_text())
    assets = json.loads((FIGURES / 'assets.json').read_text())
    for row in references['files'] + references['legacy_figures']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    for row in assets:
        assert digest(ROOT / row['asset']) == digest(ROOT / row['source_asset']) == row['sha256']
        assert digest(ROOT / row['license_evidence']) == row['license_evidence_sha256']
        with Image.open(ROOT / row['asset']) as image:
            image.verify()
            assert image.size == (row['width'], row['height'])
    for source, target in [('sbert-LICENSE', 'sbert-8a880ec7-Apache-2.0.txt'),
                           ('simcse-LICENSE', 'simcse-49fa580a-MIT.txt')]:
        assert (ROOT / 'site/public/licenses' / target).read_text() == (SOURCES / source).read_text()
    normalize = lambda s: ' '.join(s.replace('>', '').split())
    for start, source in [('> For a batch size of 64', 'msmarco.md'),
                          ('> Despite the embeddings being smaller', 'matryoshka.md')]:
        quote = chapter[chapter.index(start):].split('\n\n')[0]
        assert normalize(quote) in normalize((SOURCES / source).read_text())
    assert 'license: cc-by-nc-sa-4.0' in (SOURCES / 'hf-images-README.md').read_text()
    cases['frozen_sources_original_figures_quotes_licenses'] = len(references['files']) + len(references['legacy_figures']) + len(assets) + 5

    page = BeautifulSoup((ROOT / 'site/dist/textbook/retrieval/embeddings-metric-learning/index.html').read_text(), 'html.parser')
    article = page.select_one('.sl-markdown-content')
    assert article is not None
    old = ['1-от-токенных-представлений-к-представлению-последовательности',
           '2-почему-обычный-bert-не-является-готовой-моделью-предложений',
           '3-контрастивная-задача', 'один-пакет-в-числах', '4-пары-определяют-геометрию',
           '5-triplet-multiple-negatives-и-distillation', '6-анизотропия-pooling-и-размерность',
           '7-как-проверять-эмбеддинг', 'источники-и-дальнейшее-чтение']
    assert all(page.find(id=name) for name in old)
    assert len(article.select('img')) == 8
    assert len(article.select('pre')) == 3
    assert not article.select('.katex-error')
    assert len(article.select('.katex-display')) == 11
    assert len(article.select('blockquote')) == 2
    assert 'cross-encoder MiniLM' in article.get_text(' ', strip=True)
    assert 'без NLI-классификатора' in article.get_text(' ', strip=True)
    assert article.select_one('a[href$="/textbook/retrieval/sparse-dense-hybrid/"]') is not None
    cases['built_anchors_images_math_quotes_and_next_link'] = len(old) + 7
    print(json.dumps({'chapter_sha256': digest(CHAPTER), 'case_groups': cases,
                      'cases_total': sum(cases.values()),
                      'scope': 'synthetic examples and built page; no real-model retrieval benchmark'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
