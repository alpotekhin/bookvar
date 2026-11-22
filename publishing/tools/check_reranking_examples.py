#!/usr/bin/env python3
"""Chapter 62 numerical/source checks; two-pair CPU inference is not a benchmark."""
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
CHAPTER = ROOT / '00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction.md'
SOURCES = ROOT / '05 Источники/Originals/reranking-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/reranking-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 8
    namespaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        ns = {}
        code = compile(block, f'{CHAPTER}:block-{i}', 'exec')
        exec(code, ns)
        exec(code, shared)
        namespaces.append(ns)
    counts = {'standalone_and_shared_code_blocks': 16}
    model_ns = namespaces[0]
    assert model_ns['scores'].dtype == model_ns['torch'].float64
    observed = model_ns['scores'].tolist()
    assert np.allclose(observed, [8.607138, -4.320078], atol=2e-4, rtol=0)
    tok = model_ns['tokenizer']
    assert len(tok.encode('alpha ' * 64, add_special_tokens=False)) == 64
    assert len(tok.encode('beta ' * 500, add_special_tokens=False)) == 500
    features = tok('alpha ' * 64, 'beta ' * 500, truncation='only_second', max_length=512)
    assert len(features['input_ids']) == 512
    assert sum(features['token_type_ids']) == 446  # 445 document tokens plus SEP
    assert tok.num_special_tokens_to_add(pair=True) == 3
    counts['pinned_model_scores_dtype_and_pair_token_budget'] = 6

    bce, pair, listnet, softmax = [namespaces[1][name] for name in ['bce_logit', 'pair_loss', 'listnet', 'softmax']]
    rng = np.random.default_rng(20261020)
    eps = 1e-5
    for _ in range(60):
        s = float(rng.normal() * 5)
        for y in [0, 1]:
            numeric = (bce(s + eps, y) - bce(s - eps, y)) / (2 * eps)
            assert np.isclose(numeric, 1 / (1 + np.exp(-s)) - y, atol=2e-9)
        a, b = rng.normal(size=2)
        numeric = (pair(a + eps, b) - pair(a - eps, b)) / (2 * eps)
        assert np.isclose(numeric, 1 / (1 + np.exp(-(a - b))) - 1, atol=2e-9)
        assert np.isclose(pair(a + 100, b + 100), pair(a, b))
        scores, labels = rng.normal(size=5), rng.integers(0, 4, size=5).astype(float)
        gradient = softmax(scores) - softmax(labels)
        for i in range(5):
            delta = np.eye(5)[i] * eps
            assert np.isclose((listnet(scores + delta, labels) - listnet(scores - delta, labels)) / (2 * eps),
                              gradient[i], atol=2e-9)
        entropy = -np.sum(softmax(labels) * np.log(softmax(labels)))
        assert listnet(scores, labels) >= entropy - 1e-12
    counts['bce_pair_listnet_gradients_and_entropy'] = 60
    assert bce(1000., 0) == 1000 and bce(1000., 1) == 0
    assert pair(1000., -1000.) == 0
    for fn, args in [(softmax, ([],)), (softmax, ([math.inf],)), (bce, (0, 2)),
                     (bce, (math.nan, 1)), (pair, (math.inf, 0)),
                     (listnet, ([1, 2], [1])), (listnet, ([1, 2], [0, math.nan]))]:
        rejects(fn, *args)
    counts['loss_boundaries_and_extreme_logits'] = 10
    teacher_mild, teacher_strong = softmax([2, 1]), softmax([3, 0])
    np.testing.assert_allclose([teacher_mild[0], teacher_strong[0]], [.73105858, .95257413])
    assert np.argmax(teacher_mild) == np.argmax(teacher_strong) == 0
    assert np.isclose(np.sum(teacher_strong * np.log(teacher_strong / teacher_strong)), 0)
    assert np.sum(teacher_strong * np.log(teacher_strong / teacher_mild)) > 0
    counts['same_order_different_teacher_distributions'] = 4

    sigmoid = namespaces[3]['sigmoid']
    assert np.isclose(namespaces[3]['temperature'], 5.41902258270291)
    for _ in range(60):
        scores = rng.normal(size=20)
        temperature = float(rng.uniform(.5, 10))
        assert np.array_equal(np.argsort(scores), np.argsort(sigmoid(scores / temperature)))
        labels = rng.integers(0, 2, size=20)
        assert 0 <= np.mean((sigmoid(scores) - labels) ** 2) <= 1
    assert np.allclose(sigmoid([-1000., 0., 1000.]), [0., .5, 1.])
    rejects(sigmoid, [math.nan])
    counts['temperature_order_and_probability_boundaries'] = 62

    maxsim = namespaces[4]['maxsim_matrix']
    for _ in range(60):
        q, d = rng.normal(size=(4, 8)), rng.normal(size=(7, 8))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        matrix = q @ d.T
        total, winners, contributions = maxsim(matrix)
        assert np.isclose(total, sum(max(float(row @ token) for token in d) for row in q))
        assert np.allclose(contributions, matrix[np.arange(4), winners])
        assert -4 - 1e-12 <= total <= 4 + 1e-12
        assert np.isclose(total, maxsim(matrix[:, rng.permutation(7)])[0])
        assert np.isclose(total, maxsim(matrix[rng.permutation(4)])[0])
        padded = np.column_stack([matrix, np.full(4, 1000.)])
        assert np.isclose(maxsim(padded, np.array([True] * 7 + [False]))[0], total)
        extra = rng.uniform(-1, 1, size=(4, 1))
        assert maxsim(np.column_stack([matrix, extra]))[0] >= total - 1e-12
    for matrix, kwargs in [([], {}), ([[math.nan]], {}), ([[1]], {'doc_mask': [False]}),
                           ([[1]], {'query_mask': [False]}), ([[1]], {'doc_mask': [1]}),
                           ([[1]], {'query_mask': [True, True]})]:
        rejects(maxsim, matrix, **kwargs)
    counts['maxsim_dot_products_permutations_masks_and_boundaries'] = 66
    # Match rows to individual neighbors first, then collect their document IDs.
    token_rows = np.column_stack([namespaces[4]['a'], namespaces[4]['b']])
    token_to_doc = np.array(['A'] * 3 + ['B'] * 3)
    neighbors = np.argsort(-token_rows, axis=1)
    assert set(token_to_doc[neighbors[:, :1]].flat) == {'A'}
    assert set(token_to_doc[neighbors[:, :2]].flat) == {'A', 'B'}
    assert maxsim(namespaces[4]['a'])[0] > maxsim(namespaces[4]['b'])[0]
    counts['token_neighbors_to_document_ids'] = 3
    assert 180 * 128 * 2 == 46080 and 768 * 4 == 3072
    assert 4 + 128 * 2 // 8 == 36 and np.isclose(256 / 36, 7.111111111)
    assert 4 * 512**2 == 1048576
    assert 2 * 32**2 + 2 * 512**2 == 526336
    counts['storage_residual_and_padding_arithmetic'] = 4

    sliding = namespaces[6]['sliding_rerank']
    window_cases = 0
    for n in range(1, 25):
        ids = list(range(n))
        for width in range(1, 9):
            for step in range(1, width + 1):
                result, trace = sliding(ids, lambda values: sorted(values, reverse=True), width, step)
                assert sorted(result) == ids
                assert trace[-1][0] == 0
                assert len(trace) == (1 if n <= width else math.ceil((n - width) / step) + 1)
                assert all(len(before) == min(n, width) and sorted(before) == sorted(after)
                           for _, before, after in trace)
                window_cases += 1
    for kwargs in [{'width': 0}, {'width': True}, {'step': 0}, {'step': 5}]:
        rejects(sliding, [1, 2], lambda v: v, **kwargs)
    rejects(sliding, [1, 1], lambda v: v)
    rejects(sliding, [1, 2], lambda v: [1, 1])
    rejects(sliding, [1, 2], lambda v: [1, 3])
    assert sliding([], lambda v: v) == ([], [])
    counts['sliding_windows_ids_call_counts_and_invalid_permutations'] = window_cases + 8
    windows = namespaces[6]
    assert [after[:2] for _, _, after in windows['trace']] == [
        ['p8', 'p5'], ['p8', 'p3'], ['p2', 'p8']]
    assert all('p8' in before for _, before, _ in windows['trace'])
    assert windows['reverse'].index('p6') == 6
    counts['rankgpt_figure_trace_and_local_sort_limit'] = 3

    metric = namespaces[7]['ranking_metrics']
    for order in itertools.permutations(['A', 'B', 'X']):
        for k in [1, 2, 3]:
            result = metric(order, {'A': 1, 'B': 1, 'C': 1}, k)
            assert result['recall'] == sum(doc != 'X' for doc in order[:k]) / 3
            assert 0 <= result['ndcg'] <= 1 and 0 <= result['rr'] <= 1
    for args in [(['A', 'A'], {'A': 1}, 2), (['A'], {}, 1),
                 (['A'], {'A': 0}, 1), (['A'], {'A': -1}, 1),
                 (['A'], {'A': math.nan}, 1), (['A'], {'A': 1}, 0)]:
        rejects(metric, *args)
    counts['fixed_candidate_metrics_and_oracle_boundaries'] = 24
    e42 = chapter.split('id="rerank-e42-case"', 1)[1].split('![[', 1)[0]
    rows = re.findall(r'^\| ([A-Z]) \| (\d+) \| ([−\d.]+) \|$', e42, re.M)
    baseline = [doc for doc, _, _ in sorted(rows, key=lambda row: int(row[1]))]
    scores = {doc: float(score.replace('−', '-')) for doc, _, score in rows}
    qrels = {'A': 1, 'C': 1}
    reordered = sorted(baseline, key=lambda doc: -scores[doc])
    assert baseline == ['C', 'B', 'A'] and reordered == ['C', 'A', 'B']
    assert metric(baseline, qrels, 3)['recall'] == 1
    assert metric(baseline, qrels, 2)['recall'] == .5
    assert metric(reordered, qrels, 2)['recall'] == 1
    for order in itertools.permutations(baseline[:2]):
        assert metric(order, qrels, 2)['recall'] == .5
    assert math.isclose(metric(baseline[:2], qrels, 2)['ndcg'], 1 / (1 + 1 / math.log2(3)))
    counts['atlas_e42_fixed_candidates_and_cutoff'] = 7

    references = json.loads((SOURCES / 'references.json').read_text())
    for row in references['files'] + references['supporting_files'] + references['legacy_figures']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    assert len(assets) == 7
    for row in assets:
        assert digest(ROOT / row['asset']) == row['sha256']
        assert digest(ROOT / row['source_asset']) == row['source_asset_sha256']
        assert digest(ROOT / row['license_evidence']) == row['license_evidence_sha256']
        with Image.open(ROOT / row['asset']) as output:
            assert output.size == (row['width'], row['height'])
            if row['derivation'] == 'source-raster-copy':
                assert digest(ROOT / row['asset']) == digest(ROOT / row['source_asset'])
            else:
                assert digest(ROOT / row['source_page_snapshot']) == row['source_page_sha256']
                with Image.open(ROOT / row['source_page_snapshot']) as original:
                    assert original.size == tuple(row['render_page_size_px'])
                    assert np.array_equal(np.array(original.crop(row['crop_box_px']).convert('RGB')),
                                          np.array(output.convert('RGB')))
    source = (SOURCES / 'st-retrieve-rerank.md').read_text()
    source = re.sub(r':class:`[^`]*\.([A-Za-z]+)`', r'\1', source)
    quote = chapter[chapter.index('> Given a search query'):].split('\n\n')[0]
    plain = lambda text: ' '.join(text.replace('>', '').replace('**', '').split())
    assert plain(quote) in plain(source)
    assert model_ns['passages'][0] in (SOURCES / 'minilm-model-card.md').read_text()
    assert model_ns['passages'][1] in (SOURCES / 'minilm-model-card.md').read_text()
    assert 'normalize(centroids_' in (SOURCES / 'colbert-residual.py').read_text()
    assert 'not independently verified' in (SOURCES / 'legacy-figure-record.json').read_text()
    counts['frozen_sources_quote_figure_pixels_and_permissions'] = (
        len(references['files']) + len(references['supporting_files']) + len(references['legacy_figures']) + len(assets) + 5)

    page = BeautifulSoup((ROOT / 'site/dist/textbook/retrieval/reranking/index.html').read_text(), 'html.parser')
    article = page.select_one('.sl-markdown-content')
    assert article is not None
    anchors = ['1-bi-encoder-и-cross-encoder-решают-разные-вычислительные-задачи',
               '2-обучающие-примеры-должны-приходить-из-настоящего-retriever',
               '3-калибровка-не-равна-ранжированию', '4-late-interaction-промежуточная-точка-colbert',
               'maxsim-для-двух-кандидатов', '5-cascade-и-бюджет-вычислений',
               '6-метрики-и-ablation', '7-типичные-ошибки', 'источники-и-дальнейшее-чтение']
    assert all(page.find(id=name) for name in anchors)
    assert len(article.select('img')) == 8 and len(article.select('pre')) == 8
    assert len(article.select('details[id^="rerank-code-"]')) == 8
    assert not article.select('.katex-error') and len(article.select('.katex-display')) == 9
    assert article.select_one('a[href$="/textbook/retrieval/sparse-dense-hybrid/"]')
    assert article.select_one('a[href$="/textbook/retrieval/sparse-dense-hybrid/#ret-fusion-limits"]')
    assert article.select_one('a[href$="/textbook/rag/full-pipeline/"]')
    assert not article.select('img[src$="colbert-late-interaction.png"]')
    assert len(page.select('[data-figure-viewer]')) == 1
    counts['built_anchors_images_formulas_code_and_continuation'] = len(anchors) + 9
    print(json.dumps({'chapter_sha256': digest(CHAPTER), 'model_logits_fp64_cpu': observed,
                      'torch_version': model_ns['torch'].__version__,
                      'case_groups': counts, 'cases_total': sum(counts.values()),
                      'scope': 'synthetic numerical checks and two fixed model-card pairs, not a model benchmark'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
