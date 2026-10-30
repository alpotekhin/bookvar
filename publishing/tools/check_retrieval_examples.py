#!/usr/bin/env python3
"""Chapter 61 arithmetic, source and built-page checks, not a retrieval benchmark.

Requires numpy, faiss-cpu, Pillow, beautifulsoup4 and Poppler's pdftoppm.
All vectors and queries used for algorithm checks are synthetic.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from bs4 import BeautifulSoup
import faiss
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid.md'
SOURCES = ROOT / '05 Источники/Originals/retrieval-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/retrieval-review-2026-10'


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
    assert len(blocks) == 6
    namespaces, shared = [], {}
    for index, block in enumerate(blocks, 1):
        namespace = {}
        code = compile(block, f'{CHAPTER}:block-{index}', 'exec')
        exec(code, namespace)
        exec(code, shared)
        namespaces.append(namespace)
    counts = {'standalone_and_shared_code_blocks': len(blocks) * 2}
    make, search = [namespaces[0][name] for name in ['make_bm25', 'bm25_search']]
    ivf = namespaces[1]['ivf_flat']
    windows = namespaces[4]['token_windows']
    rrf = namespaces[5]['rrf']
    rng = np.random.default_rng(20261020)

    # Independent brute-force BM25 reference: scan documents, not posting lists.
    for _ in range(60):
        documents = {str(i): rng.choice(list('abcde'), size=int(rng.integers(1, 16))).tolist()
                     for i in range(12)}
        query = rng.choice(list('abcdef'), size=4).tolist()
        postings, lengths = make(documents)
        average, n = np.mean(list(lengths.values())), len(documents)
        expected = []
        for doc, tokens in documents.items():
            frequency, score = Counter(tokens), 0.
            for term in sorted(set(query)):
                if not frequency[term]:
                    continue
                df = sum(term in text for text in documents.values())
                idf = math.log((n + 1) / (df + .5))
                score += idf * frequency[term] * 2.2 / (frequency[term] + 1.2 * (.25 + .75 * len(tokens) / average))
            if score:
                expected.append((doc, score))
        expected.sort(key=lambda row: (-row[1], row[0]))
        actual = search(query, postings, lengths)
        assert [row[0] for row in expected] == [row[0] for row in actual]
        assert np.allclose([row[1] for row in expected], [row[1] for row in actual])
        assert search(query * 3, postings, lengths) == actual
    counts['postings_vs_document_scan'] = 60
    idf = math.log1p(990.5 / 10.5)
    assert math.isclose(idf, 4.557379522151743)
    assert math.isclose(idf * 6.6 / 5.1, 5.8977852639610795)
    assert math.isclose(idf * 6.6 / 4.2, 7.161596391952739)
    assert 1 < 3 * 2.2 / 4.2 < 10 * 2.2 / 11.2 < 2.2
    assert 1000 * 2.2 / 1001.2 < 2.2
    assert math.log1p(.5 / 1000.5) > 0
    rejects(make, {})
    rejects(make, {'a': []})
    for kwargs in [{'k1': 0}, {'k1': math.inf}, {'b': -1}, {'b': 2}, {'b': math.nan}]:
        rejects(search, ['a'], *make({'a': ['a']}), **kwargs)
    counts['bm25_values_saturation_and_invalid_inputs'] = 13

    for _ in range(60):
        vectors, centers, query = rng.normal(size=(25, 4)), rng.normal(size=(5, 4)), rng.normal(size=4)
        exact_distance = ((vectors - query) ** 2).sum(axis=1)
        expected = np.argsort(exact_distance, kind='stable')[:7]
        ids, distance = ivf(query, vectors, centers, 5, 7)
        assert np.array_equal(ids, expected)
        assert np.allclose(distance, exact_distance[expected])
        one, _ = ivf(query, vectors, centers, 1, 7)
        assert set(one) <= set(range(len(vectors)))
    counts['ivf_all_lists_vs_exact'] = 60
    valid = ([0.], [[0.], [1.]], [[0.]], 1, 1)
    for argument, bad in [(0, [math.nan]), (1, []), (1, np.empty((0, 1))),
                          (2, np.empty((0, 1))), (2, [[0., 1.]]),
                          (3, 0), (3, 2), (3, True), (4, 0), (4, .5)]:
        args = list(valid)
        args[argument] = bad
        rejects(ivf, *args)
    ids, distance = ivf([4.9], [[4.], [5.1], [-4.], [14.9]], [[0.], [10.]], 1, 1)
    assert ids.tolist() == [0] and np.allclose(distance, [.81])
    ids, distance = ivf([4.9], [[4.], [5.1], [-4.], [14.9]], [[0.], [10.]], 2, 1)
    assert ids.tolist() == [1] and np.allclose(distance, [.04])
    counts['ivf_boundaries_and_missed_cluster'] = 12

    # A toy graph, not an implementation or benchmark of HNSW construction.
    points, graph, query = np.array([4., 5., .1]), [[1], [0, 2], [1]], 0.
    greedy = 0
    while True:
        best = min([greedy] + graph[greedy], key=lambda i: abs(points[i] - query))
        if best == greedy:
            break
        greedy = best
    frontier, visited = [0], set()
    while frontier:
        current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(graph[current])
    assert greedy == 0 and min(visited, key=lambda i: abs(points[i] - query)) == 2
    counts['synthetic_greedy_graph_counterexample'] = 1

    for _ in range(60):
        books = rng.normal(size=(3, 8, 2))
        vectors, query = rng.normal(size=(12, 6)), rng.normal(size=6)
        codes = ((vectors.reshape(12, 3, 1, 2) - books[None]) ** 2).sum(axis=3).argmin(axis=2)
        reconstructed = books[np.arange(3)[None], codes].reshape(12, 6)
        table = ((query.reshape(3, 1, 2) - books) ** 2).sum(axis=2)
        lookup = table[np.arange(3)[None], codes].sum(axis=1)
        assert np.allclose(lookup, ((reconstructed - query) ** 2).sum(axis=1))
        orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        assert np.allclose(((vectors @ orthogonal - query @ orthogonal) ** 2).sum(axis=1),
                           ((vectors - query) ** 2).sum(axis=1))
    assert 96 * 8 / 8 == 96 and 768 * 4 / 96 == 32
    assert np.array_equal(namespaces[2]['codes'][0], namespaces[2]['codes'][1])
    counts['pq_adc_reconstruction_rotation_and_collision'] = 62

    for length in range(1, 25):
        tokens = list(range(length))
        for width in range(1, 9):
            for overlap in range(width):
                result = windows(tokens, width, overlap)
                assert result[0][0] == 0 and result[-1][1] == length
                assert all(part == tokens[start:end] and 0 < end - start <= width
                           for start, end, part in result)
                assert all(b[0] - a[0] == width - overlap for a, b in zip(result, result[1:]))
                # Any span no longer than overlap+1 is contained in some window.
                for start in range(length):
                    end = min(length, start + overlap + 1)
                    assert any(a <= start and b >= end for a, b, _ in result)
    assert windows([], 6, 3) == []
    for width, overlap in [(0, 0), (True, 0), (3, 3), (3, -1), (3, .5)]:
        rejects(windows, [1, 2, 3], width, overlap)
    counts['chunk_offsets_and_span_coverage_parameter_sets'] = 24 * sum(range(1, 9)) + 6

    assert [doc for doc, _ in rrf([['A', 'B', 'C'], ['C', 'B', 'D']])] == ['C', 'B', 'A', 'D']
    assert rrf([['A'], ['B']], weights=[0, 2]) == [('B', 2 / 61)]
    assert rrf([[], []]) == []
    assert rrf([['B'], ['A']]) == [('A', 1 / 61), ('B', 1 / 61)]
    for kwargs in [{'c': -1}, {'c': math.inf}, {'weights': []}, {'weights': [-1]}, {'weights': [math.nan]}]:
        rejects(rrf, [['A']], **kwargs)
    rejects(rrf, [['A', 'A']])
    scores = np.array([[20., .1], [10., .9]])
    assert scores.sum(axis=1).argmax() == 0 and (scores * [.01, 1]).sum(axis=1).argmax() == 1
    relevant = {'A', 'C'}
    assert len(set(['A', 'B']) & relevant) / 2 == .5
    assert len(set(['C', 'B']) & relevant) / 2 == .5
    assert len(set(['A', 'B', 'C']) & relevant) / 2 == 1
    assert len(set(['B', 'C']) & relevant) / 2 == .5
    ranked, allowed = ['A', 'B', 'C', 'D'], {'C', 'D'}
    assert [doc for doc in ranked[:2] if doc in allowed] == []
    assert [doc for doc in ranked if doc in allowed][:2] == ['C', 'D']
    counts['fusion_scale_weights_union_and_filter_order'] = 17

    references = json.loads((SOURCES / 'references.json').read_text())
    assets = json.loads((FIGURES / 'assets.json').read_text())
    for row in references['files'] + references['legacy_figures']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    for row in assets:
        assert digest(ROOT / row['asset']) == row['sha256']
        assert digest(ROOT / row['source_asset']) == row['source_asset_sha256']
        assert digest(ROOT / row['license_evidence']) == row['license_evidence_sha256']
        with Image.open(ROOT / row['asset']) as target:
            assert target.size == (row['width'], row['height'])
            if row['derivation'] == 'source-raster-copy':
                assert digest(ROOT / row['asset']) == digest(ROOT / row['source_asset'])
            else:
                with Image.open(ROOT / row['source_page_snapshot']) as source:
                    assert np.array_equal(np.array(source.crop(row['crop_box_px']).convert('RGB')),
                                          np.array(target.convert('RGB')))
    # Re-render both historical OpenQA pages independently, not just hash old PNGs.
    with tempfile.TemporaryDirectory(prefix='bookvar-retrieval-slides-') as temp:
        prefix = str(Path(temp) / 'page')
        subprocess.run(['pdftoppm', '-f', '4', '-l', '5', '-scale-to-x', '2560',
                        '-scale-to-y', '-1', '-png', str(SOURCES / 'openqa-part4.pdf'), prefix],
                       check=True, capture_output=True)
        renders = sorted(Path(temp).glob('page-*.png'))
        assert len(renders) == 2
        for source, name in zip(renders, ['openqa-retriever-reader.png', 'openqa-tfidf-retriever.png']):
            with Image.open(source) as a, Image.open(ROOT / '00 Учебник/Assets/Figures/curated/source-first-60-64' / name) as b:
                assert np.array_equal(np.array(a.convert('RGB')), np.array(b.convert('RGB')))
    for source, notice in [('milvus-LICENSE', 'milvus-docs-9acdc999-Apache-2.0.txt'),
                           ('faiss-LICENSE', 'faiss-42aac2fb-MIT.txt'),
                           ('nanopq-LICENSE', 'nanopq-898fad82-MIT.txt')]:
        assert (SOURCES / source).read_text() == (ROOT / 'site/public/licenses' / notice).read_text()
    quote = chapter[chapter.index('> The elements of `dt.dtable`'):].split('\n\n')[0]
    normalized = lambda s: ' '.join(s.replace('>', '').split())
    assert normalized(quote) in normalized((SOURCES / 'nanopq-tutorial.rst').read_text())
    counts['frozen_sources_figures_pixel_identity_notices_quote'] = len(references['files']) + len(references['legacy_figures']) + len(assets) + 6

    page = BeautifulSoup((ROOT / 'site/dist/textbook/retrieval/sparse-dense-hybrid/index.html').read_text(), 'html.parser')
    article = page.select_one('.sl-markdown-content')
    assert article is not None
    for heading in ['1-формальная-постановка', '2-инвертированный-индекс-и-bm25',
                    '3-плотный-поиск', '4-что-именно-индексировать', '5-query-processing',
                    '6-гибридная-выдача', 'два-списка-и-одно-решение',
                    '7-обновления-и-версии-индекса', '8-диагностика-retrieval',
                    'источники-и-дальнейшее-чтение']:
        assert page.find(id=heading), heading
    assert len(article.select('img')) == 8 and len(article.select('pre')) == 6
    assert not article.select('.katex-error') and len(article.select('.katex-display')) == 8
    assert len(page.select('[data-figure-viewer]')) == 1
    assert not article.select('[data-figure-viewer]')
    assert article.select_one('a[href$="/textbook/retrieval/embeddings-metric-learning/"]')
    assert article.select_one('a[href$="/textbook/retrieval/reranking/"]')
    counts['built_anchors_math_images_code_and_viewer'] = 17
    print(json.dumps({'chapter_sha256': digest(CHAPTER), 'faiss_version': faiss.__version__,
                      'case_groups': counts, 'case_groups_total': sum(counts.values()),
                      'scope': 'synthetic arithmetic, real Faiss API, frozen sources and built HTML; no model benchmark'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
