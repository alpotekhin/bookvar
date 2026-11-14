#!/usr/bin/env python3
"""Chapter graph-search reasoning and source identity; no HNSW speed benchmark."""
import hashlib
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid.md'
FIXTURE = ROOT / 'publishing/fixtures/hnswlib-3f342966'


def scan_reference(points, graph, query, entry, ef, k):
    # List implementation of the source routine's unfiltered two-queue rules.
    # No heapq and no calls into the chapter function.
    distance = lambda vertex: (points[vertex] - query) ** 2
    frontier, best, visited = [(distance(entry), entry)], [(distance(entry), entry)], {entry}
    while frontier:
        frontier.sort()
        d, vertex = frontier.pop(0)
        worst = max(x[0] for x in best)
        if len(best) == ef and d > worst:
            break
        for neighbor in graph[vertex]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            item = distance(neighbor), neighbor
            if len(best) < ef or item[0] < max(x[0] for x in best):
                frontier.append(item)
                best.append(item)
                if len(best) > ef:
                    best.remove(max(best, key=lambda x: x[0]))
    return sorted(best)[:k]


def main():
    body = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', body, re.M | re.S)
    block, = [b for b in blocks if 'def search_layer_demo' in b]
    ns = {}
    exec(compile(block, str(CHAPTER) + ':graph-layer', 'exec'), ns)
    search = ns['search_layer_demo']
    rng, compared = random.Random(20261106), 0
    for n in range(3, 14):
        for _ in range(8):
            points = {str(i): rng.uniform(-20, 20) for i in range(n)}
            graph = {u: [v for v in points if v != u and rng.random() < .3] for u in points}
            query, entry = rng.uniform(-20, 20), '0'
            for ef in [1, 2, n]:
                actual = search(points, graph, query, entry, ef, 1)[0]
                assert actual == scan_reference(points, graph, query, entry, ef, 1)
                compared += 1
    points, graph = {'A': 4., 'B': 5., 'C': .1, 'D': .01}, {'A': ['B'], 'B': ['A', 'C'], 'C': ['B'], 'D': []}
    ranked, visited, _ = search(points, graph, 0., 'A', ef=4)
    assert ranked[0][1] == 'C' and 'D' not in visited
    complete = {u: [v for v in points if v != u] for u in points}
    assert search(points, complete, 0., 'A', ef=1)[0][0][1] == 'D'
    for ef, k in [(0, 1), (1, 2), (True, 1), (2, True), (2.5, 1)]:
        try:
            search(points, graph, 0., 'A', ef, k)
        except ValueError:
            continue
        raise AssertionError('invalid search width accepted')
    candidates, accepted = {'A': 1., 'B': 1.1, 'C': -1.2}, []
    for vertex in sorted(candidates, key=lambda v: candidates[v] ** 2):
        if all((candidates[vertex] - candidates[a]) ** 2 >= candidates[vertex] ** 2 for a in accepted):
            accepted.append(vertex)
        if len(accepted) == 2:
            break
    assert accepted == ['A', 'C']
    sources = json.loads((FIXTURE / 'sources.json').read_text())
    quote = body[body.index('> ```ef```'):].split('\n\n')[0]
    normalize = lambda text: ' '.join(text.replace('>', '').split())
    assert normalize(quote) in normalize((FIXTURE / 'ALGO_PARAMS.md').read_text())
    for source in sources['files']:
        assert hashlib.sha256((FIXTURE / source['file']).read_bytes()).hexdigest() == source['sha256']
    header = (FIXTURE / 'hnswalg.h').read_text()
    for statement in ['candidate_dist > lowerBound', 'top_candidates.size() < ef || lowerBound > dist',
                      'if (curdist < dist_to_query)']:
        assert statement in header
    assert (FIXTURE / 'LICENSE').read_bytes() == (ROOT / 'site/public/licenses/hnswlib-3f342966-Apache-2.0.txt').read_bytes()
    print(json.dumps({'chapter_sha256': hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      'random_graph_comparisons': compared, 'disconnected_and_complete_controls': 2,
                      'invalid_width_controls': 5, 'neighbor_selection_example': accepted,
                      'source_commit': sources['commit'],
                      'scope': 'one-layer unfiltered teaching search; not full HNSW construction or a library benchmark'}, indent=2))


if __name__ == '__main__':
    main()
