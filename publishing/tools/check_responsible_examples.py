#!/usr/bin/env python3
"""Check synthetic chapter examples and local source identity, not social outcomes."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/18 Evaluation и методология/59a Responsible systems.md'
SOURCES = ROOT / '05 Источники/Originals/responsible-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/responsible-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 4
    ns = {'__name__': '__chapter_examples__'}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{i}', 'exec'), ns)
        exec(compile(block, f'{CHAPTER}:standalone-{i}', 'exec'), {})
    cases = {'standalone_code_blocks': len(blocks)}
    rates = ns['group_rates']
    rng = random.Random(20261020)
    for _ in range(300):
        tp, fn, fp, tn = [rng.randrange(1, 100) for _ in range(4)]
        y = [1] * (tp + fn) + [0] * (fp + tn)
        pred = [1] * tp + [0] * fn + [1] * fp + [0] * tn
        r = rates(tp, fn, fp, tn)
        assert r['n'] == len(y)
        assert math.isclose(r['accuracy'], sum(a == b for a, b in zip(y, pred)) / len(y))
        assert math.isclose(r['tpr'], sum(p for p, t in zip(pred, y) if t) / sum(y))
        assert math.isclose(r['fpr'], sum(p for p, t in zip(pred, y) if not t) / (len(y) - sum(y)))
    cases['group_counts_independent_row_enumeration'] = 300
    assert math.isclose(ns['a']['selection'] - ns['b']['selection'], 0.3)
    assert math.isclose(ns['a']['tpr'] - ns['b']['tpr'], 13 / 45)
    assert math.isclose(ns['eo_gap'], 19 / 55)
    assert math.isclose((65 + 70) / 200, 0.675)
    for base in [i / 20 for i in range(1, 20)]:
        for t, f in itertools.product([.2, .5, .8], repeat=2):
            selection = t * base + f * (1 - base)
            positive, negative = round(200 * base), round(200 * (1 - base))
            tp, fp = round(t * positive), round(f * negative)
            r = rates(tp, positive - tp, fp, negative - fp)
            assert math.isclose(selection, r['selection'])
            assert math.isclose(r['ppv'], t * base / selection)
    cases['bayes_equal_error_rates'] = 19 * 9
    ra, rb = rates(64, 16, 4, 16), rates(16, 4, 16, 64)
    assert ra['accuracy'] == rb['accuracy'] == 0.8
    assert ra['tpr'] == rb['tpr'] == 0.8 and ra['fpr'] == rb['fpr'] == 0.2
    assert math.isclose(ra['ppv'], 16 / 17) and rb['ppv'] == 0.5
    grid = [[49, 31], [31, 49]]
    assert all(sum(row) == 80 for row in grid)
    assert all(sum(grid[i][j] for i in range(2)) == 80 for j in range(2))
    cases['equal_accuracy_and_intersection_counterexamples'] = 2
    lo10, hi10 = ns['wilson'](9, 10)
    lo100, hi100 = ns['wilson'](90, 100)
    assert math.isclose(lo10, 0.5958499732047615)
    assert math.isclose(hi10, 0.9821237869049271)
    assert 0.8256 < lo100 < 0.8257 and 0.944 < hi100 < 0.945
    assert ns['wilson'](0, 10)[0] < 1e-15
    assert math.isclose(ns['wilson'](10, 10)[1], 1)
    cases['wilson_intervals_and_edges'] = 4
    # Independently average marginal contributions over both feature orders.
    for feature in ['I', 'D']:
        contributions = []
        for order in itertools.permutations(['I', 'D']):
            previous = set()
            for item in order:
                before = ns['v'][frozenset(previous)]
                previous.add(item)
                if item == feature:
                    contributions.append(ns['v'][frozenset(previous)] - before)
        expected = ns['phi_i'] if feature == 'I' else ns['phi_d']
        assert math.isclose(sum(contributions) / len(contributions), expected)
    assert 0.00116 < 1 / (1 + math.exp(6.75)) < 0.00118
    z = lambda income, debt: -1 + .03 * income - 2 * debt
    assert z(50, .5) == -.5
    assert abs(z(50, .25)) < 1e-12 and abs(z(200 / 3, .5)) < 1e-12
    cases['shap_orders_link_and_counterfactual'] = 5
    xs, ys, ws = [.9, 1., 1.1], [.81, 1., 1.21], [1., 2., 1.]
    mx, my = (sum(w * x for w, x in zip(ws, values)) / sum(ws) for values in (xs, ys))
    slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    intercept = my - slope * mx
    assert math.isclose(slope, 2.) and math.isclose(intercept, -.995)
    assert all(math.isclose(abs(slope * x + intercept - y), .005) for x, y in zip(xs, ys))
    assert math.isclose(slope * 3 + intercept, 5.005)
    cases['local_surrogate_fit_and_out_of_neighborhood'] = 4
    for eps in [.1, .5, 1, 2]:
        for y in [97 + .1 * i for i in range(71)]:
            log_ratio = eps * (abs(y - 101) - abs(y - 100))
            assert abs(log_ratio) <= eps + 1e-12
        radius = math.log(20) / eps
        assert math.isclose(1 - math.exp(-eps * radius), .95)
    g = [(3., 4.), (0., 1.)]
    clipped = [tuple(c * min(1., 2 / math.hypot(*v)) for c in v) for v in g]
    assert all(math.hypot(*v) <= 2 + 1e-12 for v in clipped)
    assert all(math.isclose(a, b) for a, b in zip(clipped[0], [1.2, 1.6]))
    assert all(math.isclose(sum(v[i] for v in clipped) / 2, target)
               for i, target in enumerate([.6, 1.3]))
    cases['laplace_density_bound_and_per_example_clipping'] = 4 * 72 + 1
    shards = [list(range(i * 100, (i + 1) * 100)) for i in range(10)]
    # Third slice of seventh shard, regardless of a record's external ID.
    removed = shards[6][51]
    replay = [x for end in (75, 100) for x in shards[6][:end] if x != removed]
    full = [x for shard in shards for end in (25, 50, 75, 100)
            for x in shard[:end] if x != removed]
    assert len(replay) == 173 and len(full) == 2498
    assert removed not in replay and set(replay) <= set(full)
    cases['sisa_data_replay_boundary'] = 1
    for _ in range(100):
        dt, p0, slope = rng.random() * 100 + .01, rng.random() * 1000, rng.random() * 10
        expected = p0 * dt + slope * dt * dt / 2
        assert math.isclose(ns['trace_energy_j']([0, dt], [p0, p0 + slope * dt]), expected)
    assert ns['trace_energy_j']([0, 10, 20], [200, 400, 200]) == 6000
    assert 6000 - 200 * 20 == 2000
    assert math.isclose(6000 / 3.6e6, 1 / 600)
    assert math.isclose(100 * 1.2 * (.7 - .1), 72)
    assert 12000 / (1_000_000 * .001 * .4) == 30
    assert math.isclose(300 / (.05 * 4000 * .4), 3.75)
    assert 20 * 1.8 == 36 and .5 * 3 == 1.5
    cases['energy_trace_carbon_and_lifecycle'] = 107
    scores = [1] * 14 + [.5] * 4 + [0] * 2
    assert scores.count(1) / len(scores) == .7 and sum(scores) / len(scores) == .8
    assert (81 + 5) / 100 == .86 and math.isclose((81 / 90 + 5 / 10) / 2, .7)
    assert .651 < 1 - .9 ** 10 < .652
    assert min([(1.1, 'first'), (.9, 'second')])[1] == 'second'
    cases['safety_scoring_aggregation_and_search'] = 4

    references = json.loads((SOURCES / 'references.json').read_text())
    for row in references['files'] + references['legacy_figures']:
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    assets = json.loads((FIGURES / 'assets.json').read_text())
    assert len(assets) == 3
    for row in assets:
        assert digest(ROOT / row['asset']) == digest(ROOT / row['source_asset']) == row['sha256']
    assert (ROOT / 'site/public/licenses/shap-cc2007ee-MIT.txt').read_text() == (SOURCES / 'shap-LICENSE').read_text()
    normalize = lambda s: ' '.join(s.replace('*', '').replace('>', '').split())
    quote = chapter.split('> **Group A (100 applicants)**:')[1].split('\n\nVijay')[0]
    assert normalize('**Group A (100 applicants)**:' + quote) in normalize((SOURCES / 'responsible_ai.qmd').read_text())
    assert 'Differential privacy is a property of algorithms, and not a property of data.' in BeautifulSoup((SOURCES / 'programming-dp.html').read_text(), 'html.parser').get_text(' ', strip=True)
    cases['frozen_sources_figures_quotes_license'] = len(references['files']) + len(references['legacy_figures']) + 6

    page = BeautifulSoup((ROOT / 'site/dist/textbook/evaluation/responsible-systems/index.html').read_text(), 'html.parser')
    old = ['ответственность--свойство-всей-системы', 'почему-средняя-точность-скрывает-вред',
           'proxy-петли-обратной-связи-и-сдвиг-распределения',
           'объяснение-прозрачность-и-возможность-оспорить-решение',
           'документация-должна-быть-исполнимой', 'экологическая-стоимость-как-системная-метрика',
           'от-перечня-рисков-к-адаптивной-проверке-безопасности',
           'air-bench-покрытие-заданной-таксономии', 'gcg-статический-отказ-ещё-не-означает-устойчивость',
           'предрелизная-проверка', 'практика-и-первоисточники', 'связанные-страницы']
    assert all(page.find(id=name) for name in old)
    article = page.select_one('.sl-markdown-content')
    assert len(article.select('img')) == 12
    assert not article.select('.katex-error')
    assert len(article.select('.katex-display')) == 12
    assert len(article.select('table')) == 3
    assert 'Полный notebook' in article.get_text()
    assert '-2.562' in chapter and '-2.256' not in chapter
    cases['built_anchors_images_math_tables'] = len(old) + 4
    print(json.dumps({'chapter_sha256': digest(CHAPTER), 'python_blocks': len(blocks),
                      'case_groups': cases, 'cases_total': sum(cases.values()),
                      'scope': 'synthetic examples, frozen sources and built page; not a benchmark'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
