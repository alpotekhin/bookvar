#!/usr/bin/env python3
"""Synthetic teaching examples, source identity and built anchors; not a benchmark."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import itertools
import json
import math
from pathlib import Path
import random
import re
from unittest.mock import patch
import warnings

import imblearn
import numpy as np
from PIL import Image
import scipy
from scipy.optimize import minimize_scalar
import sklearn
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import (
    ClusterCentroids, EditedNearestNeighbours, NearMiss,
    NeighbourhoodCleaningRule, TomekLinks,
)
from imblearn.model_selection import InstanceHardnessCV

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация.md'
SOURCES = ROOT / '05 Источники/Originals/imbalance-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/imbalance-review-2026-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    text = re.sub(r':(?:math|class|term|func|ref):`([^`]+)`', r'\1', text)
    return ' '.join(text.replace('$', '').replace('`', '').split())


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.images = set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'img':
            self.images.append(attrs.get('src', ''))


def main():
    assert sklearn.__version__ == '1.9.1'
    assert imblearn.__version__ == '0.14.2'
    chapter = CHAPTER.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 9
    ns = {'__name__': '__chapter_examples__'}
    for number, block in enumerate(blocks, 1):
        exec(compile(block, f'{CHAPTER}:block-{number}', 'exec'), ns)
    standalone = {'__name__': '__standalone_pipeline__'}
    exec(compile(blocks[-1], f'{CHAPTER}:standalone-pipeline', 'exec'), standalone)
    np.testing.assert_allclose(standalone['probability'], ns['probability'], atol=1e-12)
    rng = random.Random(20261020)
    cases = {}
    cases['complete_pipeline_standalone'] = 1

    for _ in range(200):
        n = rng.randrange(2, 100)
        y = np.array([rng.randrange(2) for _ in range(n)])
        score = np.array([rng.uniform(-2, 2) for _ in range(n)])
        threshold = rng.uniform(-2, 2)
        tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
        assert ns['binary_counts'](y, score, threshold) == dict(tn=tn, fp=fp, fn=fn, tp=tp)
    cases['counts_against_sklearn'] = 200

    for _ in range(200):
        n = rng.randrange(2, 40)
        y = [rng.randrange(2) for _ in range(n)]
        score = [rng.randrange(5) / 4 for _ in range(n)]
        capacity, cfp, cfn = rng.randrange(n + 1), rng.randrange(1, 10), rng.randrange(1, 100)
        threshold, cost, alerts = ns['choose_threshold'](y, score, capacity, cfp, cfn)
        unique = sorted(set(score))
        boundaries = [-math.inf, math.inf, *unique]
        boundaries += [(a+b)/2 for a, b in itertools.pairwise(unique)]
        feasible = []
        for t in boundaries:
            chosen = [s >= t for s in score]
            if sum(chosen) <= capacity:
                loss = sum(cfp if label == 0 and pred else cfn if label == 1 and not pred else 0
                           for label, pred in zip(y, chosen))
                feasible.append((loss, sum(chosen)))
        assert (cost, alerts) == min(feasible)
        assert alerts == sum(s >= threshold for s in score)
    cases['all_threshold_regions'] = 200

    for _ in range(200):
        n = rng.randrange(3, 50)
        y = np.array([1, 0] + [rng.randrange(2) for _ in range(n-2)])
        scores = np.array([rng.randrange(8) for _ in range(n)])
        previous_tp, total, ap = 0, int(y.sum()), 0.0
        for t in sorted(set(scores), reverse=True):
            selected = scores >= t
            tp = int(y[selected].sum())
            ap += (tp-previous_tp)/total * tp/int(selected.sum())
            previous_tp = tp
        assert abs(ap - average_precision_score(y, scores)) < 1e-12
        positives, negatives = scores[y == 1], scores[y == 0]
        pair_score = sum(float(a > b) + .5 * float(a == b)
                         for a in positives for b in negatives) / (len(positives)*len(negatives))
        assert abs(pair_score - roc_auc_score(y, scores)) < 1e-12
    cases['ap_groups_and_auc_pairs'] = 200
    assert abs(.99 * 19800 / 19900 - .9850251256281407) < 1e-12
    assert abs((.9 + .1 + 0) / 3 - 1/3) < 1e-12
    assert abs(.9 * .01 / (.9 * .01 + .01 * .99) - 900/1890) < 1e-12

    for _ in range(200):
        p, w0, w1 = rng.uniform(.01, .99), rng.uniform(.1, 10), rng.uniform(.1, 10)
        q = w1*p / (w1*p + w0*(1-p))
        objective = lambda value: -w1*p*math.log(value) - w0*(1-p)*math.log1p(-value)
        optimum = minimize_scalar(objective, bounds=(1e-10, 1-1e-10), method='bounded',
                                  options={'xatol': 1e-12})
        assert abs(optimum.x-q) < 1e-6
    cases['weighted_log_loss_minimum'] = 200

    for _ in range(200):
        a = np.array([rng.uniform(-3, 3) for _ in range(8)])
        b = np.array([rng.uniform(-3, 3) for _ in range(8)])
        fraction = rng.random()
        point = ns['interpolate_sample'](a, b, fraction)
        np.testing.assert_allclose(point-a, fraction*(b-a), atol=1e-12)
        assert (point >= np.minimum(a, b)-1e-12).all()
        assert (point <= np.maximum(a, b)+1e-12).all()
    cases['convex_interpolation'] = 200
    generator = np.random.default_rng(7)
    x = generator.normal(size=(48, 4))
    y = np.array([1]*12 + [0]*36)
    sx, sy = SMOTE(sampling_strategy=.5, k_neighbors=3, random_state=7).fit_resample(x, y)
    assert len(sy) == 54 and sum(sy) == 18
    np.testing.assert_array_equal(sx[:48], x)
    for point in sx[48:]:
        matches = []
        for i, j in itertools.combinations(range(12), 2):
            delta = x[j]-x[i]
            fraction = np.dot(point-x[i], delta)/np.dot(delta, delta)
            if 0 <= fraction <= 1 and np.linalg.norm(point-(x[i]+fraction*delta)) < 1e-10:
                matches.append((i, j))
        assert matches

    for _ in range(100):
        x = np.array([rng.random()*100 for _ in range(25)]).reshape(-1, 1)
        y = np.array([0]*20 + [1]*5)
        distance = np.abs(x-x.T)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.argmin(axis=1)
        for strategy in ['auto', 'all']:
            removed = {i for i, j in enumerate(nearest)
                       if nearest[j] == i and y[i] != y[j]
                       and (strategy == 'all' or y[i] == 0)}
            model = TomekLinks(sampling_strategy=strategy)
            model.fit_resample(x, y)
            assert set(model.sample_indices_) == set(range(25)) - removed
    cases['mutual_neighbours'] = 100

    for _ in range(200):
        minority = [rng.uniform(-10, 10) for _ in range(8)]
        point, k = rng.uniform(-10, 10), rng.randrange(1, 9)
        distances = sorted(math.dist([point], [other]) for other in minority)
        for farthest in [False, True]:
            expected = sum(distances[-k:] if farthest else distances[:k])/k
            assert abs(ns['nearmiss_score'](point, minority, k, farthest)-expected) < 1e-12
    cases['nearest_and_farthest_means'] = 200
    # Keep the targeted class strictly larger, so inferred minority matches the text.
    x = np.array([.1, 4, 20, 30, 0, 1, 10]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    for version, expected in [(1, .1), (2, 4), (3, 4)]:
        model = NearMiss(version=version, n_neighbors=2, n_neighbors_ver3=1,
                         sampling_strategy={0: 1})
        ax, ay = model.fit_resample(x, y)
        assert abs(float(ax[ay == 0].item())-expected) < 1e-12

    x = np.array([0, .1, .2, -.3, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 0, 0])
    for kind, retained in [('mode', True), ('all', False)]:
        model = EditedNearestNeighbours(n_neighbors=3, kind_sel=kind)
        model.fit_resample(x, y)
        assert (0 in model.sample_indices_) == retained
    ax, _ = NeighbourhoodCleaningRule(n_neighbors=3).fit_resample(
        np.array([0, .1, .2, 4, 10]).reshape(-1, 1), [1, 0, 0, 0, 0])
    np.testing.assert_allclose(ax.ravel(), [0, 10])
    x = np.array([[0, 0], [0, 2], [2, 0], [10, 10]])
    y = np.array([0, 0, 0, 1])
    for voting, expected in [('soft', [2/3, 2/3]), ('hard', [0, 0])]:
        cx, cy = ClusterCentroids(sampling_strategy={0: 1}, voting=voting, random_state=7).fit_resample(x, y)
        np.testing.assert_allclose(cx[cy == 0].ravel(), expected)
    cases['cleaning_and_prototype_examples'] = 5

    for _ in range(200):
        prior, target = rng.uniform(.01, .99), rng.uniform(.01, .99)
        likelihood1, likelihood0 = rng.uniform(.01, .99), rng.uniform(.01, .99)
        q = prior*likelihood1 / (prior*likelihood1+(1-prior)*likelihood0)
        expected = target*likelihood1 / (target*likelihood1+(1-target)*likelihood0)
        observed = ns['correct_prior'](q, prior, target)
        assert abs(float(observed)-expected) < 1e-12
        assert abs(float(ns['correct_prior'](observed, target, prior))-q) < 1e-12
    cases['prior_bayes_and_inverse'] = 200

    for _ in range(200):
        n, bins = rng.randrange(2, 100), rng.randrange(1, 12)
        y = np.array([rng.randrange(2) for _ in range(n)])
        ps = np.array([rng.random() for _ in range(n)])
        expected = []
        for k in range(bins):
            indices = [i for i, p in enumerate(ps) if math.floor(p*bins) == k]
            if indices:
                expected.append((sum(ps[indices])/len(indices), sum(y[indices])/len(indices), len(indices)))
        np.testing.assert_allclose(ns['reliability_bins'](y, ps, bins), expected, atol=1e-12)
    assert ns['reliability_bins']([0, 1], [0, 1]) == [(0., 0., 1), (1., 1., 1)]
    cases['reliability_bins'] = 200

    partitions = [set(map(tuple, ns[name])) for name in ['X_fit', 'X_calib', 'X_test']]
    assert len(set.union(*partitions)) == 1000
    for a, b in itertools.combinations(partitions, 2):
        assert not a & b
    cases['disjoint_partitions'] = 3

    # Inspect actual learned state on the declared training indices of every fold.
    for estimator, train, test in zip(ns['cv_results']['estimator'],
                                     ns['cv_results']['indices']['train'],
                                     ns['cv_results']['indices']['test']):
        assert len(train) == 480 and len(test) == 120 and not set(train) & set(test)
        assert ns['y_fit'][train].sum() == 24 and ns['y_fit'][test].sum() == 6
        np.testing.assert_allclose(estimator.named_steps['standardscaler'].mean_,
                                   ns['X_fit'][train].mean(axis=0), atol=1e-12)
        assert estimator.named_steps['standardscaler'].n_samples_seen_ == 480
        assert estimator.named_steps['smote'].sampling_strategy_ == {1: 204}
        with patch.object(estimator.named_steps['smote'], 'fit_resample',
                          side_effect=AssertionError('sampler called at inference')):
            assert estimator.predict_proba(ns['X_fit'][test]).shape == (120, 2)
    assert ns['trained'].named_steps['smote'].sampling_strategy_ == {1: 255}
    assert ns['trained'].named_steps['standardscaler'].n_samples_seen_ == 600
    independent = clone(ns['pipeline']).fit(ns['X_fit'], ns['y_fit'])
    np.testing.assert_allclose(ns['trained'][-1].coef_, independent[-1].coef_, atol=1e-12)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        folds = list(InstanceHardnessCV(ns['pipeline'][-1], n_splits=2).split(
            ns['X_fit'], ns['y_fit'], groups=np.zeros(600)))
        assert len(folds) == 2 and any('ignored' in str(w.message) for w in messages)
    cases['fold_fit_boundaries'] = 5

    manifest = json.loads((SOURCES/'references.json').read_text())
    assert len(manifest['files']) == 50
    for item in manifest['files']:
        assert digest(ROOT/item['path']) == item['sha256'], item['path']
    assets = json.loads((FIGURES/'assets.json').read_text())
    assert assets == manifest['new_assets'] and len(assets) == 10
    dimensions = {}
    for item in [*assets, *manifest['retained_assets']]:
        image = ROOT/item['asset']
        assert image.read_bytes() == (ROOT/item['source_asset']).read_bytes()
        if 'sha256' in item:
            assert digest(image) == item['sha256']
            for field, digest_field in [('source_code', 'source_code_sha256'),
                                        ('source_page_snapshot', 'source_page_sha256'),
                                        ('license_evidence', 'license_evidence_sha256')]:
                assert digest(ROOT/item[field]) == item[digest_field]
        with Image.open(image) as img:
            img.verify()
        with Image.open(image) as img:
            img.load()
            dimensions[image.name] = list(img.size)
    for item in manifest['excerpts']:
        found = re.search(r'<!-- source-excerpt: '+re.escape(item['id'])+r' -->\n(.*?)\n<!-- /source-excerpt -->', chapter, re.S)
        assert found
        quote = '\n'.join(re.sub(r'^> ?', '', row) for row in found[1].splitlines())
        assert normalized(quote) in normalized((ROOT/item['source']).read_text()), item['id']
    assert (ROOT/'site/public/licenses/imbalanced-learn-8504e95f-MIT.txt').read_bytes() == (
        ROOT/'05 Источники/imbalanced-learn/0.14.2/original/LICENSE').read_bytes()
    assert (ROOT/'site/public/licenses/scikit-learn-866c0f51-BSD-3-Clause.txt').read_bytes() == (
        SOURCES/'sklearn-COPYING').read_bytes()

    built = (ROOT/'site/dist/textbook/evaluation/imbalanced-classification/index.html').read_text()
    page = Page()
    page.feed(built)
    anchors = [
        '1-сначала-определить-что-именно-несбалансировано',
        '2-baseline-должен-показывать-цену-бездействия',
        '3-метрика-зависит-от-решения-а-не-от-моды',
        'один-набор-оценок-два-рабочих-порога',
        '4-score-вероятность-и-порог--разные-сущности',
        '5-изменение-функции-потерь-class-и-sample-weights',
        '6-over-sampling-повторение-и-синтетические-примеры',
        '7-under-sampling-уменьшить-доминирование-большинства',
        '8-комбинации-и-balanced-ensembles',
        '9-главная-инженерная-ловушка-leakage',
        '10-кросс-валидация-при-редком-классе',
        '11-calibration-после-изменения-class-prior',
        '12-воспроизводимый-порядок-экспериментов',
        '13-типичные-неверные-выводы',
        'практика-и-первоисточники',
    ]
    assert all(a in page.ids for a in anchors), set(anchors)-page.ids
    for filename in dimensions:
        assert any(filename in src for src in page.images), filename
    assert built.count('class="expressive-code"') == 10  # nine Python + pip command
    assert 'BOOKVARPROTECTED' not in built
    assert '[[02 Areas/ML &amp; DL/' not in built
    assert chr(167) not in chapter
    print(json.dumps(dict(
        chapter_sha256=digest(CHAPTER), executed_blocks=len(blocks), independent_cases=cases,
        source_hashes=len(manifest['files']), figures=dimensions,
        original_english_excerpts=len(manifest['excerpts']), retained_anchors=len(anchors),
        versions=dict(sklearn=sklearn.__version__, imblearn=imblearn.__version__,
                      numpy=np.__version__, scipy=scipy.__version__),
        synthetic_test_metrics=ns['test_metrics'],
        scope='Synthetic arithmetic/API/state checks only; no external benchmark or real deployment'
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

