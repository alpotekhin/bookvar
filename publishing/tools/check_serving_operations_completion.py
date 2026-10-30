#!/usr/bin/env python3
"""Current chapter examples, additive bounds, source identity; no live service."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from serving_operations_baseline import before_serving_operations
from serving_decisions_baseline import before_serving_decisions
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / 'docs/audit/2026-10-29-serving-operations-additions.json'
MANIFEST = ROOT / '00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/assets.json'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rejects(fn, *args):
    try:
        fn(*args)
    except (ValueError, TypeError):
        return
    raise AssertionError(f'{fn.__name__} accepted {args!r}')


def main():
    namespaces, totals = {}, {}
    for row in json.loads(JOURNAL.read_text())['chapters']:
        path = ROOT / row['source']
        current = path.read_text()
        text = before_serving_decisions(before_topic_consolidation(current, path), path)
        assert digest(text.encode()) == row['after_sha256'], row['source']
        # The restoration helper expects the raw current revision and performs
        # its own topic/decision reversal. Do not rewind the same input twice.
        original = before_serving_operations(current, path)
        assert digest(original.encode()) == row['before_sha256']
        reviewed_blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
        old_blocks = re.findall(r'^```python\n(.*?)^```', original, re.M | re.S)
        assert len(reviewed_blocks) - len(old_blocks) == row['new_python_blocks']
        blocks = re.findall(r'^```python\n(.*?)^```', current, re.M | re.S)
        namespace = {'__name__': '__chapter_examples__'}
        for index, block in enumerate(blocks, 1):
            exec(compile(block, f'{path}:python-{index}', 'exec'), namespace)
        namespaces[row['label']] = namespace
        totals[row['label']] = len(blocks)

    energy = namespaces['energy']['interval_energy']
    rng = random.Random(20261029)
    for _ in range(250):
        durations = [rng.randrange(1, 20) for _ in range(rng.randrange(1, 12))]
        powers = [rng.randrange(0, 2000) for _ in durations]
        boundaries = [0, *itertools.accumulate(durations)]
        results = rng.randrange(1, 1000)
        expected = sum(p for dt, p in zip(durations, powers) for _ in range(dt))
        joules, mean, per_result = energy(boundaries, powers, results)
        assert joules == expected
        assert math.isclose(mean * boundaries[-1], expected)
        assert math.isclose(per_result * results, expected)
    for args in [([0, 0], [1], 1), ([1, 0], [1], 1), ([0, 1], [-1], 1),
                 ([0, math.inf], [1], 1), ([0, 1], [math.nan], 1),
                 ([0, 1], [1], -1), ([0, 1], [1], True), ([0], [], 1)]:
        rejects(energy, *args)

    fanout = namespaces['fanout']['independent_fanout_slow']
    for n in range(1, 9):
        for p in [.01, .1, .5]:
            expected = sum(math.prod(p if slow else 1-p for slow in state)
                           for state in itertools.product([0, 1], repeat=n) if any(state))
            assert math.isclose(fanout(p, n), expected, rel_tol=1e-12)
    hedge = namespaces['fanout']['hedge_completion']
    for primary, backup, delay in itertools.product(range(0, 101, 10), repeat=3):
        actual, sent = hedge(primary, backup, delay)
        assert actual <= primary
        assert sent == (primary > delay)
        candidates = [primary] + ([delay + backup] if sent else [])
        assert actual == sorted(candidates)[0]
    for args in [(math.nan, 1, 1), (1, -1, 1), (1, 1, math.inf)]:
        rejects(hedge, *args)
    for args in [(-.1, 1), (1.1, 1), (math.nan, 1), (.1, 0), (.1, True)]:
        rejects(fanout, *args)

    score = namespaces['readiness']['readiness_scores']
    keys = ['data', 'model', 'infrastructure', 'monitoring']
    for _ in range(250):
        sections = {key: [rng.choice([0, .5, 1]) for _ in range(7)] for key in keys}
        sums, value = score(sections)
        assert value == min(sum(section) for section in sections.values())
        assert all(sums[key] == sum(sections[key]) for key in keys)
    rejects(score, {})
    rejects(score, {key: [1] * 6 for key in keys})
    rejects(score, {key: [True] * 7 for key in keys})
    rejects(score, {key: [math.nan] * 7 for key in keys})

    for asset in json.loads(MANIFEST.read_text()):
        target, source = ROOT / asset['asset'], ROOT / asset['source_asset']
        assert target.read_bytes() == source.read_bytes()
        assert digest(target.read_bytes()) == asset['sha256'] == asset['source_asset_sha256']
        assert digest((ROOT / asset['license_evidence']).read_bytes()) == asset['license_evidence_sha256']
        ET.fromstring(target.read_bytes())

    excerpts = [
        ('energy', 'harvard-power-batch-idle-thermal',
         '05 Источники/Originals/inference-benchmark-2026-10-20/harvard-benchmarking.qmd'),
        ('fanout', 'harvard-hedge-cancellation',
         '05 Источники/Originals/inference-queueing-2026-10-20/harvard-model-serving.qmd'),
        ('readiness', 'harvard-readiness-four-areas',
         '05 Источники/Originals/mlops-2026-10-20/harvard-ml-ops.qmd'),
    ]
    rows = {row['label']: row for row in json.loads(JOURNAL.read_text())['chapters']}
    normalize = lambda value: ' '.join(value.split())
    for label, marker, source in excerpts:
        text = (ROOT / rows[label]['source']).read_text()
        quote = re.search(r'<!-- source-excerpt: ' + marker + r' -->\n(.*?)\n<!-- /source-excerpt -->', text, re.S)[1]
        quote = re.sub(r'^> ?', '', quote, flags=re.M)
        assert normalize(quote) in normalize((ROOT / source).read_text()), marker

    print(json.dumps({'all_current_python_blocks': totals, 'new_blocks': 4,
        'energy_cases': 250, 'fanout_enumerated_cases': 24, 'hedge_cases': 1331,
        'rubric_cases': 250, 'unchanged_original_figures': 2,
        'exact_source_passages': 3,
        'scope': 'CPU teaching arithmetic, source fidelity and bounded restoration; no measured power or production serving'}, indent=2))


if __name__ == '__main__':
    main()
