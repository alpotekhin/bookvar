#!/usr/bin/env python3
"""Check new retraining/A-B/host-loading explanations; no serving benchmark."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import NormalDist

from serving_decisions_baseline import before_serving_decisions
from serving_node_baseline import before_serving_node
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / '05 Источники/Originals/serving-decisions-2026-10-29'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    text = re.sub(r'^# ?', '', text, flags=re.M)
    text = re.sub(r'^> ?', '', text, flags=re.M)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'`([^`<>]+) <[^>]+>`_', r'\1', text)
    return ' '.join(text.replace('`', '').split())


def excerpt(text, marker):
    return re.search(r'<!-- source-excerpt: ' + marker + r' -->\n(.*?)\n<!-- /source-excerpt -->', text, re.S)[1]


def rejects(fn, *args):
    try:
        fn(*args)
    except (ValueError, TypeError):
        return
    raise AssertionError(f'{fn.__name__} accepted {args}')


def main():
    rows = {}
    for name, restore in [('decisions', before_serving_decisions), ('node', before_serving_node)]:
        row = json.loads((ROOT / f'docs/audit/2026-10-29-serving-{name}-additions.json').read_text())
        current = (ROOT / row['source']).read_text()
        text = before_topic_consolidation(current, row['source'])
        assert digest(text.encode()) == row['after_sha256'], row['source']
        assert digest(restore(text, row['source']).encode()) == row['before_sha256']
        # An addition outside the bounded section must survive restoration.
        assert restore(text + '\nUNREVIEWED\n', row['source']).endswith('\nUNREVIEWED\n')
        changed = text.replace('An automatic training pipeline' if name == 'decisions' else 'The process diagrams above', 'MODIFIED', 1)
        try:
            restore(changed, row['source'])
        except AssertionError:
            pass
        else:
            raise AssertionError('unrecorded block edit was accepted')
        # Historical hashes describe their recorded revision; execute the
        # actual current chapter below, including later reviewed navigation.
        rows[name] = (row, current)

    text = rows['decisions'][1]
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    ns = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f'MLOps:block-{i}', 'exec'), ns)
    plan = ns['retraining_plan']
    rng = random.Random(20261029)
    for _ in range(300):
        q, value, a = rng.uniform(1e3, 1e6), rng.uniform(.1, 2.), rng.uniform(.5, 1.)
        gamma = rng.uniform(.001, .2)
        scale = q * value * a
        cost = rng.uniform(.0001, .8) * scale / gamma
        approximate, optimum = plan(q, value, a, gamma, cost)
        x = gamma * optimum
        assert math.isclose(scale/gamma * (1-(1+x)*math.exp(-x)), cost, rel_tol=1e-9)
        def average(t):
            return (scale*(t + math.expm1(-gamma*t)/gamma) + cost)/t
        assert average(optimum) < average(optimum*.9)
        assert average(optimum) < average(optimum*1.1)
        # Trapezoidal integration independently checks accumulated staleness.
        steps = 4000
        dt = optimum/steps
        losses = [scale*(1-math.exp(-gamma*i*dt)) for i in range(steps+1)]
        integrated = dt*(sum(losses[1:-1]) + (losses[0]+losses[-1])/2)
        formula = scale*(optimum + math.expm1(-gamma*optimum)/gamma)
        assert math.isclose(integrated, formula, rel_tol=1e-6)
        assert approximate < optimum
    for bad in [(0,1,.9,.1,1), (1,1,1.1,.1,1), (1,1,.9,math.nan,1), (1,1,.9,.1,-1)]:
        rejects(plan, *bad)
    assert math.isinf(plan(1,1,1,1,1)[1])

    sample = ns['ab_sample_size']
    normal = NormalDist()
    for _ in range(300):
        p0 = rng.uniform(.02, .4)
        p1 = p0 + rng.uniform(.0001, .01)
        power = rng.uniform(.6, .95)
        rough, n = sample(p0, p1, .05, power)
        pooled = (p0+p1)/2
        null_sd = math.sqrt(2*pooled*(1-pooled)/n)
        alternative_sd = math.sqrt((p0*(1-p0)+p1*(1-p1))/n)
        cutoff = normal.inv_cdf(.975)*null_sd
        achieved = 1-normal.cdf((cutoff-(p1-p0))/alternative_sd)
        assert achieved >= power-1e-12
        # The refined calculation is symmetric when the variants are swapped.
        assert sample(p1,p0,.05,power)[1] == n
        assert rough > 0
    for bad in [(.05,.05), (0,.1), (.1,1), (.1,.2,0), (.1,.2,.05,math.nan)]:
        rejects(sample, *bad)

    # Run only the declared editorial example, not the original GPU-serving APIs.
    node = rows['node'][1]
    loading = re.search(r'<!-- checked-example: serving-mmap-loading -->\n```python\n(.*?)^```', node, re.M|re.S)[1]
    load_ns = {}
    exec(compile(loading, 'serving-mmap-loading', 'exec'), load_ns)

    imported = (ROOT / '05 Источники/Courses/Harvard ML Systems/vol1/ml_ops.md').read_text()
    passages = [('harvard-retraining-schedule', text, imported),
                ('harvard-experiment-delayed-feedback', text, imported),
                ('pytorch-openmp-affinity', node, (SOURCE_DIR/'pytorch-tuning-guide.py').read_text()),
                ('pytorch-loading-mmap', node, (SOURCE_DIR/'pytorch-module-loading.py').read_text()),
                ('safetensors-zero-copy', node, (SOURCE_DIR/'safetensors-README.md').read_text())]
    for marker, chapter, source in passages:
        assert normalized(excerpt(chapter, marker)) in normalized(source), marker

    for source in json.loads((ROOT / 'docs/audit/2026-10-29-serving-decisions-sources.json').read_text()):
        assert digest((ROOT/source['file']).read_bytes()) == source['sha256']
    manifest = ROOT/'00 Учебник/Assets/Figures/curated/serving-decisions-2026-10-29/assets.json'
    for asset in json.loads(manifest.read_text()):
        data = (ROOT/asset['asset']).read_bytes()
        assert data == (ROOT/asset['source_asset']).read_bytes()
        assert digest(data) == asset['sha256']
        assert digest((ROOT/asset['license_evidence']).read_bytes()) == asset['license_evidence_sha256']
    print(json.dumps({'mlops_current_python_blocks':len(blocks),
        'retraining_independent_cases':300, 'ab_power_cases':300,
        'mmap_output_and_storage_check':True, 'torch':load_ns['torch'].__version__,
        'exact_source_passages':5, 'original_png_copies':1,
        'scope':'CPU formula/storage verification; no NUMA/GPU serving speed measurement'}, indent=2))


if __name__ == '__main__':
    main()
