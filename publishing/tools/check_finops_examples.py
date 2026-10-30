#!/usr/bin/env python3
"""Check printed FinOps arithmetic and source provenance, not real savings."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import struct
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
MANIFEST = ROOT / '00 Учебник/Assets/Figures/curated/finops-2026-10-26/assets.json'
SOURCE = ROOT / '05 Источники/Originals/mlops-2026-10-20/harvard-ops-scale.qmd'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    value = re.sub(r'\\index\{[^}]*\}', '', value)
    return ' '.join(value.replace('*', '').split())


def main():
    current_chapter = CHAPTER.read_text()
    chapter = before_serving_operations(current_chapter, CHAPTER)
    section = re.search(r'<!-- BEGIN REVIEW: mlops finops 2026-10-26 -->\n'
                        r'(.*?)<!-- END REVIEW: mlops finops 2026-10-26 -->\n\n',
                        chapter, re.S)
    assert section
    previous = chapter[:section.start()] + chapter[section.end():]
    # Later MIG detail has its own guard restoring this FinOps revision.
    previous = re.sub(r'<!-- BEGIN REVIEW: mlops (mig-detail|fleet-alerts) 2026-10-26 -->\n'
                      r'.*?<!-- END REVIEW: mlops \1 2026-10-26 -->\n\n',
                      '', previous, flags=re.S)
    assert sha(previous.encode()) == '0f76754dfcd1c28d515724662fac537ad978f7acfa47f0138a3b8cd63aa5f251'
    source = SOURCE.read_text()
    for key in ['harvard-tco-definition', 'harvard-cost-attribution']:
        excerpt = re.search(r'<!-- source-excerpt: ' + key + r' -->\n(.*?)\n<!-- /source-excerpt -->',
                            chapter, re.S)[1]
        excerpt = '\n'.join(re.sub(r'^> ?', '', line) for line in excerpt.splitlines())
        assert normalized(excerpt) in normalized(source), key

    blocks = re.findall(r'^```python\n(.*?)^```', section[1], re.M | re.S)
    assert len(blocks) == 4
    ns = {}
    for i, block in enumerate(blocks):
        # Each new printed block is independently executable.
        own = {}
        exec(compile(block, str(CHAPTER) + f':finops-{i}', 'exec'), own)
        ns.update(own)
    rng = random.Random(20261026)
    for _ in range(300):
        total = rng.randrange(100_000)
        weights = [rng.randrange(20) for _ in range(rng.randrange(1, 8))]
        if not sum(weights):
            weights[0] = 1
        allocation = ns['allocate_shared'](total, weights)
        assert sum(allocation) == total
        for charge, weight in zip(allocation, weights):
            assert charge * sum(weights) == total * weight
        results = rng.randrange(100_000)
        unit = ns['cost_per_result'](total, results)
        assert unit is None if results == 0 else unit * results == total
        gross, extra = rng.randrange(100_000), rng.randrange(100_000)
        months = ns['payback_months'](total, gross, extra)
        assert months is None if gross <= extra else months * (gross - extra) == total
    for function, arguments in [
        ('allocate_shared', (10, [])), ('allocate_shared', (10, [0, 0])),
        ('allocate_shared', (10, [-1, 2])), ('allocate_shared', (10, [True, 1])),
        ('allocate_shared', (-1, [1])), ('cost_per_result', (10, -1)),
        ('cost_per_result', (10, True)), ('payback_months', (100, -1, 0)),
        ('payback_months', (100, 10, True)),
    ]:
        try:
            ns[function](*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError((function, arguments))

    # Recompute the source scenarios, rather than accepting formatted numbers.
    source_scenarios = {
        's_training': 5000, 's_inference': 2000, 's_data': 500, 's_iteration': 40000,
        'p_training': 150000, 'p_inference': 400000, 'p_data': 80000, 'p_iteration': 350000,
    }
    for name, value in source_scenarios.items():
        found = re.search(r'^    ' + name + r' = ([\d_]+)$', source, re.M)
        assert found and int(found[1]) == value
    assert sum(source_scenarios[k] for k in source_scenarios if k.startswith('s_')) == 47500
    assert sum(source_scenarios[k] for k in source_scenarios if k.startswith('p_')) == 980000
    assert round(40000 / 47500 * 100, 1) == 84.2
    assert round(400000 / 980000 * 100, 1) == 40.8
    assert 4 * 1 / 8 == .5 and 100 * 1.2 * .90 == 108
    assert [500 + 100, 5000 + 500, 50000 + 2000] == [600, 5500, 52000]
    assert Fraction(1, 1) / Fraction(4, 5) == Fraction(5, 4)
    assert 'intersection directly' in chapter

    asset, = json.loads(MANIFEST.read_text())
    for path_key, hash_key in [('asset', 'sha256'), ('source_code', 'source_code_sha256'),
                               ('source_page_snapshot', 'source_page_sha256'),
                               ('license_evidence', 'license_evidence_sha256')]:
        assert sha((ROOT / asset[path_key]).read_bytes()) == asset[hash_key]
    cell_section = source.split('::: {#fig-tco-iceberg ', 1)[1].split('\n:::', 1)[0]
    cell = re.search(r'^```\{python\}\n(.*?)^```', cell_section, re.M | re.S)[1]
    assert sha(cell.encode()) == asset['source_member_sha256']
    image = (ROOT / asset['asset']).read_bytes()
    assert image[:8] == b'\x89PNG\r\n\x1a\n'
    assert list(struct.unpack('>II', image[16:24])) == asset['dimensions_px'] == [3099, 2130]
    assert asset['asset'] in chapter and asset['derivation'] == 'source-code-render'
    print(json.dumps({'chapter_sha256': sha(current_chapter.encode()),
                      'accepted_1656_lines_preserved': True,
                      'new_python_blocks_executed': 4, 'generated_accounting_cases': 900,
                      'verbatim_source_excerpts': 2, 'source_figure_cells_verified': 1,
                      'scope': 'Printed accounting scenarios and source fidelity; no real prices, invoices or savings measured'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
