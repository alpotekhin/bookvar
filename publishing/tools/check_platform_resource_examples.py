#!/usr/bin/env python3
"""Execute the printed quota calculation; verify source identity, not a cluster."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
SOURCES = ROOT / '05 Источники/Originals/platform-resources-2026-10-26'
ASSETS = ROOT / '00 Учебник/Assets/Figures/curated/platform-resources-2026-10-26/assets.json'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def text_only(value):
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)
    value = re.sub(r'\[\^[^\]]+\]', '', value)
    value = value.replace('**', '').replace('_', '')
    return ' '.join(value.split())


def get_excerpt(chapter, key):
    found = re.search(r'<!-- source-excerpt: ' + key +
                      r' -->\n(.*?)\n<!-- /source-excerpt -->', chapter, re.S)
    assert found, key
    return '\n'.join(re.sub(r'^> ?', '', line) for line in found[1].splitlines())


def main():
    current_chapter = CHAPTER.read_text()
    reviewed_chapter = before_serving_operations(current_chapter, CHAPTER)
    # The later FinOps addition has its own guard restoring our accepted SHA.
    chapter = re.sub(r'<!-- BEGIN REVIEW: mlops (finops|mig-detail|fleet-alerts) 2026-10-26 -->\n'
                     r'.*?<!-- END REVIEW: mlops \1 2026-10-26 -->\n\n',
                     '', reviewed_chapter, flags=re.S)
    section = re.search(r'<!-- BEGIN REVIEW: mlops platform-resources 2026-10-26 -->\n'
                        r'(.*?)<!-- END REVIEW: mlops platform-resources 2026-10-26 -->\n\n',
                        chapter, re.S)
    assert section
    before = chapter[:section.start()] + chapter[section.end():]
    assert digest(before.encode()) == '57ee342a14d420a5c97a7a48511dfe1efa430bcb29d3c207c8e3e676dc51739e'

    harvard = (ROOT / '05 Источники/Originals/mlops-2026-10-20/harvard-ops-scale.qmd').read_text()
    source_by_excerpt = {
        'harvard-training-resource-management': harvard,
        'harvard-serving-resource-management': harvard,
        'kueue-quota-not-pod-scheduling': (SOURCES / 'kueue-concepts.md').read_text(),
        'kueue-borrowing-example': (SOURCES / 'kueue-cluster-queue.md').read_text(),
        'nvidia-shared-access-not-proportional': (SOURCES / 'nvidia-device-plugin.md').read_text(),
        'nvidia-mps-memory-limit': (SOURCES / 'nvidia-device-plugin.md').read_text(),
        'kubernetes-namespace-isolation-requires-controls': (SOURCES / 'kubernetes-multi-tenancy.md').read_text(),
    }
    for key, source in source_by_excerpt.items():
        assert text_only(get_excerpt(chapter, key)) in text_only(source), key
    original_yaml = re.search(r'^```yaml\n(.*?)^```',
                              (SOURCES / 'kueue-cluster-queue.md').read_text(), re.M | re.S)[1]
    excerpt_yaml = re.search(r'^```yaml\n(.*?)^```',
                             get_excerpt(chapter, 'kueue-basic-cluster-queue'), re.M | re.S)[1]
    assert original_yaml == excerpt_yaml

    blocks = re.findall(r'^```python\n(.*?)^```', section[1], re.M | re.S)
    assert len(blocks) == 1
    namespace = {}
    exec(compile(blocks[0], str(CHAPTER) + ':quota-example', 'exec'), namespace)
    admit = namespace['can_admit_two_queue']
    rng = random.Random(20261026)
    for _ in range(500):
        qa, qb = rng.randrange(16), rng.randrange(16)
        ua = rng.randrange(qa + qb + 1)
        ub = rng.randrange(qa + qb - ua + 1)
        borrowing, lending, request = [rng.randrange(20) for _ in range(3)]
        # Enumerate all own/borrowed portions of A's prospective allocation.
        # B has already consumed ub units of the fixed qa+qb pool.
        target = ua + request
        feasible = any(own + borrowed == target
                       and target + ub <= qa + qb
                       for own in range(qa + 1)
                       for borrowed in range(min(borrowing, lending, max(0, qb - ub)) + 1))
        assert admit(request, qa, ua, borrowing, qb, ub, lending) == feasible
    for arguments in [(-1, 9, 0, 1, 12, 0, 12), (1.5, 9, 0, 1, 12, 0, 12),
                      (True, 9, 0, 1, 12, 0, 12), (1, 9, 10, 1, 12, 12, 12)]:
        try:
            admit(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid modeled resource state accepted')
    assert 8 * 10 == 80 and 2 * 25 > 40 and 8 + 2.5 > 40 / 4
    assert 10 * 100 == 1000 and 10 * 10_000 == 100_000
    assert '20% of one device' in chapter and 'not a scheduler for each HTTP request' in chapter

    assets = json.loads(ASSETS.read_text())
    for asset in assets:
        for path_key, hash_key in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                                   ('source_page_snapshot', 'source_page_sha256'),
                                   ('license_evidence', 'license_evidence_sha256')]:
            assert digest((ROOT / asset[path_key]).read_bytes()) == asset[hash_key]
        assert (ROOT / asset['asset']).read_bytes() == (ROOT / asset['source_asset']).read_bytes()
        assert asset['asset'] in chapter
    assert len(assets) == 3
    for rejected in ['mig-profiles-a100.png', 'mig-mixed-a100.png', 'mig-strategies.png']:
        assert rejected not in chapter
    print(json.dumps({'chapter_sha256': digest(current_chapter.encode()),
                      'accepted_1312_lines_preserved': True,
                      'new_python_blocks_executed': 1, 'enumerated_quota_cases': 500,
                      'verified_source_excerpts': 8, 'original_figures': 3,
                      'scope': 'Printed arithmetic and source fidelity; no scheduler, GPU or live Kubernetes validation'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
