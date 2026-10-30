#!/usr/bin/env python3
"""Verify the printed MIG example and original figures, not a GPU scheduler."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import subprocess
import tarfile
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
SOURCES = ROOT / '05 Источники/Originals/mig-2026-10-26'
ASSETS = ROOT / '00 Учебник/Assets/Figures/curated/mig-2026-10-26/assets.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(render=False):
    current_chapter = CHAPTER.read_text()
    reviewed_chapter = before_serving_operations(current_chapter, CHAPTER)
    # The fleet-alert extension independently restores the accepted MIG revision.
    chapter = re.sub(r'<!-- BEGIN REVIEW: mlops fleet-alerts 2026-10-26 -->\n'
                     r'.*?<!-- END REVIEW: mlops fleet-alerts 2026-10-26 -->\n\n',
                     '', reviewed_chapter, flags=re.S)
    found = re.search(r'<!-- BEGIN REVIEW: mlops mig-detail 2026-10-26 -->\n'
                      r'(.*?)<!-- END REVIEW: mlops mig-detail 2026-10-26 -->\n\n',
                      chapter, re.S)
    assert found
    before = chapter[:found.start()] + chapter[found.end():]
    assert sha(before.encode()) == '6b8633c35ce4c25317ca731feefc9e4811c7ee1caaf7e252624086f7626f6c20'
    blocks = re.findall(r'^```python\n(.*?)^```', found[1], re.M | re.S)
    assert len(blocks) == 1
    ns = {}
    exec(compile(blocks[0], str(CHAPTER) + ':relative-progress', 'exec'), ns)
    progress = ns['normalized_progress']
    rng = random.Random(20261026)
    for _ in range(300):
        p = [rng.randrange(1, 100) for _ in range(rng.randrange(1, 8))]
        q = [rng.randrange(200) for _ in p]
        # Count the normalized contributions in common integer baseline units.
        common = math.lcm(*p)
        expected = Fraction(sum(qi * (common // pi) for pi, qi in zip(p, q)), common)
        assert progress(p, q) == expected
        scales = [rng.randrange(1, 10) for _ in p]
        assert progress([pi * s for pi, s in zip(p, scales)],
                        [qi * s for qi, s in zip(q, scales)]) == expected
    for p, q in [([], []), ([1], []), ([0], [1]), ([-1], [1]),
                 ([True], [1]), ([1], [-1]), ([1], [True])]:
        try:
            progress(p, q)
        except ValueError:
            pass
        else:
            raise AssertionError((p, q))
    assert 2 * 3 < 7 and 2 * 4 == 8 and 2 * 4 + 1 > 8
    tex = (SOURCES / 'sections/background.tex').read_text()
    profiles = re.findall(r'^(\dg\.\d+gb) & (\d) GPC & (\d+) GB & (Full|\d/8)', tex, re.M)
    assert len(profiles) == 5
    for profile, compute, memory, cache in profiles:
        slices = 8 if cache == 'Full' else int(cache.split('/')[0])
        row = f'| `{profile}` | {compute} of 7 | {slices} of 8 | {memory} GB |'
        assert row in found[1], row
    assert '4g.20gb' in found[1] and 'removed starting with R510' in found[1]
    assert 'not an A100 slice-placement map' in found[1]

    original = (SOURCES / 'sections/motivation.tex').read_text()
    while re.search(r'\\(?:revision|texttt)\{[^{}]*\}', original):
        original = re.sub(r'\\(?:revision|texttt)\{([^{}]*)\}', r'\1', original)
    original = original.replace(r'\%', '%').replace('~', ' ')
    excerpt = re.search(r'<!-- source-excerpt: miso-configuration-dependent-performance -->\n'
                        r'(.*?)\n<!-- /source-excerpt -->', found[1], re.S)[1]
    excerpt = '\n'.join(re.sub(r'^> ?', '', line) for line in excerpt.splitlines())
    assert ' '.join(excerpt.split()) in ' '.join(original.split())

    for name, expected in {
        'nvidia-concepts.html': '61a05b2bede36a0cf30076b755c29d8613379c3ba84cd84e26747e461982467b',
        'nvidia-mps.html': '1878a3f987d89e5ae6f81ffd21c652a6867822e12634fa7962325368779a1c59',
        'sections/background.tex': '70b784dd252e0edaffb04820b9792b5e1e004c5acc7c15677780503aa40d435f',
    }.items():
        assert sha((SOURCES / name).read_bytes()) == expected
    assert 'http://creativecommons.org/licenses/by/4.0/' in (SOURCES / 'miso-license.html').read_text()
    assets = json.loads(ASSETS.read_text())
    assert len(assets) == 2
    with tarfile.open(SOURCES / 'miso-source.tar') as archive:
        for asset in assets:
            for path_key, hash_key in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                                       ('source_page_snapshot', 'source_page_sha256'),
                                       ('license_evidence', 'license_evidence_sha256'),
                                       ('source_archive', 'source_archive_sha256'),
                                       ('source_pdf', 'source_pdf_sha256')]:
                assert sha((ROOT / asset[path_key]).read_bytes()) == asset[hash_key]
            assert archive.extractfile(asset['source_member']).read() == (ROOT / asset['source_asset']).read_bytes()
            png = (ROOT / asset['asset']).read_bytes()
            assert png[:8] == b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II', png[16:24])) == asset['dimensions_px']
            assert asset['asset'] in chapter and asset['derivation'] == 'source-pdf-render'
            if render:
                output = subprocess.check_output(['pdftoppm', '-r', '400', '-png', '-singlefile',
                                                   str(ROOT / asset['source_asset'])])
                assert sha(output) == asset['sha256'], asset['asset']
    print(json.dumps({'chapter_sha256': sha(current_chapter.encode()),
                      'accepted_2080_lines_preserved': True,
                      'new_python_blocks_executed': 1, 'relative_progress_cases': 300,
                      'source_profile_rows': 5, 'original_figures': 2,
                      'verbatim_source_excerpts': 1,
                      'full_native_pdf_rerenders': 2 if render else 'not-run; use --render',
                      'scope': 'Teaching arithmetic and figure provenance; no MIG driver, GPU or scheduler experiment'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--render', action='store_true')
    main(args.parse_args().render)
