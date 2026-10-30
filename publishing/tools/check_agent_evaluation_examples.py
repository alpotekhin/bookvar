#!/usr/bin/env python3
"""Check chapter 68 arithmetic, fixtures and source crops; no agent/LLM run."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import comb, isclose
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/17 Tools и Agents/68 Оценивание агентных систем.md'
ARCHIVE = ROOT / '05 Источники/Originals/agent-evaluation-2026-10-20'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r'^~~~python\n(.*?)^~~~', text, re.M | re.S)
    assert len(blocks) == 4
    scopes, shared = [], {}
    for index, block in enumerate(blocks):
        code = compile(block, f'{CHAPTER}:example-{index+1}', 'exec')
        scope = {}
        exec(code, scope)
        exec(code, shared)
        scopes.append(scope)
    booking, metric, paired, rubric = scopes
    grade = booking['grade_booking']
    state = {'bookings': [deepcopy(booking['booking'])]}
    trace = deepcopy(booking['allowed'])
    before = deepcopy((state, trace))
    assert grade(state, trace) == {'outcome': True, 'policy': True, 'success': True}
    assert (state, trace) == before
    assert not grade({'bookings': []}, trace)['outcome']
    assert not grade({'bookings': state['bookings'] * 2}, trace)['outcome']
    wrong = deepcopy(state)
    wrong['bookings'][0]['start'] = '2026-10-09T18:00:00+03:00'
    assert not grade(wrong, trace)['outcome']
    assert grade(state, [{'kind': 'write', 'authorized': False}]) == {
        'outcome': True, 'policy': False, 'success': False}
    # Deliberate limitations described in prose: complete logs and other events
    # are not checked by this tiny fixture, and no real calendar is contacted.
    assert grade(state, [])['success']
    extra = deepcopy(state)
    extra['bookings'].append({'seminar_id': 'other-event'})
    assert grade(extra, trace)['success']
    assert 'полноту журнала' in text

    source = (ARCHIVE / 'human-eval-evaluation.py').read_text()
    original = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'estimate_pass_at_k')
    copied = next(node for node in ast.parse(blocks[1]).body
                  if isinstance(node, ast.FunctionDef) and node.name == original.name)
    assert ast.dump(original) == ast.dump(copied)
    assert ast.get_source_segment(source, original) == ast.get_source_segment(blocks[1], copied)
    estimate = metric['estimate_pass_at_k']
    tested = 0
    for n in range(1, 13):
        for c in range(n+1):
            for k in range(1, n+1):
                expected = 1 - Fraction(comb(n-c, k), comb(n, k)) if n-c >= k else Fraction(1)
                assert isclose(estimate(n, [c], k)[0], float(expected), abs_tol=1e-14)
                # Check the all-success formula by enumerating finite subsets.
                observed = sum(all(j < c for j in indices)
                               for indices in combinations(range(n), k))
                assert Fraction(observed, comb(n, k)) == Fraction(comb(c, k), comb(n, k))
                tested += 1
    for n in range(1, 8):
        for k in range(1, n+1):
            for p in [0, 0.2, 0.75, 1]:
                mean = sum(estimate(n, [c], k)[0] * comb(n, c) * p**c * (1-p)**(n-c)
                           for c in range(n+1))
                assert isclose(mean, 1-(1-p)**k, abs_tol=1e-14)
    assert np.allclose(estimate([10, 10], [2, 0], 3), [8/15, 0])
    assert isclose(1-(1-0.75)**3, 0.984375)
    assert isclose(0.75**3, 0.421875)
    assert np.array_equal(paired['difference'], [0, -1, 1, 1])
    assert len(paired['resampled_means']) == 256
    assert np.allclose(paired['interval'], [-0.5, 1])
    assert [rubric[key] for key in ['score_075', 'score_025', 'score_050', 'score_040', 'replication_score']] == [
        Fraction(3, 4), Fraction(1, 4), Fraction(1, 2), Fraction(2, 5), Fraction(11, 20)]
    leaves = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
    assert Fraction(sum(leaves), len(leaves)) != rubric['replication_score']
    assert isclose((18+24)/50, 0.84) and isclose(18/24, 0.75) and isclose(18/20, 0.9)
    upper = lambda n: 1-0.05**(1/n)
    assert isclose(upper(100), 0.029513049607039932)
    assert upper(298) > 0.01 and upper(299) <= 0.01
    assert isclose(70/98, 5/7) and isclose(70/100, 0.7)

    refs = json.loads((ARCHIVE / 'references.json').read_text())
    preceding = before_topic_consolidation(text, CHAPTER)
    assert refs['chapter_sha256'] == hashlib.sha256(preceding.encode()).hexdigest()
    for row in refs['downloaded'] + refs['derived'] + refs['existing_sources']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    for figure in refs['figures']:
        for field, digest in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                              ('source_page_snapshot', 'source_page_sha256'),
                              ('license_evidence', 'license_evidence_sha256')]:
            assert sha(ROOT / figure[field]) == figure[digest]
        original_page = Image.open(ROOT / figure['source_page_snapshot']).convert('RGB')
        crop = original_page.crop(figure['crop_box_px'])
        published = Image.open(ROOT / figure['asset']).convert('RGB')
        assert crop.size == published.size == (figure['width'], figure['height'])
        assert ImageChops.difference(crop, published).getbbox() is None
        assert 'href="http://creativecommons.org/licenses/by/4.0/"' in (
            ROOT / figure['license_evidence']).read_text()
    excerpt = refs['code_excerpt']
    assert sha(ROOT / excerpt['published_license']) == excerpt['published_license_sha256']
    assert (ROOT / excerpt['published_license']).read_text().strip() == (
        ARCHIVE / 'human-eval-LICENSE').read_text().strip()
    print(f'PASS: 4 examples individually/shared (8 executions); {tested} finite metric cases; '
          'unbiased expectations; booking failures; 256 paired resamples; rubric/judge/risk arithmetic; '
          'source function unchanged; source hashes and 5 exact original crops. No LLM/agent benchmark run.')


if __name__ == '__main__':
    main()
