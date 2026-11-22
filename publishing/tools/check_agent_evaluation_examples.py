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
ENGLISH = ROOT / 'en/00 Textbook/17 Tools and Agents/68 Evaluating agent systems.md'
ARCHIVE = ROOT / '05 Источники/Originals/agent-evaluation-2026-10-20'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_examples(path, text):
    panels = re.findall(
        r'^<details id="agent-eval-code-([^"]+)">\n<summary>[^\n]+</summary>\n\n'
        r'~~~python\n(.*?)^~~~\n\n</details>', text, re.M | re.S)
    assert [panel for panel, _ in panels] == ['booking', 'pass-k', 'paired', 'rubric']
    blocks = [block for _, block in panels]
    scopes, shared = [], {}
    for panel, block in panels:
        code = compile(block, f'{path}:example-{panel}', 'exec')
        scope = {}
        exec(code, scope)
        exec(code, shared)
        scopes.append(scope)
    return blocks, scopes


def main():
    text, english = CHAPTER.read_text(), ENGLISH.read_text()
    blocks, scopes = execute_examples(CHAPTER, text)
    en_blocks, _ = execute_examples(ENGLISH, english)
    assert en_blocks == blocks
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    figures = lambda source: re.findall(r'^!\[\[([^|\]]+)(?:\|[^\]]*)?\]\]', source, re.M)
    assert len(figures(text)) == 9 and figures(english) == figures(text)
    def display_math(source):
        return re.findall(r'^\$\$\n(.*?)^\$\$$', source, re.M | re.S)
    assert len(display_math(text)) == 3
    assert [formula.replace('0{,}05', '0.05') for formula in display_math(text)] == display_math(english)
    booking, metric, paired, rubric = scopes
    grade = booking['grade_booking']
    old, new, other, invite, audit = (
        booking[name] for name in ('old', 'new', 'other', 'invite', 'audit'))
    assert old[2] == '2026-10-08T18:00:00+03:00'
    assert new[2] == '2026-10-09T19:00:00+03:00'
    assert invite[2:] == ('204', new[2])
    start, after, sent = [old, other], [other, new], [invite]
    saved = deepcopy((start, after, sent, audit))
    correct = "correct", {"calendar": True, "mail": True, "policy": True}
    assert grade(start, after, sent, audit) == correct
    assert (start, after, sent, audit) == saved
    assert grade(start, list(reversed(after)), sent, [*audit, audit[0]]) == correct
    wrong = (new[0], new[1], "2026-10-09T18:00:00+03:00", new[3])
    for calendar in ([], [other, wrong], [new], [*after, old], [*after, other]):
        assert grade(start, calendar, sent, audit) == (
            "incorrect", {"calendar": False, "mail": True, "policy": True})
    assert grade(start, after, [invite, invite], None) == (
        "incorrect", {"calendar": True, "mail": False, "policy": None})
    denied = (*booking['mail_send'], False)
    assert grade(start, after, sent, [*audit, denied]) == (
        "incorrect", {"calendar": True, "mail": True, "policy": False})
    assert grade(start, after, sent, [*audit, ("mail-send", "other@example.org", True)]) == (
        "incorrect", {"calendar": True, "mail": True, "policy": False})
    for partial_audit in (None, [], [audit[0]]):
        assert grade(start, after, sent, partial_audit) == (
            "unscored", {"calendar": True, "mail": True, "policy": None})
    assert grade(start, after, None, audit) == (
        "unscored", {"calendar": True, "mail": None, "policy": True})
    assert grade(start, None, sent, audit) == (
        "unscored", {"calendar": None, "mail": True, "policy": True})
    try:
        grade([*start, old], after, sent, audit)
    except ValueError as error:
        assert str(error) == "invalid starting snapshot"
    else:
        raise AssertionError("duplicate starting booking was accepted")
    assert 'Полноту журнала проверяет оболочка оценивания' in text

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
                all_success = sum(comb(c, k) / comb(n, k) * comb(n, c) * p**c * (1-p)**(n-c)
                                  for c in range(n+1))
                assert isclose(all_success, p**k, abs_tol=1e-14)
    assert np.allclose(estimate([10, 10], [2, 0], 3), [8/15, 0])
    assert isclose(1-(1-0.75)**3, 0.984375)
    assert isclose(0.75**3, 0.421875)
    assert 1 - Fraction(comb(8, 3), comb(10, 3)) == Fraction(8, 15)
    assert 1 - (1 - Fraction(2, 10))**3 == Fraction(488, 1000)
    tasks = [Fraction(1), Fraction(0)]
    assert sum(tasks)/2 == sum(1-(1-p)**3 for p in tasks)/2 == sum(p**3 for p in tasks)/2 == Fraction(1, 2)
    assert 1 - (1 - sum(tasks)/2)**3 == Fraction(7, 8)
    assert (sum(tasks)/2)**3 == Fraction(1, 8)
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
    print(f'PASS: 4 examples per locale individually/shared (16 executions); {tested} finite metric cases; '
          'unbiased expectations; booking failures; 256 paired resamples; rubric/judge/risk arithmetic; '
          'source function unchanged; source hashes and 5 exact original crops. No LLM/agent benchmark run.')


if __name__ == '__main__':
    main()
