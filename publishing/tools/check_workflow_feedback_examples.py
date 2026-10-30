"""Execute the new visible workflow examples, not training or a monitoring service."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md'
SOURCE = ROOT / '05 Источники/Originals/ml-workflow-2026-10-20/harvard-ml-workflow.qmd'


def main():
    text = CHAPTER.read_text()
    sections = re.findall(r'<!-- BEGIN REVIEW: workflow (?:iteration|feedback) 2026-10-26 -->'
                          r'(.*?)<!-- END REVIEW: workflow (?:iteration|feedback) 2026-10-26 -->', text, re.S)
    assert len(sections) == 2
    blocks = re.findall(r'^```python\n(.*?)^```', '\n'.join(sections), re.M | re.S)
    assert len(blocks) == 3
    ns = {'__name__': '__workflow_feedback_examples__'}
    for block in blocks:
        exec(compile(block, str(CHAPTER), 'exec'), ns)
    assert ns['large_iters'] == 26
    assert ns['large_final_acc'] == 98.9 and ns['small_final_acc'] == 99
    source = SOURCE.read_text()
    excerpt = source.split('    weeks_total = 26', 1)[1].split('    # ┌── 3. GUARD', 1)[0]
    excerpt = ('    weeks_total = 26' + excerpt)
    excerpt = re.sub(r'^    ', '', excerpt, flags=re.M).strip()
    assert blocks[0].strip() == excerpt
    paragraph = source.split('### Multi-scale feedback', 1)[1].split('\n\n')[1]
    paragraph = re.sub(r'\\index\{[^}]*\}', '', paragraph)
    quote = re.search(r'<!-- BEGIN SOURCE: Harvard multi-scale feedback -->\n> (.*?)\n<!-- END SOURCE:', text, re.S)[1]
    assert quote == paragraph

    rng = random.Random(20261026)
    budget = ns['iteration_budget']
    quality = ns['mature_quality']
    for _ in range(300):
        cycle_ticks, horizon_ticks = rng.randrange(1, 673), rng.randrange(5000)
        cap = rng.randrange(300)
        start, gain, ceiling = rng.randrange(90), Fraction(rng.randrange(30), 100), 99
        expected_n = min(horizon_ticks // cycle_ticks, cap)
        result = budget(start, float(gain), cycle_ticks / 4, horizon_ticks / 4, cap, ceiling)
        assert result[0] == expected_n
        assert math.isclose(result[1], float(min(Fraction(start) + expected_n * gain, ceiling)))
        assert result == budget(start, float(gain), cycle_ticks * 2, horizon_ticks * 2, cap, ceiling)
    for args in [(90, .1, 0, 10, 10), (90, .1, 1, -1, 10),
                 (90, -.1, 1, 10, 10), (90, .1, 1, 10, 1.5),
                 (101, .1, 1, 10, 10), (90, .1, 1, float('nan'), 10)]:
        try:
            budget(*args)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'invalid budget accepted: {args}')
    for _ in range(300):
        observations = []
        for _ in range(30):
            predicted = rng.randrange(200)
            label = None if rng.randrange(4) == 0 else predicted + rng.randrange(100)
            observations.append({'predicted_at': predicted, 'label_at': label,
                                 'correct': None if label is None else bool(rng.randrange(2))})
        now, horizon = rng.randrange(300), rng.randrange(100)
        mature = usable = errors = 0
        for r in observations:
            if now - r['predicted_at'] < horizon:
                continue
            mature += 1
            if r['label_at'] is None or r['label_at'] > now or r['correct'] is None:
                continue
            usable += 1
            errors += int(r['correct'] is False)
        expected = {'mature': mature, 'usable': usable,
                    'coverage': usable / mature if mature else None,
                    'error_ratio': errors / usable if usable else None}
        assert quality(observations, now, horizon) == expected
        shifted = [{**r, 'predicted_at': r['predicted_at'] + 100,
                    'label_at': None if r['label_at'] is None else r['label_at'] + 100}
                   for r in observations]
        assert quality(shifted, now + 100, horizon) == expected
        rng.shuffle(observations)
        assert quality(observations, now, horizon) == expected
    assert quality([], 72, 72) == {'mature': 0, 'usable': 0, 'coverage': None, 'error_ratio': None}
    print(json.dumps({'chapter_sha256': hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      'executed_new_blocks': 3, 'source_parameters': 'exact dedented copy',
                      'source_paragraph': 'exact copy without index markup',
                      'independent_budget_cases': 300, 'independent_cohort_cases': 300,
                      'scope': 'deterministic teaching arithmetic; no trained model, clinical evaluation, live monitor or Marimo runtime'}, indent=2))


if __name__ == '__main__':
    main()
