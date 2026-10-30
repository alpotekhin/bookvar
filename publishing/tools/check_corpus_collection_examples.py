"""CPU-only checks for Bookvar chapter 41. No dataset/model benchmark."""

from pathlib import Path
from fractions import Fraction
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REL = '00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md'
chapter = (ROOT / REL).read_text()
before = subprocess.check_output(['git', 'show', f'HEAD:{REL}'], cwd=ROOT, text=True)

# Execute the actual displayed function, not a second implementation.
snippets = re.findall(r'```python\n(.*?)\n```', chapter, re.S)
assert len(snippets) == 2
for snippet in snippets:
    ast.parse(snippet)
mixture_snippet = next(s for s in snippets if 'def mixture_report' in s)
namespace = {}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(ast.parse(mixture_snippet), REL, 'exec'), namespace)
assert capture.getvalue().splitlines() == [
    'web 150.0 0.1875', 'code 90.0 1.125', 'science 60.0 3.0'
]
report = namespace['report']
assert math.isclose(sum(x['sampled_tokens'] for x in report.values()), 300e9)
check = namespace['mixture_report']
for available, shares, budget in [
    ({'x': 1}, {'y': 1}, 1), ({}, {}, 1),
    ({'x': 0}, {'x': 1}, 1), ({'x': 1}, {'x': 1}, 0),
    ({'x': 1}, {'x': -1}, 1), ({'x': 1}, {'x': 0.9}, 1),
]:
    try:
        check(available, shares, budget)
    except ValueError:
        pass
    else:
        raise AssertionError((available, shares, budget))
print('PASS: actual Markdown Python example and invalid-input cases')

selection = {}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(next(s for s in snippets if 'def gpt3_keep_probability' in s),
                 REL + ':probabilistic-filter', 'exec'), selection)
assert capture.getvalue().splitlines() == [
    '0.5 [True, False, False]', '0.8 [True, True, False]', '1.0 [True, True, True]'
]
retention = selection['gpt3_keep_probability']
for score, exact in ((0, Fraction(1, 512)), (0.5, Fraction(2, 3)**9),
                     (0.8, Fraction(5, 6)**9), (1, Fraction(1))):
    assert math.isclose(retention(score), float(exact), rel_tol=1e-14)
for score in (-0.1, 1.1, math.nan, math.inf):
    try:
        retention(score)
    except ValueError:
        pass
    else:
        raise AssertionError(score)
for alpha in (0, -1, math.inf, math.nan):
    try:
        retention(0.5, alpha)
    except ValueError:
        pass
    else:
        raise AssertionError(alpha)
assert round(1000 * retention(0.5)) == 26
assert round(1000 * retention(0.8)) == 194
# Enumerate a uniform grid and compare inverse-CDF Lomax to the Bernoulli form.
# This is a deterministic distribution identity, not a classifier evaluation.
for score in (0, 0.5, 0.8, 0.9, 1):
    outcomes = sum((1-u)**(-1/9)-1 > 1-score
                   for u in ((i+0.5)/10000 for i in range(10000)))
    assert abs(outcomes / 10000 - retention(score)) <= 0.0001
print('PASS: actual probability fence, exact Pareto tails, invalid inputs and inverse-CDF grid')

assert round(100 * 7.8 / 59, 1) == 13.2
assert round(100 * 6.3e6 / 237e9, 5) == 0.00266
archive = ROOT / '05 Источники/Originals/corpus-collection-2026-10-23'
source = (archive / 'extract_from_warc.py').read_text()
tree = ast.parse(source)
comparisons = {(node.left.id, type(node.ops[0]).__name__, node.comparators[0].value)
    for node in ast.walk(tree) if isinstance(node, ast.Compare)
    and isinstance(node.left, ast.Name) and len(node.ops) == 1
    and isinstance(node.comparators[0], ast.Constant)
    and isinstance(node.comparators[0].value, (float, int))}
assert {('score', 'Lt', 0.15), ('score', 'Lt', 0.8),
        ('perplexity', 'Gt', 30000)} <= comparisons
contains = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'contains_math')
fallback = next(n for n in contains.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'latex_match')
calls = lambda node: [n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'score_text']
assert len(calls(contains)) == len(calls(fallback)) == 1
paper_keep = lambda found, score, ppl: score > (0.17 if found else 0.8) and ppl <= 15000
code_keep = lambda found, score, ppl: score >= (0.15 if found else 0.8) and ppl <= 30000
assert paper_keep(True, 0.5, 10000) and not paper_keep(False, 0.5, 10000)
assert not paper_keep(True, 0.16, 20000) and code_keep(True, 0.16, 20000)
assert not paper_keep(True, 0.17, 15000) and code_keep(True, 0.17, 15000)
print('PASS: OpenWebMath stage counts, archived code predicates and documented recipe divergence')

readme = (archive / 'OpenWebMath-README.md').read_text()
original = readme.split('1. **Prefiltering HTML Documents**:', 1)[1].split(
    'For a detailed discussion', 1)[0]
quoted = chapter.split('> 1. **Prefiltering HTML Documents**:', 1)[1].split(
    'На схеме стадии', 1)[0]
def normalize_quote(value):
    value = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', value)
    value = value.replace('_', '').replace('*', '')
    return ' '.join(re.sub(r'^> ?', '', value, flags=re.M).split())
assert normalize_quote(original) == normalize_quote(quoted)
print('PASS: original English README steps preserved verbatim apart from link formatting')

sample_json = json.loads(re.search(r'```json\n(.*?)\n```', chapter, re.S).group(1))
assert sample_json['token_count'] == 420
assert sample_json['url'].startswith('https://example.org/')
print('PASS: illustrative JSON record parses')

assert round(100 * 78 / 90, 1) == 86.7
assert 100 * 78 / 80 == 97.5
assert round(100 * 70 / 72, 1) == 97.2
assert 100 * 70 / 80 == 87.5
assert Fraction(9 * 100, 9 * 100 + 9100) == Fraction(9, 100)
assert round(math.exp(3), 1) == 20.1
assert round(math.exp(4), 1) == 54.6
assert round(100 * 1.4 / 13.7, 1) == 10.2
assert round(4.42e9 / 97.6e6) == 45
assert 100e9 / 10e9 == 10
print('PASS: LID precision/recall, token-weighted fraction, perplexity, retention, epochs')

lengths = [100, 900]
def token_shares(q):
    total = sum(prob * length for prob, length in zip(q, lengths))
    return [prob * length / total for prob, length in zip(q, lengths)]
assert token_shares([Fraction(1, 2)] * 2) == [Fraction(1, 10), Fraction(9, 10)]
assert token_shares([Fraction(9, 10), Fraction(1, 10)]) == [Fraction(1, 2)] * 2
p = [Fraction(1, 2)] * 2
z = sum(pi / li for pi, li in zip(p, lengths))
assert [pi / li / z for pi, li in zip(p, lengths)] == [Fraction(9, 10), Fraction(1, 10)]
print('PASS: document-vs-token weighting and inverse conversion')

# UniMax-style water filling in token units. Original paper uses characters.
sizes = {'web': 800, 'code': 80, 'science': 20}
remaining, allocation = Fraction(300), {}
ordered = sorted(sizes, key=sizes.get)
cap = 2
for index, name in enumerate(ordered):
    share = min(remaining / (len(ordered) - index), cap * sizes[name])
    allocation[name] = share
    remaining -= share
assert remaining == 0
assert allocation == {'science': 40, 'code': 130, 'web': 130}
assert all(allocation[k] <= cap * sizes[k] for k in sizes)
assert [float(allocation[k] / sizes[k]) for k in ['web', 'code', 'science']] == [0.1625, 1.625, 2.0]
print('PASS: capped equal-budget allocation is feasible and matches the text')

small, large = Fraction(10_000_000_000), Fraction(1_000_000_000_000)
ratio = small / large
assert ratio == Fraction(1, 100)
for amount in [10_000_000_000_000, 10_000_000_000]:
    expected = Fraction(1, 2) * large / amount
    scaled = Fraction(1, 2) * small / (ratio * amount)
    assert expected == scaled
assert Fraction(1, 2) * large / 10_000_000_000 == 50
print('PASS: simulated-epoching equality and 50-pass example')

units = lambda s: set(re.findall(r'(?:- |source_unit_id: )((?:lecture|assignment)-[a-z0-9-]+)', s))
assert units(before) <= units(chapter)
old_ids = set(re.findall(r'<a id="([^"]+)"', before))
assert old_ids <= set(re.findall(r'<a id="([^"]+)"', chapter))
assert chapter.count('```') % 2 == 0
for target in re.findall(r'!\[\[(.*?)\]\]', chapter):
    path = target.split('|', 1)[0].removeprefix('02 Areas/ML & DL/')
    assert (ROOT / path).exists(), target
assert len(re.findall(r'!\[\[', chapter)) == 9
print('PASS: source units, explicit anchors, code fences, nine image paths')

# Independent filter rates overlap; they are not a sequential retention chain.
assert math.isclose(sum([6.0, 0.2, 2.7, 9.9, 1.1, 18.1, 9.0]), 47.0)
assert all(value in chapter for value in ['sum to 47%', 'remove 23.1%', '18.1% here'])
assert 23.1 < 47.0
release_assets = json.loads((ROOT / '00 Учебник/Assets/Figures/curated/corpus-releases-2026-10-29/assets.json').read_text())
assert len(release_assets) == 1
figure = release_assets[0]
for key, digest_key in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                        ('source_page_snapshot', 'source_page_sha256'),
                        ('license_evidence', 'license_evidence_sha256')]:
    assert hashlib.sha256((ROOT / figure[key]).read_bytes()).hexdigest() == figure[digest_key]
assert (ROOT / figure['asset']).read_bytes() == (ROOT / figure['source_asset']).read_bytes()
assert 'License: CC BY 4.0' in (ROOT / figure['license_evidence']).read_text()
assert figure['license'] == 'CC BY 4.0' and figure['dimensions_px'] == [1552, 279]
print('PASS: ROOTS overlapping-rate example, complete original figure bytes and license evidence')
print('Chapter SHA-256:', hashlib.sha256((ROOT / REL).read_bytes()).hexdigest())
print('LIMITS: no model trained; no full-corpus filtering run; publication/UI not checked here')
