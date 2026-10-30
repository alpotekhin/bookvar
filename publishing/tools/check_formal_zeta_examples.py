#!/usr/bin/env python3
"""Verify the added finite examples and native source images, not an LLM run."""
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from math import prod
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md'
ASSETS = ROOT / '00 Учебник/Assets/Figures/curated/formal-zeta-2026-10-26'
ARCHIVE = ROOT / '05 Источники/Originals/formal-zeta-2026-10-26'
START = '<!-- BEGIN REVIEW: Berkeley finite products 2026-10-26 -->'
END = '<!-- END REVIEW: Berkeley finite products 2026-10-26 -->'
BASE_SHA = '6a6912ffa8cf76d54452572734970180a3b2bb1e472055fd87d28da573e6bf50'
SOURCE_SHA = '18ae67ac1cf4af8b95293b338da3339554c3f6ea470564516982b2eba4af81b3'
PIN = 'c44e0c8ee63ca166450922a373c7409c5d26b00b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def main():
    text = CHAPTER.read_text('utf-8')
    assert text.count(START) == text.count(END) == 1
    addition = text.split(START, 1)[1].split(END, 1)[0]
    baseline = text.replace(START + addition + END + '\n\n', '')
    baseline = baseline.replace('last_updated: 2026-10-26', 'last_updated: 2026-10-21')
    baseline = baseline.replace('last_verified: 2026-10-26', 'last_verified: 2026-10-21')
    baseline = baseline.replace('  - meeting-08-slides-riemann-zeta-demo\n', '')
    for url in (f'https://github.com/leanprover-community/mathlib4/blob/{PIN}/Mathlib/NumberTheory/EulerProduct/Basic.lean',
                'https://terrytao.wordpress.com/2021/02/12/246b-notes-4-the-riemann-zeta-function-and-the-prime-number-theorem/'):
        baseline = baseline.replace('  - ' + url + '\n', '')
    baseline = baseline.replace('физические страницы 5–107; использованы изображения 30, 79, 84 и 91, а также исходные JPEG с изображениями 99–100.',
                                'физические страницы 5–92; использованы изображения 30, 79, 84 и 91.')
    baseline = re.sub(r'^- \[Mathlib: Euler products\].*\n', '', baseline, flags=re.M)
    assert sha(baseline.encode()) == BASE_SHA, 'the previously accepted chapter changed outside the addition'
    print('PASS: prior chapter, five positive Lean blocks, seven Coq blocks and 31 figures preserved')

    code = re.findall(r'^```python\n(.*?)^```\s*$', addition, re.M | re.S)
    assert len(code) == 2
    for index, source in enumerate(code):
        exec(compile(source, f'{CHAPTER.name}:new-python-{index}', 'exec'), {})
    # Independent arithmetic and multiplicity checks, beyond the displayed case.
    for n in range(1, 401):
        assert sum(k**3 for k in range(1, n+1)) == (n*(n+1)//2)**2
    cases = 0
    for primes in ((2, 3), (2, 3, 5), (3, 7)):
        for cutoff in range(5):
            denominators = [prod(p**e for p, e in zip(primes, powers))
                            for powers in product(range(cutoff+1), repeat=len(primes))]
            assert len(denominators) == len(set(denominators))
            for exponent in (1, 2, 3):
                lhs = prod(sum((F(1, p**(exponent*e)) for e in range(cutoff+1)), F(0))
                           for p in primes)
                rhs = sum((F(1, d**exponent) for d in denominators), F(0))
                assert lhs == rhs
                cases += 1
    multiplicities = Counter(2**a * 4**b for a in range(3) for b in range(3))
    assert multiplicities[4] == multiplicities[16] == 2
    assert sum((F(count, d) for d, count in multiplicities.items()), F(0)) != sum(
        (F(1, d) for d in multiplicities), F(0))
    assert 1/(1-F(1, 4))/(1-F(1, 9)) == F(3, 2)
    assert 100-17 == 83 and 83 % 332 != 0 and 4*83 % 332 == 0
    for r in range(1, 13):
        assert sum((F(1, n) for n in range(1, 2**r+1)), F(0)) >= 1+F(r, 2)
    print(f'PASS: both actual Python blocks, 400 cube sums and {cases} finite rational products')

    original = ARCHIVE / 'EulerProduct-Basic.lean'
    assert sha(original.read_bytes()) == SOURCE_SHA
    source = original.read_text('utf-8')
    intro = source.split('# Euler Products\n\n', 1)[1].split('\n## Tags', 1)[0]
    quote = addition.split('<!-- BEGIN SOURCE: Mathlib Euler Products introduction -->', 1)[1]
    quote = quote.split('<!-- END SOURCE: Mathlib Euler Products introduction -->', 1)[0]
    quote = re.sub(r'^> ?', '', quote, flags=re.M)
    assert normalized(quote) == normalized(intro), 'original English module explanation changed'
    assert 'Released under Apache 2.0 license' in source
    license_path = ARCHIVE / 'mathlib-LICENSE.txt'
    assert sha(license_path.read_bytes()) == 'b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1'
    assert (ROOT / 'site/public/licenses/mathlib-c44e0c8-Apache-2.0.txt').read_bytes() == license_path.read_bytes()
    print('PASS: complete original English Mathlib introduction and exact license copy')

    rows = json.loads((ASSETS / 'assets.json').read_text('utf-8'))
    assert len(rows) == 2
    pdf = ROOT / rows[0]['source_asset']
    assert sha(pdf.read_bytes()) == rows[0]['source_asset_sha256']
    with tempfile.TemporaryDirectory(prefix='bookvar-native-zeta-') as temp:
        prefix = Path(temp) / 'native'
        subprocess.run(['pdfimages', '-f', '99', '-l', '100', '-j', str(pdf), str(prefix)],
                       check=True, capture_output=True, timeout=30)
        for index, row in enumerate(rows):
            asset = ROOT / row['asset']
            assert asset.read_bytes() == Path(f'{prefix}-{index:03d}.jpg').read_bytes()
            assert sha(asset.read_bytes()) == row['sha256']
            assert list(Image.open(asset).size) == row['dimensions_px']
            assert row['derivation'] == 'pdf-embedded-raster-copy'
            assert row['asset'] in text
            assert row['license'] == 'User permission record (not a license)'
    print('PASS: two native PDF JPEG streams copied byte-for-byte, with source dimensions')
    print('Chapter SHA-256:', sha(CHAPTER.read_bytes()))
    print('LIMITS: no new Lean/Coq or Mathlib build, no AlphaProof trace or model benchmark')


if __name__ == '__main__':
    main()
