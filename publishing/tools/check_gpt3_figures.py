#!/usr/bin/env python3
"""Check archived source bytes, license evidence, GIF frames and lossless poster."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
specs = json.loads(Path(__file__).with_name('gpt3-figures.json').read_text())
assert len(specs) == 5
assert all(s['rights_status'] == 'source-license-verified' for s in specs)
for s in specs:
    for field, digest in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256')]:
        assert hashlib.sha256((ROOT / s[field]).read_bytes()).hexdigest() == s[digest]
    with Image.open(ROOT / s['asset']) as im:
        assert im.size == (s['width'], s['height'])
        assert getattr(im, 'n_frames', 1) == s['frames']
        for i in range(s['frames']):
            im.seek(i)
            im.load()
    if s['rights_status'] == 'source-license-verified':
        for field, digest in [('source_page_snapshot', 'source_page_sha256'),
                              ('license_evidence', 'license_evidence_sha256')]:
            assert hashlib.sha256((ROOT / s[field]).read_bytes()).hexdigest() == s[digest]
        page = (ROOT / s['source_page_snapshot']).read_text()
        assert s['source_image_url'].split('jalammar.github.io')[-1] in page
        assert 'creativecommons.org/licenses/by-nc-sa/4.0/' in page
        if s['derivation'] == 'source-gif-copy':
            assert (ROOT / s['asset']).read_bytes() == (ROOT / s['source_asset']).read_bytes()
        else:
            with Image.open(ROOT / s['source_asset']) as source, Image.open(ROOT / s['asset']) as target:
                source.seek(s['source_frame'])
                assert ImageChops.difference(source.convert('RGB'), target.convert('RGB')).getbbox() is None
    else:
        assert s['rights_status'] == 'paper-figure-license-unverified'
        assert s['metadata_status'] == 'needs-confirmation'
        assert s['source_page'] in (7, 8)
    print('PASS', s['name'], s['frames'], 'frames;', s['rights_status'])
print('PASS all five published GPT-3 visuals have verified source licenses.')

# Transcribe factual configuration data, not a raster copy of the paper table.
# Compare every numeric column with the pinned PDF, including reported anomalies.
import re
import subprocess
pdf = ROOT / '05 Источники/Originals/gpt3-2026-10-22/gpt3-v4.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest() == '97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83'
page = subprocess.run(['pdftotext', '-f', '8', '-l', '8', '-layout', str(pdf), '-'],
                      check=True, capture_output=True, text=True).stdout
source_rows = [re.split(r'\s{2,}', line.strip()) for line in page.splitlines()
               if re.match(r'^\s*GPT-3 (?:Small|Medium|Large|XL|2\.7B|6\.7B|13B|175B)', line)]
assert len(source_rows) == 8 and all(len(row) == 8 for row in source_rows)
for chapter in [
    '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md',
    'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md',
]:
    blocks = (ROOT / chapter).read_text().split('\n\n')
    table = next(block for block in blocks if block.startswith('| Model | Parameters |'))
    rows = [[cell.strip() for cell in line.split('|')[1:-1]] for line in table.splitlines()[2:]]
    assert len(rows) == 8
    for actual, expected in zip(rows, source_rows):
        assert actual[0] == expected[0].split(' or ')[0]
        assert actual[1:7] == expected[1:7], (actual, expected)
        source_rate = re.fullmatch(r'([\d.]+)\s*×\s*10[−-](\d+)', expected[7])
        markdown_rate = re.fullmatch(r'\$([\d.]+)\\times10\^\{(-?\d+)\}\$', actual[7])
        assert source_rate and markdown_rate
        assert float(source_rate[1]) == float(markdown_rate[1])
        assert -int(source_rate[2]) == int(markdown_rate[2])
    print('PASS eight configurations, all numeric columns, source Table 2.1:', chapter)
