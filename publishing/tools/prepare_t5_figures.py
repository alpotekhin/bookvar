#!/usr/bin/env python3
"""Regenerate original JMLR crops, or compare them without rewriting assets."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
parser.add_argument('--only', help='Process one named figure; other assets remain untouched.')
args = parser.parse_args()
specs = json.loads(Path(__file__).with_name('t5-figures.json').read_text())
assert len(specs) == 6
if args.only:
    specs = [s for s in specs if s['name'] == args.only]
    assert len(specs) == 1, args.only
for s in specs:
    for file_key, hash_key in [('source_asset', 'source_asset_sha256'),
                               ('source_page_snapshot', 'source_page_sha256'),
                               ('license_evidence', 'license_evidence_sha256')]:
        assert hashlib.sha256((ROOT / s[file_key]).read_bytes()).hexdigest() == s[hash_key]
    text = subprocess.check_output(['pdftotext', '-f', '1', '-l', '1',
                                   str(ROOT / s['source_asset']), '-'], text=True)
    assert 'License: CC-BY 4.0' in text
    assert s['license'] == 'CC BY 4.0'
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / 'page'
        subprocess.run(['pdftoppm', '-f', str(s['source_page']), '-l', str(s['source_page']),
                        '-r', str(s['render_dpi']), '-singlefile', '-png',
                        str(ROOT / s['source_asset']), str(prefix)], check=True)
        with Image.open(str(prefix) + '.png') as page:
            assert list(page.size) == s['render_page_size_px']
            result = page.crop(s['crop_box_px'])
            assert result.size == (s['width'], s['height'])
            if args.check:
                with Image.open(ROOT / s['asset']) as saved:
                    assert ImageChops.difference(result.convert('RGB'), saved.convert('RGB')).getbbox() is None
                assert hashlib.sha256((ROOT / s['asset']).read_bytes()).hexdigest() == s['sha256']
            else:
                result.save(ROOT / s['asset'])
    print('PASS unchanged pixels/source/CC BY 4.0', s['name'])
