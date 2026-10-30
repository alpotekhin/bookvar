"""Verify original MLA figures and extract the complete author's MLA panel."""
from pathlib import Path
import hashlib, json, sys
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads(Path(__file__).with_name('mla-figures.json').read_text())
for row in rows:
    source, target = ROOT / row['source_asset'], ROOT / row['asset']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_asset_sha256']
    if 'crop_box_px' in row:
        with Image.open(source) as original:
            assert original.size == (1139, 918)
            expected = original.crop(row['crop_box_px'])
            if not target.exists() and '--check' not in sys.argv:
                target.parent.mkdir(parents=True, exist_ok=True)
                expected.save(target)
            with Image.open(target) as actual:
                assert actual.size == expected.size
                assert ImageChops.difference(actual.convert('RGB'), expected.convert('RGB')).getbbox() is None
    else:
        assert source.read_bytes() == target.read_bytes()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if row.get('sha256'):
        assert digest == row['sha256']
    print('PASS original pixels/bytes', target.name, digest)
