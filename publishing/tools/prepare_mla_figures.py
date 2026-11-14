"""Verify original MLA figures and extract the complete author's MLA panel."""
from pathlib import Path
import argparse, hashlib, json, subprocess
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
parser.add_argument('--deepseek-v2-pdf', type=Path, help='Verify the three current crops against the pinned author PDF.')
args = parser.parse_args()

# Retain the checks for the historical CS336 copies, even after replacing the embeds.
rows = json.loads(Path(__file__).with_name('mla-figures.json').read_text())
for row in rows:
    source, target = ROOT / row['source_asset'], ROOT / row['asset']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_asset_sha256']
    if 'crop_box_px' in row:
        with Image.open(source) as original:
            assert original.size == (1139, 918)
            expected = original.crop(row['crop_box_px'])
            if not target.exists() and not args.check:
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

if args.deepseek_v2_pdf:
    rows = json.loads(Path(__file__).with_name('deepseek-v2-mla-originals.json').read_text())
    pdf = args.deepseek_v2_pdf
    source_digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    for row in rows:
        assert source_digest == row['source_asset_sha256']
        page = str(row['source_page'])
        x1, y1, x2, y2 = row['crop_box_px']
        raster = subprocess.check_output([
            'pdftoppm', '-hide-annotations', '-f', page, '-l', page,
            '-r', str(row['render_dpi']), '-x', str(x1), '-y', str(y1),
            '-W', str(x2 - x1), '-H', str(y2 - y1), '-png', '-singlefile', str(pdf),
        ])
        asset = ROOT / row['asset']
        assert hashlib.sha256(raster).hexdigest() == row['sha256']
        assert raster == asset.read_bytes()
        with Image.open(asset) as image:
            assert list(image.size) == row['dimensions_px']
        print('PASS original author PDF crop', page, asset.name, row['sha256'])
