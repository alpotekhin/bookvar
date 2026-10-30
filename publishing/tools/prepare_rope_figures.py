"""Render or verify full source figures; never redraw labels or edit source content."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
specs = json.loads(Path(__file__).with_name('rope-figures.json').read_text())
for row in specs:
    source, target = ROOT / row['source_asset'], ROOT / row['asset']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_asset_sha256']
    if 'source_page' not in row:
        assert target.read_bytes() == source.read_bytes()
    else:
        with tempfile.TemporaryDirectory(prefix='bookvar-rope-') as folder:
            prefix = Path(folder) / 'page'
            subprocess.run(['pdftoppm', '-f', str(row['source_page']), '-l', str(row['source_page']),
                            '-r', str(row['render_dpi']), '-singlefile', '-png', str(source), str(prefix)], check=True)
            with Image.open(prefix.with_suffix('.png')) as page:
                assert list(page.size) == row['render_page_size_px']
                expected = page.crop(row['crop_box_px']).convert('RGB')
                if '--check' not in sys.argv:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    expected.save(target)
                with Image.open(target) as actual:
                    assert expected.size == actual.size
                    assert ImageChops.difference(expected, actual.convert('RGB')).getbbox() is None
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if row.get('sha256'):
        assert digest == row['sha256']
    print('PASS original pixels/bytes', target.name, digest)
print('PASS five source figures; no synthetic diagrams')
