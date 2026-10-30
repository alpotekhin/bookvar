"""Render complete Mamba-2 Figures 5/6 with captions; verify deterministic pixels."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/sequence-alternatives-2026-10-23/mamba2-v1.pdf'
DEST = ROOT / '00 Учебник/Assets/Figures/curated/sequence-alternatives-2026-10-23'
MANIFEST = Path(__file__).with_name('sequence-figures.json')
# Full panel and full English caption, not the surrounding article text.
BOX = [283, 267, 2367, 1417]  # pixels on a 300 dpi render of a 612 x 792 pt page
OBJECTS = [(20, 5, 'mamba2-ssd-algorithm.png'), (23, 6, 'mamba2-block-complete.png')]
check = '--check' in sys.argv
rows = []
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='bookvar-sequence-figures-') as tmp:
    for page, figure, name in OBJECTS:
        prefix = str(Path(tmp) / name.removesuffix('.png'))
        subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
                        '-r', '300', '-x', str(BOX[0]), '-y', str(BOX[1]),
                        '-W', str(BOX[2]-BOX[0]), '-H', str(BOX[3]-BOX[1]),
                        '-png', str(SOURCE), prefix], check=True)
        rendered = Path(prefix + '.png')
        with Image.open(rendered) as im:
            size = list(im.size)
            pixels = im.convert('RGB').tobytes()
        row = dict(asset=str((DEST/name).relative_to(ROOT)), page=page, figure=figure,
                   source=str(SOURCE.relative_to(ROOT)), source_sha256=source_hash,
                   source_url='https://arxiv.org/pdf/2405.21060v1#page='+str(page),
                   license='CC BY 4.0', license_url='https://arxiv.org/html/2405.21060v1',
                   dpi=300, crop_pixels=BOX, dimensions=size,
                   sha256=hashlib.sha256(rendered.read_bytes()).hexdigest())
        if check:
            with Image.open(DEST/name) as old:
                assert list(old.size) == size and old.convert('RGB').tobytes() == pixels
        else:
            DEST.mkdir(parents=True, exist_ok=True)
            (DEST/name).write_bytes(rendered.read_bytes())
        rows.append(row)
if check:
    assert json.loads(MANIFEST.read_text()) == rows
    print('PASS two complete Mamba-2 source figures, PDF hash, geometry and pixels')
else:
    MANIFEST.write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(rows, indent=2, ensure_ascii=False))
