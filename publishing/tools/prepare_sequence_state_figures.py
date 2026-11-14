"""Prepare complete original Table 1 and Figure 1, with captions, at 300 dpi.

Only source-page raster extraction: no upscaling, re-typesetting or redraw.
The downloaded PDF and versioned arXiv licence page stay in ignored Originals.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/sequence-state-2026-11-03/hybrid-linear-v1.pdf'
LICENCE_PAGE = SOURCE.with_suffix('.html')
DEST = ROOT / '00 Учебник/Assets/Figures/curated/sequence-state-2026-11-03'
MANIFEST = Path(__file__).with_name('sequence-state-figures.json')
SOURCE_SHA = 'bfca970ee2a85ed15b7e94adfd4e246366988ff1d927a861cfffa9cd591b23f7'
OBJECTS = [
    ('hybrid-state-update-table.png', 'Table 1', [283, 310, 2271, 1360]),
    ('hybrid-state-update-generations.png', 'Figure 1', [283, 1390, 2271, 2675]),
]
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(SOURCE.read_bytes()) == SOURCE_SHA
licence = LICENCE_PAGE.read_text()
assert 'creativecommons.org/licenses/by/4.0' in licence and '2507.06457v1' in licence
check = '--check' in sys.argv
rows = []
with tempfile.TemporaryDirectory(prefix='bookvar-state-source-') as tmp:
    for name, number, box in OBJECTS:
        prefix = str(Path(tmp) / name.removesuffix('.png'))
        subprocess.run(['pdftoppm', '-f', '4', '-l', '4', '-singlefile', '-r', '300',
                        '-x', str(box[0]), '-y', str(box[1]), '-W', str(box[2]-box[0]),
                        '-H', str(box[3]-box[1]), '-png', str(SOURCE), prefix], check=True)
        rendered = Path(prefix + '.png')
        with Image.open(rendered) as im:
            dimensions = list(im.size)
            pixels = im.convert('RGB').tobytes()
        rows.append(dict(asset=str((DEST/name).relative_to(ROOT)), page=4, object=number,
                         source=str(SOURCE.relative_to(ROOT)), source_sha256=SOURCE_SHA,
                         source_url='https://arxiv.org/pdf/2507.06457v1#page=4',
                         license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
                         license_evidence=str(LICENCE_PAGE.relative_to(ROOT)),
                         license_evidence_sha256=sha(LICENCE_PAGE.read_bytes()),
                         dpi=300, crop_pixels=box, dimensions=dimensions,
                         sha256=sha(rendered.read_bytes())))
        if check:
            with Image.open(DEST/name) as stored:
                assert list(stored.size) == dimensions
                assert stored.convert('RGB').tobytes() == pixels
        else:
            DEST.mkdir(parents=True, exist_ok=True)
            (DEST/name).write_bytes(rendered.read_bytes())
if check:
    assert json.loads(MANIFEST.read_text()) == rows
    print('PASS complete original Table 1/Figure 1, source hash, CC BY page, crop geometry and pixels')
else:
    MANIFEST.write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(rows, indent=2, ensure_ascii=False))
