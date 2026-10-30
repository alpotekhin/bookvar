"""Render complete source figures with captions; retain labels and all panels."""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads(Path(__file__).with_name('long-context-figures.json').read_text())
for row in rows:
    source, target = ROOT / row['source_asset'], ROOT / row['asset']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_asset_sha256']
    with tempfile.TemporaryDirectory(prefix='bookvar-long-') as folder:
        prefix = Path(folder) / 'page'
        subprocess.run(['pdftoppm', '-f', str(row['source_page']), '-l', str(row['source_page']),
            '-r', str(row['render_dpi']), '-singlefile', '-png', str(source), str(prefix)], check=True)
        with Image.open(prefix.with_suffix('.png')) as page:
            assert list(page.size) == row['render_page_size_px']
            expected = page.convert('RGB').crop(row['crop_box_px'])
            if not target.exists() and '--check' not in sys.argv:
                target.parent.mkdir(parents=True, exist_ok=True)
                expected.save(target)
            with Image.open(target) as actual:
                assert actual.size == expected.size
                assert ImageChops.difference(actual.convert('RGB'), expected).getbbox() is None
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if row.get('sha256'):
        assert digest == row['sha256']
    evidence = ROOT / row['license_evidence']
    if row.get('license_evidence_sha256'):
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == row['license_evidence_sha256']
    assert 'creativecommons.org/licenses/by/4.0' in evidence.read_text()
    print('PASS original pixels', target.name, digest)
    print('PASS license evidence', evidence.name, hashlib.sha256(evidence.read_bytes()).hexdigest())

# Retained originals: compare fresh source downloads and exact old frame crops.
originals = ROOT / '05 Источники/Originals/long-context-2026-10-23'
previous = ROOT / '00 Учебник/Assets/Figures/curated/long-context-review-2026-09'
for name in ('position-interpolation.png', 'yarn-method-summary.png',
             'mistral-sliding-window.png', 'mistral-rolling-buffer.png'):
    assert (originals / name).read_bytes() == (previous / name).read_bytes()
    print('PASS retained original bytes', name)
for i, box in enumerate(((375,0,900,322), (945,0,1475,322), (1520,0,2039,322))):
    with Image.open(originals / 'mistral-rolling-buffer.png') as source, \
         Image.open(previous / f'mistral-rolling-step-{i}.png') as actual:
        assert ImageChops.difference(source.crop(box).convert('RGB'),
                                     actual.convert('RGB')).getbbox() is None
    print('PASS retained original frame', i)
course = ROOT / '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images/cla-diagram.png'
assert course.read_bytes() == (ROOT / '00 Учебник/Assets/Figures/curated/course-follow-through/cla-diagram.png').read_bytes()
print('PASS retained CLA course image bytes')
