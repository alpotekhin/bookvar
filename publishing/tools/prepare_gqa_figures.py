"""Extract full original GQA plots/captions and check all reused illustrations."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[2]
rows=json.loads(Path(__file__).with_name('gqa-figures.json').read_text())
for row in rows:
    source,target=ROOT/row['source_asset'],ROOT/row['asset']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_asset_sha256']
    if 'source_page' in row:
        with tempfile.TemporaryDirectory() as folder:
            prefix=Path(folder)/'page'
            subprocess.run(['pdftoppm','-f',str(row['source_page']),'-l',str(row['source_page']),
                '-r',str(row['render_dpi']),'-singlefile','-png',str(source),str(prefix)],check=True)
            with Image.open(prefix.with_suffix('.png')) as page:
                assert list(page.size)==row['render_page_size_px']
                expected=page.convert('RGB').crop(row['crop_box_px'])
                if not target.exists() and '--check' not in sys.argv:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    expected.save(target)
                with Image.open(target) as actual:
                    assert actual.size==expected.size
                    assert ImageChops.difference(actual.convert('RGB'),expected).getbbox() is None
    else:
        assert target.read_bytes()==source.read_bytes()
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    if row.get('sha256'):
        assert digest==row['sha256']
    print('PASS original pixels/bytes',target.name,digest)
print('PASS nine source illustrations, including four complete plots/tables with original captions')
