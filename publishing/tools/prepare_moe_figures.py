"""Reproduce source figures, preserving every panel, axis, label and legend."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).with_name('moe-figures.json')
ORIGINALS=ROOT/'05 Источники/Originals/moe-2026-10-23'
TARGETS=ROOT/'00 Учебник/Assets/Figures/curated/moe-2026-10-23'
PIN='357454f4f647385839c0ff6b99a688dc7cd9c13f'
source_hashes={
 'switch-jmlr':'8831edca732dd939854957b4b1a493fa553e7b43dd431a5d4fee15493bba6e3e',
 'olmoe-shared':'0842577d406a5a9551d976aabb6241840cfdd881c612e5eaf3b0d281cc9bd698',
 'olmoe-granularity':'f28b7b9a3d0f63ebbdec93de6c23d8cd3904e68576be3f9080fd62947266a6a7',
 'olmoe-lbl':'c1b7675721cb579800f1581af5a938a008221e706984ad7bcf4a01e43d41ed11',
 'olmoe-lbltoks':'0a2acfeb61e049cda3b70cb3796a464235df6629308a368a0c803dbd355ae044',
 'olmoe-zloss':'d067fc92fa3bebf8fa55d50e1a4b987548f65e7837406be1e043fa56b754338c',
}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for spec in json.loads(HERE.read_text()):
    name=Path(spec['asset']).stem; switch=name.startswith('switch-')
    source=ORIGINALS/(('switch-jmlr' if switch else name)+'.pdf')
    target=TARGETS/(name+'.png')
    assert digest(source)==source_hashes['switch-jmlr' if switch else name]
    with tempfile.TemporaryDirectory(prefix='bookvar-moe-') as temp:
        prefix=Path(temp)/'page'
        subprocess.run(['pdftoppm','-f',str(spec['source_page']),'-l',str(spec['source_page']),
            '-singlefile','-r',str(spec['render_dpi']),'-png',str(source),str(prefix)],check=True)
        with Image.open(prefix.with_suffix('.png')) as original:
            size=list(original.size)
            expected=original.convert('RGB')
            if 'crop_box_px' in spec: expected=expected.crop(spec['crop_box_px'])
            if not target.exists() and '--check' not in sys.argv:
                target.parent.mkdir(parents=True,exist_ok=True)
                expected.save(target)
            with Image.open(target) as actual:
                assert actual.size==expected.size
                assert ImageChops.difference(actual.convert('RGB'),expected).getbbox() is None
    if switch:
        evidence=source
        text=subprocess.check_output(['pdftotext','-f','1','-l','1',str(source),'-'],text=True)
        assert 'License: CC-BY 4.0' in text
        url='https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page='+str(spec['source_page'])
        license_name='CC BY 4.0'; license_url='https://creativecommons.org/licenses/by/4.0/'
        author='William Fedus, Barret Zoph, Noam Shazeer'
    else:
        evidence=ORIGINALS/'OLMoE-LICENSE.txt'
        assert digest(evidence)=='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
        assert 'Apache License' in evidence.read_text()
        url=f'https://github.com/allenai/OLMoE/blob/{PIN}/visuals/figures/'+name.removeprefix('olmoe-')+'.pdf'
        license_name='Apache-2.0'; license_url=f'https://github.com/allenai/OLMoE/blob/{PIN}/LICENSE'
        author='Niklas Muennighoff et al. / Allen Institute for AI'
    row={
      'asset':str(target.relative_to(ROOT)), 'author':author,
      'source_asset':str(source.relative_to(ROOT)), 'source_asset_sha256':digest(source),
      'sha256':digest(target), 'source_url':url,'source_image_url':url,
      'source_page':spec['source_page'],'render_dpi':spec['render_dpi'],'render_page_size_px':size,
      'license':license_name,'license_url':license_url,
      'license_evidence':str(evidence.relative_to(ROOT)),'license_evidence_sha256':digest(evidence),
      'derivation':'pdf-crop' if switch else 'pdf-page-render',
      'metadata_status':'verified','provenance_confirmation':'source-license-verified',
      'modifications':'Rendered original PDF; all panels, axes, legends and English labels retained'+
        ('; full English caption retained' if switch else '; no crop or redraw'),
    }
    if switch: row['crop_box_px']=spec['crop_box_px']
    else: row['commit']=PIN
    assert row==spec, (name,'source metadata or rendered bytes changed')
    rows.append(row)
if '--json' in sys.argv: print(json.dumps(rows,ensure_ascii=False,indent=2))
else:
    for row in rows: print('PASS original pixels/license',Path(row['asset']).name,row['sha256'])
