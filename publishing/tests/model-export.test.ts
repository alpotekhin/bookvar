import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const bytes=(p:string)=>readFileSync(resolve(root,p));
const sha=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const chapter='00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md';
const text=bytes(chapter).toString();
const marker=/<!-- BEGIN REVIEW: model export 2026-10-27 -->[\s\S]*?<!-- END REVIEW: model export 2026-10-27 -->\n\n/;

describe('model export teaching addition',()=>{
  it('preserves the entire accepted 832-line workflow outside the bounded addition and dates',()=>{
    const old=beforeReaderRouteNavigation(text, chapter).replace(/<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->[\s\S]*?<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->\n\n/,'')
      .replace(marker,'').replace('last_updated: 2026-10-27','last_updated: 2026-10-26')
      .replace('last_verified: 2026-10-27','last_verified: 2026-10-26');
    expect(sha(old)).toBe('329710d26fc09b696036c217d07f7151e9b0879394411dbc21e54e24d1911869');
    expect(text.match(marker)).toHaveLength(1);
    expect(text.match(marker)![0].match(/^```python$/gm)).toHaveLength(2);
    for(const item of ['checked_scores','onnx.checker.check_model','MissingKernel',
      'CPUExecutionProvider','external_data=False','rtol=1e-6','np.isfinite',
      'outside = np.ones((9, 2)','no implementation is registered']) expect(text).toContain(item);
  });
  it('pins every archived source to byte and Git-blob hashes',()=>{
    const manifest=JSON.parse(bytes('05 Источники/Originals/model-export-2026-10-27/source-manifest.json').toString());
    expect(manifest.files).toHaveLength(6);
    for(const file of manifest.files){
      const b=bytes(file.path);
      expect(sha(b)).toBe(file.sha256);
      expect(createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`),b])).digest('hex'))
        .toBe(file.git_blob_sha1);
      expect(file.source_url).toContain(`/blob/${file.commit}/`);
    }
  });
  it('retains the complete original figure and publicly served MIT license',()=>{
    const [asset]=JSON.parse(bytes('00 Учебник/Assets/Figures/curated/model-export-2026-10-27/assets.json').toString());
    expect(bytes(asset.asset)).toEqual(bytes(asset.source_asset));
    expect(bytes(asset.public_license_copy)).toEqual(bytes(asset.license_evidence));
    for(const[p,h]of[[asset.asset,asset.sha256],[asset.source_page_snapshot,asset.source_page_sha256],
      [asset.license_evidence,asset.license_evidence_sha256]]) expect(sha(bytes(p))).toBe(h);
    expect(asset.dimensions_px).toEqual([1454,910]);
    expect(asset.license).toBe('MIT');
    expect(text).toContain('/licenses/onnxruntime-e3d000e3-MIT.txt');
  });
});
