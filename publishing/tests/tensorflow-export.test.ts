import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const sha=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const chapter='00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md';
const text=read(chapter).toString();
const marker=/<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->[\s\S]*?<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->\n\n/;

describe('TensorFlow export source integration',()=>{
  it('preserves the whole accepted 1088-line chapter and its ONNX examples',()=>{
    expect(sha(beforeReaderRouteNavigation(text, chapter).replace(marker,''))).toBe('8d6cc94470edc9ca00b9bc2d032d009bd51a34b130bfbb49356bea3b1af2f4d8');
    expect(text.match(marker)).toHaveLength(1);
    expect(text.match(marker)![0].match(/^```python$/gm)).toHaveLength(11);
    for(const fact of ['CustomModule','mutate','serving_default','checked_scale',
      '4.98186','BUILTIN_WITHOUT_DEFAULT_DELEGATES','flattened signature callable',
      'format="tf_saved_model"']) expect(text).toContain(fact);
  });
  it('pins source files, retaining failed download status instead of treating a prefix as a complete page',()=>{
    const manifest=JSON.parse(read('05 Источники/Originals/tensorflow-export-2026-10-27/source-manifest.json').toString());
    expect(manifest.files).toHaveLength(8);
    expect(manifest.files.filter((x:any)=>x.status==='incomplete-download-not-used')).toHaveLength(2);
    const converter=manifest.files.find((x:any)=>x.path.endsWith('/tflite-converter.html'));
    expect(converter.previous_recorded_sha256)
      .toBe('c61ed0af26077efd745df272b7df17a484259a9a090aa1b6db10ef0104fba6cc');
    expect(converter.snapshot_reconciliation).toContain('Archive bytes were not changed');
    expect(read(converter.path).toString().trimEnd().endsWith('</html>')).toBe(true);
    for(const file of manifest.files){
      const b=read(file.path); expect(sha(b)).toBe(file.sha256);
      if(file.commit) expect(createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`),b])).digest('hex'))
        .toBe(file.git_blob_sha1);
    }
  });
  it('retains the original diagram, source language and public license',()=>{
    const[asset]=JSON.parse(read('00 Учебник/Assets/Figures/curated/tensorflow-export-2026-10-27/assets.json').toString());
    expect(read(asset.asset)).toEqual(read(asset.source_asset));
    expect(read(asset.public_license_copy)).toEqual(read(asset.license_evidence));
    expect(asset.dimensions_px).toEqual([826,1003]);
    expect(asset.license).toBe('Apache-2.0');
    expect(text).toContain('Some node names are abbreviated in the original');
    expect(text).toContain('If no Python code is saved, how does SavedModel know how to restore the function?');
    expect(text).toContain('<!-- BEGIN SOURCE: Harvard framework exit path -->');
  });
});
