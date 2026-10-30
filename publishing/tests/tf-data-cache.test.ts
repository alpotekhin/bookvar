import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const hash=(value:Buffer|string)=>createHash('sha256').update(value).digest('hex');
const chapter='00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md';
const directory='05 Источники/Originals/tf-data-2026-10-28/';
const text=read(chapter).toString();
const pattern=/<!-- BEGIN REVIEW: tf.data cache 2026-10-28 -->\n([\s\S]*?)<!-- END REVIEW: tf.data cache 2026-10-28 -->\n\n/g;
const normalize=(s:string)=>s.replace(/^> ?/gm,'').replace(/\s+/g,' ').trim();

describe('tf.data cache explanation and source preservation',()=>{
  it('preserves the prior 727-line chapter apart from review dates and one equation wrap',()=>{
    expect([...text.matchAll(pattern)]).toHaveLength(1);
    let prior=text.replace(pattern,'');
    for(const key of ['last_updated','last_verified'])
      prior=prior.replace(`${key}: 2026-10-28`,`${key}: 2026-10-23`);
    const wrapped=String.raw`\begin{aligned}
&128^2+140^2+160^2+1024^2\\
&=1\,110\,160,
\end{aligned}`;
    expect(text).toContain(wrapped);
    prior=prior.replace(wrapped,String.raw`128^2+140^2+160^2+1024^2=1\,110\,160,`);
    expect(hash(prior)).toBe('4ec7b2a6d0665e0c0c6a43337e508f4d7cb41c30009d1b850850c6559d1a25aa');
    expect(text.match(/^```python$/gm)).toHaveLength(9);
    const addition=[...text.matchAll(pattern)][0][1];
    expect(addition.match(/^```python$/gm)).toHaveLength(5);
    expect(text).toContain('id="cache-placement-and-epoch-state"');
  });

  it('retains the complete selected original English paragraphs, not a translation',()=>{
    const notebook=JSON.parse(read(directory+'performance.ipynb').toString());
    const normalized=normalize(text);
    const intro=notebook.cells[36].source.join('').split('\n\n').at(-1);
    expect(normalized).toContain(normalize(intro));
    for(const paragraph of notebook.cells[38].source.join('').split('\n\n').slice(1))
      expect(normalized).toContain(normalize(paragraph));
    expect(text).toContain('The TensorFlow Authors');
    expect(text).toContain('not a GPU throughput measurement');
    expect(text).toContain('does not guarantee that every two');
  });

  it('binds the unchanged original vector and its license to the public registry',()=>{
    const [asset]=JSON.parse(read('00 Учебник/Assets/Figures/curated/tf-data-2026-10-28/assets.json').toString());
    const registry=parse(read('05 Источники/asset-registry.yml').toString());
    expect(registry.assets.find((row:any)=>row.id===asset.id)).toEqual(asset);
    for(const [path,sha] of [[asset.asset,asset.sha256],
      [asset.source_asset,asset.source_asset_sha256],
      [asset.source_page_snapshot,asset.source_page_sha256],
      [asset.license_evidence,asset.license_evidence_sha256]]) expect(hash(read(path))).toBe(sha);
    expect(read(asset.asset)).toEqual(read(asset.source_asset));
    expect(read(asset.public_license_copy)).toEqual(read(asset.license_evidence));
    const bytes=read(asset.source_asset);
    expect(createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${bytes.length}\0`),bytes])).digest('hex'))
      .toBe('5b1aae592ab786846d5c19a5f179e57ac024c7d9');
    expect(asset.derivation).toBe('source-svg-copy');
    expect(asset.crop_box_px).toBeUndefined();
    expect(asset.source_viewbox).toEqual([0,0,1224,360]);
    const css=read('site/src/styles/custom.css').toString();
    expect(css).toContain("a:has(> img[src$='/tf-data-2026-10-28/cached-dataset.svg'])");
    expect(css).toContain('width: 76.5rem;');
    expect(asset.modifications).toContain('horizontally scrollable');
  });

  it('keeps CPU semantic checks separate from the existing packing checker and speed claims',()=>{
    const checker=read('publishing/tools/check_tf_data_cache_examples.py').toString();
    for(const mechanism of ['actual cache fence','range(8)','range(2, 7)',
      'range(5)','get_visible_devices("GPU")']) {
      expect(checker).toContain(mechanism);
    }
    expect(checker).toContain('no timing benchmark');
    expect(read('publishing/tools/check_data_pipeline_examples.py').toString())
      .toContain('The independent tf.data addition');
    expect(hash(read(directory+'dataset-ops.py')))
      .toBe('981ea5a001b08c66e5ba9341b4bd58f0896628f56cf2abdb1d5bb73c637c25cc');
  });
});
