import {createHash} from 'node:crypto';
import {readFileSync,readdirSync} from 'node:fs';
import {resolve,basename} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {withoutNativeWindowExample} from './helpers/nn-native-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
const report=JSON.parse(read('docs/audit/2026-10-27-nn-native-figures.json').toString());
const chapter='00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md';

describe('individual Harvard native-figure disposition',()=>{
  it('accounts for every immutable referenced native object without altering the archive',()=>{
    expect(hash(read(report.source))).toBe(report.source_sha256);
    expect(hash(read(report.source_code))).toBe(report.source_code_sha256);
    const source=read(report.source).toString();
    const references=Array.from(source.matchAll(/!\[\[(Assets\/Sources\/Harvard ML Systems\/vol1\/nn_architectures\/[^\]|]+)(?:\|[^\]]*)?\]\]/g),m=>m[1]);
    expect(references).toHaveLength(24);
    expect(report.figures.map((row:{asset:string})=>row.asset)).toEqual(references);
    expect(readdirSync(resolve(root,'Assets/Sources/Harvard ML Systems/vol1/nn_architectures')).sort())
      .toEqual(references.map(path=>basename(path)).sort());
    for(const [i,row] of report.figures.entries()) {
      expect(row.number).toBe(i+1);
      expect(hash(read(row.asset))).toBe(row.sha256);
      expect(row.finding.length).toBeGreaterThan(80);
      expect(source.split('\n')[row.source_line-1]).toContain(row.asset);
    }
  });

  it('distinguishes source retention from the two thematic uses',()=>{
    const uses=report.figures.filter((row:{teaching_asset:string|null})=>row.teaching_asset);
    expect(uses.map((row:{number:number})=>row.number)).toEqual([21,24]);
    for(const row of uses) expect(read(row.teaching_asset)).toEqual(read(row.asset));
    expect(report.figures[15].decision).toBe('archive-cross-attention-wiring-error');
    expect(report.figures[20].decision).toBe('integrated-corrected-caption');
    expect(report.figures[23].decision).toBe('retained-existing-practice');
  });

  it('retains the whole accepted CNN revision outside the bounded worked example',()=>{
    const text=read(chapter).toString();
    expect(hash(withoutNativeWindowExample(text)))
      .toBe('d2fe8eaa5ceda18400dbbfae92d1fdcb13a61c07a35964adae0ef76cda65ee6e');
    expect(text.match(/^```python$/gm)).toHaveLength(21);
    expect(text).toContain('This explanatory caption corrects');
    expect(text).toContain('harvard_columns[0].T, harvard_rows');
    expect(text).toContain('[356., 392., 464., 500.]');
  });

  it('keeps the exact source SVG, source-code pin and adjacent rights evidence',()=>{
    const local=JSON.parse(read('00 Учебник/Assets/Figures/curated/nn-native-2026-10-27/assets.json').toString());
    expect(local).toHaveLength(1);
    const entry=local[0];
    expect(parse(read('05 Источники/asset-registry.yml').toString()).assets
      .find((row:{id:string})=>row.id===entry.id)).toEqual(entry);
    for(const [path,sha] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]]) expect(hash(read(path))).toBe(sha);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.crop_box_px).toBeUndefined();
  });
});
