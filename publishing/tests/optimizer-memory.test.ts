import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const path='00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md';
// Keep the historical contract on its accepted revision; the adaptive-clipping
// addition is separately checked, not normalized away from current acceptance.
const text=beforeTopicConsolidation(read(path).toString(),path,root,
  '2026-10-29-retention-additions.json');
const pattern=/<!-- BEGIN REVIEW: optimizer memory 2026-10-28 -->\n([\s\S]*?)<!-- END REVIEW: optimizer memory 2026-10-28 -->\n\n/g;
const addition=[...text.matchAll(pattern)][0]?.[1];

describe('memory-efficient optimizers in the existing narrative',()=>{
  it('preserves the entire accepted prior chapter outside the bounded addition and metadata',()=>{
    expect([...text.matchAll(pattern)]).toHaveLength(1);
    const prior=text.replace(pattern,'')
      .replace('last_updated: 2026-10-28','last_updated: 2026-10-21')
      .replace('source_language: mixed\n','');
    expect(hash(prior)).toBe('ffd865468db1e8ea07c64ec84f74a31cd065935f4b8aeb09a0f1f5b65670e43b');
    expect(text.match(/^```python$/gm)).toHaveLength(7);
    expect(addition.match(/^```python$/gm)).toHaveLength(3);
    expect(text).toContain('id="memory-efficient-optimizers"');
  });

  it('keeps the complete chosen English README paragraph and distinguishes the three mechanisms',()=>{
    const source=read('05 Источники/Originals/galore-2026-10-28/README-source.md').toString();
    const paragraph=source.split('\n\n').find(x=>x.startsWith('Gradient Low-Rank Projection'))!;
    const normalize=(s:string)=>s.replace(/^> ?/gm,'').replace(/\s+/g,' ').trim();
    expect(normalize(addition)).toContain(normalize(paragraph));
    for(const phrase of ['not the complete Adafactor optimizer','same trajectory as full Adam',
      'not measured peak GPU memory','immediate update on each microbatch',
      'quality experiment ran on one consumer GPU']) expect(normalize(addition)).toContain(phrase);
  });

  it('retains both original images with exact source blobs, attribution and license copies',()=>{
    const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/galore-2026-10-28/assets.json').toString());
    const registry=parse(read('05 Источники/asset-registry.yml').toString());
    expect(assets).toHaveLength(3);
    const pngs=assets.filter((a:any)=>a.asset.endsWith('.png'));
    expect(pngs).toHaveLength(2);
    const blobs=['68fd83ed0bf9582dc162567067c572c58e251149','8494c00e5f955e38903588cae0ee38bc279e4fa8'];
    for(const [i,a] of pngs.entries()){
      expect(registry.assets.find((x:any)=>x.id===a.id)).toEqual(a);
      for(const [p,sha] of [[a.asset,a.sha256],[a.source_asset,a.source_asset_sha256],
        [a.source_page_snapshot,a.source_page_sha256],[a.license_evidence,a.license_evidence_sha256]])
        expect(hash(read(p))).toBe(sha);
      expect(read(a.asset)).toEqual(read(a.source_asset));
      expect(read(a.public_license_copy)).toEqual(read(a.license_evidence));
      const b=read(a.asset);
      expect(createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`),b])).digest('hex')).toBe(blobs[i]);
      expect([b.readUInt32BE(16),b.readUInt32BE(20)]).toEqual(a.dimensions_px);
      expect(a.derivation).toBe('source-raster-copy');
      expect(a.crop_box_px).toBeUndefined();
    }
  });

  it('adds the original memory comparison without changing the accepted optimizer explanation',()=>{
    const figure=/<!-- BEGIN FIGURE: galore memory comparison 2026-10-28 -->\n[\s\S]*?<!-- END FIGURE: galore memory comparison 2026-10-28 -->\n\n/g;
    expect([...text.matchAll(figure)]).toHaveLength(1);
    expect(hash(text.replace(figure,''))).toBe('97370cdc6665f959eb59109cbe68cf2014e5247bfa1f21b2e68e9512b7a8c903');
    const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/galore-2026-10-28/assets.json').toString());
    const a=assets.find((a:any)=>a.asset.endsWith('/memory_breakdown.svg'));
    expect(parse(read('05 Источники/asset-registry.yml').toString()).assets.find((x:any)=>x.id===a.id)).toEqual(a);
    for(const [p,sha] of [[a.asset,a.sha256],[a.source_asset,a.source_asset_sha256],
      [a.source_page_snapshot,a.source_page_sha256],[a.license_evidence,a.license_evidence_sha256]])
      expect(hash(read(p))).toBe(sha);
    expect(a.sha256).toBe('3c395fe582f012f4f47780105693631a864aadf309a79b6a4ea6ed866d7f974f');
    expect(read(a.asset)).toEqual(read(a.source_asset));
    expect(read(a.asset).toString()).toContain('viewBox="0 0 587.04 304.08"');
    expect(a.license).toBe('CC-BY-4.0');
    expect(read(a.license_evidence).toString()).toMatch(/<a[^>]+creativecommons\.org\/licenses\/by\/4\.0\/[^>]+class="has_license"/);
    const source=read(a.source_page_snapshot).toString();
    expect(source).toContain('id="S1.F1"');
    for(const phrase of ['token batch size of 256','no retaining grad',
      'cannot be attributed to projection alone','not 256 sequences',
      'not the all-FP32 matrix ledger above']) expect(text.replace(/\s+/g,' ')).toContain(phrase);
  });
});
