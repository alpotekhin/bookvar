import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe, expect, it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const packet='05 Источники/Originals/data-selection-reconciliation-2026-10-28/';
const chapterE='00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md';
const chapterI='00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества.md';

describe('complete Data Selection generated-figure disposition',()=>{
  it('covers all 14 pinned source blocks and distinguishes the two code-only decisions',()=>{
    const inventory=JSON.parse(read(packet+'generated-figure-decisions.json'));
    const source=read(inventory.source_file),lines=source.split('\n');
    expect(hash(source)).toBe(inventory.source_sha256);
    const sourceIds=[...source.matchAll(/^::: \{#(fig-[\w-]+)/gm)].map(m=>m[1]);
    expect(inventory.figures.map((r:any)=>r.id)).toEqual(sourceIds);
    expect(sourceIds).toHaveLength(14);
    expect(inventory.figures.filter((r:any)=>r.web_found)).toHaveLength(12);
    for(const row of inventory.figures){
      expect(hash(lines.slice(row.source_start-1,row.source_end).join('\n'))).toBe(row.source_span_sha256);
      expect(['keep','link-only','omit']).toContain(row.decision);
      expect(row.rationale.length).toBeGreaterThan(60);
      if(row.web_found){
        const bytes=readFileSync(resolve(root,row.local));
        expect(hash(bytes)).toBe(row.sha256);
        expect(bytes.length).toBe(row.bytes);
        expect(read(inventory.web_snapshot)).toContain(row.url.split('/').at(-1));
      }else{
        expect(row.visual_review).toContain('source-code-only');
        expect(row.decision).not.toBe('keep');
      }
    }
  });
  it('historical explanation preservation: adds figures and the ICR clarification without changing the accepted explanations around them',()=>{
    const e=readBeforeReaderRouteNavigation(chapterE),i=readBeforeReaderRouteNavigation(chapterI);
    expect(e.match(/^!\[\[/gm)).toHaveLength(4);
    expect(i.match(/^!\[\[/gm)).toHaveLength(5);
    expect(hash(e.replace(/<a id="pretraining-amortization-figure"><\/a>\n[\s\S]*?(?=The same accounting applies to a reusable pretrained model\.)/,'')))
      .toBe('ed4b3f14cf5d30dabf36f8de74e1c784f05d815ee62fa6a3c27ae779f4c26189');
    const restored=i
      .replace(/<a id="selection-artifact-flow"><\/a>\n[\s\S]*?(?=Removing half the corpus at fixed epochs can reduce)/,'')
      .replace(/<a id="selection-icr-proxy"><\/a>\n[\s\S]*?(?=### Coverage can disagree with average quality)/,'');
    expect(hash(restored)).toBe('b4ecbaa30f6367ae38d9a4e0cdfa0d89c8b17957073be3fed3a9b9d583a09009');
    expect(i.replace(/\s+/g,' ')).toContain('not bits per FLOP');
    expect(e.replace(/\s+/g,' ')).toContain('10,500 with pretraining');
    expect(i.replace(/\s+/g,' ')).toContain('artifact dependencies, not a guaranteed product of speedups');
  });
  it('keeps both original web figures and their actual distinct NC-ND license',()=>{
    const prefix='00 Учебник/Assets/Figures/curated/harvard-data-selection-2026-10-28/';
    const rows=JSON.parse(read(prefix+'assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(2);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      const bytes=readFileSync(resolve(root,row.asset));
      expect(bytes).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect(hash(bytes)).toBe(row.sha256);
      expect(hash(readFileSync(resolve(root,row.source_page_snapshot)))).toBe(row.source_page_sha256);
      expect(hash(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
      expect(row.license).toBe('CC BY-NC-ND 4.0');
      expect(read(row.used_in[0])).toContain(row.asset);
    }
  });
});
