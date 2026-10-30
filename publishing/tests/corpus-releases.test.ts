import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(v:Buffer|string)=>createHash('sha256').update(v).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md';
const packet='05 Источники/Originals/corpus-releases-2026-10-29/';

describe('corpus release scopes in the existing data chapter',()=>{
  it('pins primary sources and distinguishes raw, annotated and selected data',()=>{
    const manifest=JSON.parse(read(packet+'manifest.json').toString());
    expect(manifest.sources).toHaveLength(4);
    for(const s of manifest.sources) expect(hash(read(packet+s.file))).toBe(s.sha256);
    const text=read(chapter).toString();
    expect(text).toContain('id="redpajama-releases"');
    expect(text).toContain('id="roots-multilingual-corpus"');
    expect(text).toContain('20.8 billion documents');
    expect(text).toContain('30.4 trillion tokens');
    expect(text).toContain('1.6 **TB of text**');
    expect(text).toContain('sum to 47%');
    expect(text).toContain('remove 23.1%');
    expect(text).toContain('not training-token weights');
    expect(text.match(/^```python$/gm)).toHaveLength(2);
    expect(text.match(/^!\[\[/gm)).toHaveLength(9);
    const card=read(packet+'redpajama-v2-README.md').toString();
    expect(card).toContain('| Total | 20.8B');
    expect(card).toContain('30.4T');
    expect(card).toContain('duplicates were kept in the dataset');
    expect(card).toContain('Raw Document and Token Counts (`head_middle`)');
  });

  it('retains the complete original ROOTS image with actual license evidence',()=>{
    const [a]=JSON.parse(read('00 Учебник/Assets/Figures/curated/corpus-releases-2026-10-29/assets.json').toString());
    const registry=parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(registry.filter((r:any)=>r.asset===a.asset)).toEqual([a]);
    expect(read(a.asset)).toEqual(read(a.source_asset));
    expect(hash(read(a.asset))).toBe(a.sha256);
    expect(a.dimensions_px).toEqual([1552,279]);
    expect(a.license).toBe('CC BY 4.0');
    expect(hash(read(a.license_evidence))).toBe(a.license_evidence_sha256);
    expect(read(a.license_evidence).toString()).toContain('License: CC BY 4.0');
    expect(a.modifications).toContain('no crop, redraw');
  });

  it('restores both preceding files exactly without accepting unrelated changes',()=>{
    const j=JSON.parse(read('docs/audit/2026-10-29-corpus-releases-additions.json').toString());
    expect(j.files).toHaveLength(2);
    for(const row of j.files){
      const current=read(row.source).toString();
      const live=beforeTopicConsolidation(current,row.source,root,
        '2026-10-29-corpus-releases-additions.json');
      expect(hash(live)).toBe(row.after_sha256);
      const restored=beforeTopicConsolidation(current,row.source,root,
        '2026-10-29-tool-rewards-additions.json');
      expect(hash(restored)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(current+'\nUnreviewed change.',row.source,root,
        '2026-10-29-tool-rewards-additions.json')).toThrow();
    }
  });
});
