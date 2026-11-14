import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gzipSync, gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(v:Buffer|string)=>createHash('sha256').update(v).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md';
const auditChapter='00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md';
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
    expect(text.match(/^!\[\[/gm)).toHaveLength(11);
    const card=read(packet+'redpajama-v2-README.md').toString();
    expect(card).toContain('| Total | 20.8B');
    expect(card).toContain('30.4T');
    expect(card).toContain('duplicates were kept in the dataset');
    expect(card).toContain('Raw Document and Token Counts (`head_middle`)');
  });

  it('shows how an inclusive archive range selects exactly one gzip record',()=>{
    const text=read(chapter).toString();
    expect(text).toContain('160581106-160593800');
    expect(160593800-160581106+1).toBe(12695);
    const first=gzipSync('WARC/1.0\r\nWARC-Type: response\r\n\r\nfirst');
    const second=gzipSync('WARC/1.0\r\nWARC-Type: response\r\n\r\nsecond');
    const archive=Buffer.concat([first,second]);
    const start=first.length, end=start+second.length-1;
    expect(gunzipSync(archive.subarray(start,end+1))).toEqual(gunzipSync(second));
    expect(()=>gunzipSync(archive.subarray(start,end))).toThrow();
    expect(text).toContain('206 Partial Content');
    expect(text).toContain('66 093');
  });

  it('pins both FineWeb plots and links to the versioned experiment',()=>{
    const text=read(chapter).toString();
    const registry=parse(read('05 Источники/asset-registry.yml').toString()).assets;
    for(const [name,slug,digest] of [
      ['fineweb-global-dedup-old-snapshot.svg','S3.F4.fig2','e5c8c6b617aa40a380c7361cb1857461908fbe75b448120e12970f9ea8c542d2'],
      ['fineweb-per-snapshot-dedup.svg','S3.F5','04917415942b3720bed94b3c8e7498999afd7d5c9eb29a11435d40adf4e8c56b']
    ]){
      const row=registry.find((r:any)=>r.asset.endsWith('/'+name));
      expect(hash(read(row.asset))).toBe(digest);
      expect(row).toMatchObject({license:'CC BY 4.0',derivation:'source-svg-copy',
        source_asset_sha256:digest,figure_url:'https://arxiv.org/html/2406.17557v2#'+slug,
        used_in:[chapter]});
      expect(text).toContain(row.asset+'|');
      expect(text).toContain(row.figure_url);
      expect(text).toContain(row.source_image_url);
    }
    expect(text).toContain('те же параметры MinHash');
    expect(text).toContain('171 млрд токенов');
    expect(text).toContain('15 трлн после дополнительных фильтров');
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

  it('pins the C4 and Dolma originals and keeps the comparison denominators',()=>{
    const text=read(auditChapter).toString();
    const registry=parse(read('05 Источники/asset-registry.yml').toString()).assets;
    for(const [name,anchor,digest] of [
      ['dolma-figure31-c4.svg','#A15.F31.fig3','f3ebc96ef6eadc1d217bf6c087fccf58fd2c2822a7887bc15789305a6edea2f9'],
      ['c4-figure5-blocklist-pmi.svg','#A1.F5','f75800b58acb5c4e5d6df786dfa060239eb474b0845791b5759135c026d94da6']
    ]){
      const matches=registry.filter((r:any)=>r.asset.endsWith('/'+name));
      expect(matches).toHaveLength(1);
      const [row]=matches;
      expect(hash(read(row.asset))).toBe(digest);
      expect(row).toMatchObject({license:'CC BY 4.0',source_asset_sha256:digest,
        derivation:'source-svg-copy',used_in:[auditChapter]});
      expect(row.figure_url.endsWith(anchor)).toBe(true);
      for(const val of [row.asset+'|',row.figure_url,row.source_image_url])
        expect(text).toContain(val);
    }
    for(const val of ['от одного до пяти','шесть и больше','≥5','0,02%','0,001%',
      '53,6% предложений','1,8% вопросов','6,2%','0{,}24/0{,}08=3'])
      expect(text).toContain(val);
    expect((24/100)/(800/10_000)).toBe(3);
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
