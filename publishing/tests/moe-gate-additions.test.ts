import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
import {beforeReaderRouteNavigation} from '../tools/reader-route-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-moe-gate-additions.json',stop='2026-11-03-gan-gradient-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^~~~python\n([\s\S]*?)^~~~\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);

describe('MoE report and released-Gate boundary',()=>{
  it('restores the owner, both indexes and the exact earlier explanation-review boundary',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('9df758b69b1d720bbb8942594442f5c6e45071d70b28f740793fb57de2fd746c');
    expect(sha(beforeReaderRouteNavigation(read(row.source),row.source,root)))
      .toBe('f6f6110b23ed5defba31c16a5332e5d0bfb5f7190b0ecfc1b0d7cd42e9524042');
  });
  it('keeps all equations, three unaffected programs and all nineteen original image bytes',()=>{
    expect(programs(current)).toHaveLength(4);
    expect(programs(current).slice(0,3)).toEqual(programs(prior).slice(0,3));
    expect(math(current)).toEqual(math(prior));
    expect(images(current).map(x=>x[1])).toEqual(images(prior).map(x=>x[1]));
    expect(images(current)).toHaveLength(19);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of images(current)){
      expect(image[2]?.length).toBeGreaterThan(35);
      const asset=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const record=registry.find((x:any)=>x.asset===asset);
      expect(record,asset).toBeTruthy();
      expect(sha(readFileSync(resolve(root,asset)))).toBe(record.sha256);
    }
  });
  it('distinguishes the report generalization, biased implementation and node admission from quota',()=>{
    for(const phrase of ['group_topk=None','group_topk = k // allowed_groups',
      'group_topk=2','hard-codes a two-score sum','Zero-valued biases',
      'it does not impose a per-node quota','[[1, 3]]','[[3, 4]]',
      'route_scale','stable\nsorting for ties'])
      expect(current).toContain(phrase);
    expect(current).not.toContain('Краткое описание механизма находится');
    expect(current).not.toContain('В предыдущей главе мы разобрали FFN');
    const fm=matter(current).data;
    expect(fm.previous).toContain('03 Длинный контекст');
    expect(fm.next).toContain('03 Mamba');
  });
  it('indexes twenty-one mechanisms in both locales without claiming a translated English owner',()=>{
    const meta=matter(current).data;
    expect(meta.reference_terms).toHaveLength(21);
    expect(meta.source_language).toBeUndefined();
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
  });
});
