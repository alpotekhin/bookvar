import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString(),sha=(v:Buffer|string)=>createHash('sha256').update(v).digest('hex');
const own='2026-11-03-neural-lm-bengio-additions.json',stop='2026-11-03-ngram-makemore-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const [asset]=JSON.parse(text('00 Учебник/Assets/Figures/curated/neural-lm-bengio-2026-11-03/assets.json'));

describe('neural LM original architecture and route',()=>{
  it('reconstructs the exact preceding whole owner, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('9e7e0016e68745cfb491d3018beb2d91b81b631a6d1995b36ad3a2e03d2ba695');
  });
  it('preserves all five complete programs and previous original images/equations',()=>{
    expect(programs(current)).toHaveLength(6);
    expect(programs(current).filter(s=>!s.includes('source_vectors'))).toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(math(current).filter(s=>!s.includes('y=b+Wx+U'))).toEqual(math(prior));
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
    expect(images(current)).toHaveLength(4);
    expect(images(current)).toEqual(expect.arrayContaining(images(prior)));
    for(const old of ['autoregressive-objective','stable-cross-entropy','objective-evidence',
      'от-биграммы-к-длинному-контексту','минимальная-реализация-функции-потерь'])
      expect(current).toContain('<a id="'+old+'"></a>');
  });
  it('pins the published architecture figure and qualified publisher-policy notice',()=>{
    expect(parse(text('05 Источники/asset-registry.yml')).assets.filter((a:any)=>a.asset===asset.asset)).toEqual([asset]);
    expect(sha(read(asset.asset))).toBe(asset.sha256);
    expect(asset.dimensions_px).toEqual([1507,980]);
    expect(asset.crop_box_px).toEqual([273,283,1780,1263]);
    expect(asset.license).toBe('CC BY (JMLR publication policy; version unspecified)');
    expect(text(asset.public_license_copy)).toContain('Copyright (c) 2003 the authors.');
    for(const phrase of ['optional','not the output','newest','chronological',
      'must become $(3,4,1,2)$','is therefore $(4,6)$']) expect(current).toContain(phrase);
  });
  it('pins the retained Bengio paper and publisher-policy snapshot', { tags: ['source-archive'] }, ()=>{
    expect(sha(read(asset.source_asset))).toBe(asset.source_asset_sha256);
    expect(sha(read(asset.license_evidence))).toBe(asset.license_evidence_sha256);
  });
  it('indexes eighteen mechanisms and retains the source-unit contract',()=>{
    const meta=matter(current).data;
    expect(meta.reference_terms).toHaveLength(18);
    expect(meta.source_language).toBe('mixed');
    expect(meta.source_unit_id).toEqual(['assignment-01-task-softmax','assignment-01-task-cross-entropy']);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
  });
});
