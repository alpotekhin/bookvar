import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-kernels-course-additions.json';
const stop='2026-11-03-gmm-course-additions.json';
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0];
const raw=read(row.source);
const current=beforeTopicConsolidation(raw,row.source,root,own);
const prior=beforeTopicConsolidation(raw,row.source,root,stop);

describe('kernel chapter original source additions',()=>{
  it('reverses only exact pinned additions in the chapter, registry and both indexes',()=>{
    expect(row.before_sha256).toBe('99280edf9b5259f46e5f8d797410aad56da0bf726be38c54ac5a404fc375811f');
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });
  it('preserves nine complete programs and all eight original figure paths',()=>{
    const fences=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(fences(prior)).toHaveLength(9);
    expect(fences(current)).toHaveLength(10);
    expect(fences(current).slice(0,9)).toEqual(fences(prior));
    expect(images(prior)).toHaveLength(8);
    expect(images(current)).toHaveLength(10);
    expect(images(current).slice(0,8)).toEqual(images(prior));
    for(const phrase of ['условие Слейтера','Условия KKT','псевдообратную',
      '169 пар','rbf_folds','rescaled_gamma','её середина не закреплена в нуле',
      'а не независимая итоговая оценка','код не воспроизводит историческую карту'])
      expect(current).toContain(phrase);
  });
  it('compares the complete selected English paragraph with the archived source', { tags: ['source-archive'] }, ()=>{
    const source=read('05 Источники/Originals/kernels-2026-11-03/sklearn-rbf-parameters-1.9.1.py');
    const quote=source.match(/For intermediate values,[\s\S]*?performing models\./)![0];
    const plain=(s:string)=>s.replace(/^> ?/gm,'').replace(/`/g,'').replace(/\s+/g,' ').trim();
    expect(plain(current)).toContain(plain(quote));
  });
  it('retains the exact image bytes and source review metadata',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/kernels-2026-11-03/assets.json'));
    expect(rows).toHaveLength(2);
    for(const asset of rows)expect(sha(readFileSync(resolve(root,asset.asset)))).toBe(asset.sha256);
    expect(journal.sourceComparison.fullyRead).toHaveLength(3);
  });
  it('links nine mechanisms in both reference indexes to their canonical anchors',()=>{
    const stem=row.source.replace(/\.md$/,'');
    const anchors=['kernel-gram','svm-geometric-margin','svm-dual-kkt','kernel-prediction-cost',
      'random-fourier-features','rff-approximation-variance','nystrom-anchor-projection',
      'rbf-c-gamma','rff-fixed-map-scale'];
    for(const anchor of anchors){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+stem+'#'+anchor+'|');
    }
  });
});
