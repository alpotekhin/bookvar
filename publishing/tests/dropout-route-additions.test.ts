import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-dropout-route-additions.json',stop='2026-11-03-normalization-course-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const wrapping=row.changes.at(-1);
const unwrapped=current.replace(wrapping.new,()=>wrapping.old);
const programs=(s:string)=>[...s.matchAll(/^\x60\x60\x60python\n([\s\S]*?)^\x60\x60\x60\s*$/gm)].map(m=>m[1]);
const structured=(s:string)=>s.split('<!-- BEGIN REVIEW: structured dropout 2026-10-29 -->')[1].split('<!-- END REVIEW: structured dropout 2026-10-29 -->')[0];

describe('regularization core before structured mask extensions',()=>{
  it('restores the exact preceding chapter and both subject indexes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
  });
  it('preserves all eleven programs, original structured explanation, equations and five figures',()=>{
    const old=programs(prior);
    expect(old).toHaveLength(11);
    expect(programs(current)).toEqual([...old.slice(0,5),...old.slice(8),...old.slice(5,8)]);
    expect(structured(unwrapped)).toBe(structured(prior));
    for(const expression of [/^\$\$\n([\s\S]*?)^\$\$/gm,/!\[\[[^\]]+\]\]/g]){
      const objects=(s:string)=>[...s.matchAll(expression)].map(m=>m[0]).sort();
      expect(objects(unwrapped)).toEqual(objects(prior));
    }
    expect([...current.matchAll(/!\[\[/g)]).toHaveLength(5);
    for(const caption of ['Рисунок из статьи воспроизведён','сохранённый запуск курса',
      'The full original figure and caption are retained','Gaussian approximation'])
      expect(current).toContain(caption);
  });
  it('indexes sixteen mechanisms and keeps the core continuation ahead of extensions',()=>{
    const terms=matter(current).data.reference_terms;
    expect(terms).toHaveLength(16);
    for(const term of terms){
      expect(current).toContain('<a id="'+term.anchor+'"></a>');
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+term.anchor+'|');
    }
    const at=(s:string)=>current.indexOf(s);
    expect(at('## Как шум меняет обучающую цель')).toBeLessThan(at('## DropConnect'));
    expect(at('## Контрольный эксперимент')).toBeLessThan(at('<!-- BEGIN REVIEW: structured dropout'));
    expect(at('06a Аугментация данных')).toBeLessThan(at('<!-- BEGIN REVIEW: structured dropout'));
    expect(at('## Stochastic depth')).toBeLessThan(at('## Источники'));
  });
});
