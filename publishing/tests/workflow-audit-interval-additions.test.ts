import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-09-workflow-audit-interval-additions.json';
const previous='2026-11-09-rag-subject-index-additions.json';
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0],body=read(row.source);
const prior=beforeTopicConsolidation(body,row.source,root,previous);

it('restores the reviewed input, preserving every earlier program and image destination',()=>{
  for(const r of journal.files){
    const live=read(r.source);
    const current=beforeTopicConsolidation(live,r.source,root,own);
    expect(sha(current)).toBe(r.after_sha256);
    expect(sha(beforeTopicConsolidation(live,r.source,root,previous))).toBe(r.before_sha256);
    expect(()=>beforeTopicConsolidation(live+'\ndrift\n',r.source,root,previous)).toThrow();
  }
  const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```/gm)].map(m=>m[1]);
  expect(programs(body).filter(p=>!p.includes('def fn_interval95('))).toEqual(programs(prior));
  expect(programs(body)).toHaveLength(programs(prior).length+1);
  const figures=(s:string)=>[...s.matchAll(/^!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]$/gm)];
  expect(figures(body)).toHaveLength(9);
  expect(figures(body).map(f=>f[1])).toEqual(figures(prior).map(f=>f[1]));
  expect(figures(body).every(f=>f[2]?.trim())).toBe(true);
  expect(body).toContain('CUDA Graphs#compile-tracing-a-data-dependent-branch');
});

it('indexes 18 mechanisms in both reader languages without a separate translation',()=>{
  const p=matter(body);
  const terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
  expect(terms).toHaveLength(18);
  expect(p.data.source_language).toBe('mixed');
  for(const [path,english] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
    for(const t of terms)
      expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(english?t.label_en:t.label)+']]');
});
