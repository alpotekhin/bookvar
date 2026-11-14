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
const own='2026-11-09-rag-subject-index-additions.json';
const previous='2026-11-09-reranking-lambdarank-additions.json';
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0],liveBody=read(row.source);
const body=beforeTopicConsolidation(liveBody,row.source,root,own);

it('preserves exact prior prose, code, image destinations and legacy anchors',()=>{
  for(const r of journal.files){
    const live=read(r.source);
    const current=beforeTopicConsolidation(live,r.source,root,own);
    expect(sha(current)).toBe(r.after_sha256);
    expect(sha(beforeTopicConsolidation(live,r.source,root,previous))).toBe(r.before_sha256);
    expect(()=>beforeTopicConsolidation(live+'\ndrift\n',r.source,root,previous)).toThrow();
  }
  const prior=beforeTopicConsolidation(liveBody,row.source,root,previous);
  expect(sha(prior)).toBe('b4fe44643a27e1d9319cabad47072c10d037efa9a74eaf9b1daf9044a20c03d3');
  const figures=(s:string)=>[...s.matchAll(/^!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]$/gm)];
  expect(figures(body)).toHaveLength(8);
  expect(figures(body).map(x=>x[1])).toEqual(figures(prior).map(x=>x[1]));
  expect(figures(body).every(x=>x[2]?.trim())).toBe(true);
  expect(body.match(/^<span id="[^"]+"><\/span>$/gm)).toEqual(prior.match(/^<span id="[^"]+"><\/span>$/gm));
});

it('links 23 subjects to explicit anchors in both reference indexes',()=>{
  const p=matter(body);
  const terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
  expect(terms).toHaveLength(23);
  for(const [path,english] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
    for(const t of terms)
      expect(beforeTopicConsolidation(read(path),path,root,own)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(english?t.label_en:t.label)+']]');
});

it('adds one Self-RAG subject while preserving the 23 earlier anchors',()=>{
  const historical=matter(body), current=matter(liveBody);
  const before=referenceTerms(row.source,historical.data.reference_terms,parseMarkdownHeadings(historical.content,true));
  const after=referenceTerms(row.source,current.data.reference_terms,parseMarkdownHeadings(current.content,true));
  expect(after.filter(t=>t.anchor!=='rag-self-rag')).toEqual(before);
  expect(after.find(t=>t.anchor==='rag-self-rag')).toMatchObject({
    label: 'Self-RAG: когда искать и как выбирать продолжение',
    label_en: 'Self-RAG: when to retrieve and how to select a continuation',
  });
  for(const path of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
    expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#rag-self-rag|');
});
