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
const own='2026-11-09-reranking-lambdarank-additions.json';
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0],body=read(row.source);
const old=beforeTopicConsolidation(body,row.source,root,'2026-11-06-retrieval-hnsw-additions.json');

it('restores the exact accepted input and rejects unreviewed changes',()=>{
  expect(sha(old)).toBe('5d7522f493395addabf8cac39a95b6cc4ef0c8c134f5afa13ac488d5059657c6');
  for(const r of journal.files){
    const live=read(r.source);
    const current=beforeTopicConsolidation(live,r.source,root,own);
    expect(sha(current)).toBe(r.after_sha256);
    expect(sha(beforeTopicConsolidation(live,r.source,root,'2026-11-06-retrieval-hnsw-additions.json')))
      .toBe(r.before_sha256);
    expect(()=>beforeTopicConsolidation(live+'\ndrift\n',r.source,root,
      '2026-11-06-retrieval-hnsw-additions.json')).toThrow();
  }
});

it('keeps earlier programs, derivations and figures while adding the worked pair example',()=>{
  const blocks=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```/gm)].map(m=>m[1]);
  const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
  const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
  expect(blocks(body)).toHaveLength(8);
  expect(blocks(body).filter(b=>!b.includes('def ndcg_swap_weights'))).toEqual(blocks(old));
  expect(math(body)).toHaveLength(9);
  expect(math(body).filter(b=>!b.includes('w_{ij}'))).toEqual(math(old));
  expect(images(body)).toHaveLength(8);
  expect(images(body).map(m=>m[1])).toEqual(images(old).map(m=>m[1]));
  expect(images(body).every(m=>m[2]?.trim())).toBe(true);
});

it('indexes all subject anchors in the canonical and English fallback routes',()=>{
  const p=matter(body);
  const terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
  expect(terms).toHaveLength(28);
  expect(p.data.source_language).toBe('mixed');
  for(const [path,english] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
    for(const t of terms)
      expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(english?t.label_en:t.label)+']]');
});
