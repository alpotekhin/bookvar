import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-systems-timeline-additions.json';
const stop='2026-11-03-sequence-state-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const body=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);

describe('introductory system timeline, deadlines and matched populations',()=>{
  it('restores the entire preceding owner and both indexes without accepting unreviewed edits',()=>{
    expect(journal.files).toHaveLength(3);
    for(const f of journal.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(hash(prior)).toBe('eb51bcc5c98e9e2dc9ce0afed7f14315965a8abcdff0e176e6094d5943d40764');
    expect(images(body)).toEqual(images(prior));
    expect(images(body)).toHaveLength(6);
    const descriptions=[...body.matchAll(/!\[\[[^\]|]+\|([^\]]+)\]\]/g)].map(x=>x[1]);
    expect(descriptions).toHaveLength(6);
    expect(descriptions.every(x=>x.length>30)).toBe(true);
    expect(math(prior)).toHaveLength(2);
    expect(math(body)).toHaveLength(5);
    expect(math(body)).toEqual(expect.arrayContaining(math(prior)));
  });
  it('retains the causal schedule, explicit admission loss and consistent Little-law boundary',()=>{
    expect([...body.matchAll(/^~~~python\n([\s\S]*?)^~~~\s*$/gm)]).toHaveLength(3);
    for(const text of ['def stage_timeline(', 'def admit_serial(', 'C_{i,j-1},C_{i-1,j}',
      '[35, 69, 103]', '[35, 54, 73]', 'assert rejected == [1, 2]',
      '1.725', '1.35', '0.875', 'not a CUDA benchmark',
      'not the GPU batch size', 'absolute deadline',
      'its Queue Management and Latency and Deadlines sections'])
      expect(body).toContain(text);
    expect(body).toContain('https://sre.google/sre-book/addressing-cascading-failures/');
    expect(body).not.toContain('https://sre.google/sre-book/handling-overload/');
    expect(body).toContain('58c Queueing и capacity planning|Queueing and capacity planning');
    expect(body.indexOf('id="sys-timeline"')).toBeLessThan(body.indexOf('id="sys-dam"'));
  });
  it('links fifteen actual subjects from both indexes without a false English-edition label',()=>{
    const meta=matter(body).data;
    expect(meta.reference_terms).toHaveLength(15);
    expect(meta.source_language).toBe('ru');
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(body).toContain('_index#Углублённые модули|');
    expect(body).toContain('02 GPU, CUDA и иерархия памяти|');
  });
});
