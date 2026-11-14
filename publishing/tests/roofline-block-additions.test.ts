import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-roofline-block-additions.json';
const stop='2026-11-04-gpu-copy-additions.json';
const j=JSON.parse(read('docs/audit/'+own)),row=j.files[0];
const body=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const code=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(x=>x[1]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
describe('roofline block timings and observation boundaries',()=>{
  it('restores the exact accepted1105-line owner and both indexes',()=>{
    expect(j.files).toHaveLength(3);
    expect(hash(prior)).toBe('ee3b591a0085982c18c2ae691889e38f2f624b505cb2a414520c947c1f57323b');
    for(const f of j.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(code(body)).toEqual(code(prior));
    expect(code(prior)).toHaveLength(6);
    expect(images(body)).toEqual(images(prior));
    expect(images(body)).toHaveLength(9);
    expect([...body.matchAll(/!\[\[[^\]|]+\|[^\]]+\]\]/g)]).toHaveLength(9);
    expect(math(body)).toHaveLength(math(prior).length+1);
    expect(math(body)).toEqual(expect.arrayContaining(math(prior)));
    const source=(s:string)=>[...s.matchAll(/<!-- BEGIN SOURCE: (.*?) -->([\s\S]*?)<!-- END SOURCE: \1 -->/g)].map(x=>x[0]);
    expect(source(body)).toEqual(source(prior));
    expect(source(prior)).toHaveLength(7);
  });
  it('explains calibration,stored units,lost tails and the narrow meaning of the IQR criterion',()=>{
    expect([...body.matchAll(/^~~~python\n/gm)]).toHaveLength(1);
    for(const text of ['2012 microseconds','not measurements of Python or CUDA',
      'not an upper bound','only one returned measurement',
      'number_per_run','raw_times','98 calls at 1 ms, two at 451 ms',
      'TaskSpec','It does not run a timed operation',
      'nearest_rank(bursty_calls, .99) == 451','IQR / median < threshold',
      'cannot reconstruct','the errors and timeouts'])
      expect(body).toContain(text);
    expect(body.indexOf('id="perf-blocked-samples"')).toBeGreaterThan(body.indexOf('id="perf-statistics"'));
    expect(body.indexOf('id="perf-blocked-samples"')).toBeLessThan(body.indexOf('id="perf-load"'));
    expect(body).toContain('pytorch-tutorials-5672cd98-BSD-3-Clause.txt');
  });
  it('links24 actual subjects from both indexes without a separate translated owner',()=>{
    const meta=matter(body).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(24);
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(body.match(/id="cs336-systems-roofline"/g)).toHaveLength(1);
    expect(body).toContain('02 GPU, CUDA и иерархия памяти|');
    expect(body).toContain('04 Арифметика Transformer и MoE|');
  });
});

