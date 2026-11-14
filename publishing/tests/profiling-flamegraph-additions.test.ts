import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-profiling-flamegraph-additions.json',stop='2026-11-04-data-pipeline-additions.json';
const j=JSON.parse(read('docs/audit/'+own)),row=j.files[0],body=read(row.source);
const prior=beforeTopicConsolidation(body,row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^```(?:python|bash)\n([\s\S]*?)^```\s*$/gm)].map(x=>x[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
describe('sampled flame graphs and profiling subject integration',()=>{
  it('restores the full1105-line accepted explanation and every previous program/equation/figure',()=>{
    expect(hash(prior)).toBe('f6fdc8028d115f4026643fc090bc843f2fdd8ddfc97207332681bca22e672a80');
    expect(j.files).toHaveLength(4);
    for(const f of j.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(programs(body)).toHaveLength(13);
    expect(programs(body)).toEqual(programs(prior));
    expect(formulas(body)).toEqual(formulas(prior));
    expect(figures(prior)).toHaveLength(5);
    expect(figures(body)).toHaveLength(6);
    expect(figures(body)).toEqual(expect.arrayContaining(figures(prior)));
  });
  it('explains the actual source counts,ancestry,population and loss of temporal order',()=>{
    for(const fact of ['2893','1490','1371','814','813','51.50%','47.39%','28.14%','28.10%',
      'root is at the top','not chronology','not mean that the function was called',
      'three times','[D,D,D,D,D,D,W,W]','[D,D,D,W,D,D,D,W]',
      'native extensions that release it','not a fresh','not establish a GPU bottleneck'])
      expect(body).toContain(fact);
    const m=JSON.parse(read('publishing/tools/profiling-flamegraph-sources.json'));
    expect(read(m.figure)).toBe(read(m.sources[0].path));
    expect(body).not.toContain('конвейер из S6');
    expect(body).not.toContain('продолжается в S8');
    expect(body).not.toContain('гипотезу для S8');
  });
  it('indexes24 actual subjects in both interfaces without a translation or lost legacy anchors',()=>{
    const meta=matter(body).data;
    expect(meta.status).toBe('draft');
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(24);
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    const oldAnchors=[...prior.matchAll(/<a id="([^"]+)"><\/a>/g)].map(x=>x[1]);
    for(const anchor of oldAnchors)expect(body.split('<a id="'+anchor+'"></a>')).toHaveLength(2);
    expect(meta.source_unit_id).toHaveLength(8);
  });
});
