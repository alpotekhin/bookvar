import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-data-pipeline-additions.json';
const stop='2026-11-04-precision-accumulation-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const body=read(row.source);
const prior=beforeTopicConsolidation(body,row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(x=>x[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);

describe('data pipeline DALI graph and subject integration',()=>{
  it('restores every byte of the accepted964-line chapter and preceding shared files',()=>{
    expect(hash(prior)).toBe('5f2a58816bfc82f2ccb0e43cea2a4bdbae1d22d8aae17d0fb36f82a2baac2917');
    expect(journal.files).toHaveLength(4);
    for(const f of journal.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(programs(prior)).toHaveLength(9);
    expect(programs(body)).toHaveLength(10);
    expect(programs(body)).toEqual(expect.arrayContaining(programs(prior)));
    expect(figures(prior)).toHaveLength(6);
    expect(figures(body)).toHaveLength(7);
    expect(figures(body)).toEqual(expect.arrayContaining(figures(prior)));
    expect(formulas(body)).toEqual(formulas(prior));
  });
  it('explains the actual two-reader graph without a GPU or speedup claim',()=>{
    const manifest=JSON.parse(read('publishing/tools/dali-data-sources.json'));
    expect(body).toContain('```python\n'+manifest.source_code+'\n```');
    for(const fact of ["first reader's labels bypass the decoder",
      "second reader's labels are unused","connect an operator's",
      'executes the definition function once','prunes subgraphs',
      'not the preceding EDLS','not a validation','CPU coin flip',
      'labels.gpu()','syntax-checked, not executed on a GPU'])
      expect(body).toContain(fact);
    expect(read(manifest.figure)).toBe(read(manifest.sources[1].path));
  });
  it('links21 subjects to the same canonical explanation in both locales',()=>{
    const meta=matter(body).data;
    expect(meta.status).toBe('canonical');
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(21);
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    for(const name of ['cached-dataset.svg','pinmem.png','same-stream.png',
      'separate-stream.png','optimal-sequence-processing.png','padding-packing.png'])
      expect(body).toMatch(new RegExp(name.replace('.','\\.')+'\\|[^\\]]+\\]\\]'));
  });
});
