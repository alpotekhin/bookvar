import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString(),sha=(s:Buffer|string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-rnn-library-contract-additions.json',stop='2026-11-03-word-embeddings-data-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);

describe('RNN library parameterization and valid-length state handling',()=>{
  it('restores the exact preceding whole owner and both indexes, rejecting extra edits',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('9e7990cd4670623c0f8b4cc78fc233d30dfca3696143dee5ad01f87f2fa136b4');
  });
  it('retains all six previous programs, display equations and original SVG bytes',()=>{
    expect(programs(current)).toHaveLength(8);
    expect(programs(current).filter(x=>!x.includes('effective_after_two_biases')&&!x.includes('packed_output')))
      .toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(math(current)).toEqual(math(prior));
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
    expect(images(current)).toEqual(images(prior));
    expect(images(current)).toHaveLength(6);
    const assets=parse(text('05 Источники/asset-registry.yml')).assets;
    for(const prefixed of images(current)){
      const path=prefixed.replace(/^02 Areas\/ML & DL\//,'');
      const entries=assets.filter((a:any)=>a.asset===path);
      expect(entries,path).toHaveLength(1);
      expect(sha(read(path)),path).toBe(entries[0].sha256);
      expect(entries[0].license).toBe('CC BY-SA 4.0');
    }
    const excerpt=(s:string)=>s.split('<!-- source-excerpt: d2l-stacked-inputs -->\n')[1].split('\n<!-- /source-excerpt -->')[0];
    expect(excerpt(current)).toBe(excerpt(prior));
  });
  it('distinguishes a feature extractor, two-bias updates and true final states',()=>{
    for(const phrase of ['does not\ninclude the embedding table','not the hidden-state\nlayout',
      'single_bias.detach() - 2 * rate * g','give 100 parameters instead of 96',
      'seven vectors instead of twelve padded slots','[3,2,1,1]',
      'lengths are positive','not a mask\nfor left padding',
      'PAD-time transitions','does not reset states between documents'])
      expect(current).toContain(phrase);
    expect(current).toContain('работает одна и та же ячейка с общими весами');
    expect(current).not.toContain('меняется с длиной текста, поэтому число параметров');
  });
  it('indexes twenty-two mechanisms in both languages without translating the owner',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(22);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(current).toContain('https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html');
  });
});
