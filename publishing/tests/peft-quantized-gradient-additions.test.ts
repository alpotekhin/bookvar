import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-peft-quantized-gradient-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0];
const body=read(owner.source);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const figs=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
const reviewed=()=>beforeTopicConsolidation(body,owner.source,root,own);
const original=()=>{let value=reviewed();for(const c of [...owner.changes].reverse())value=value.replace(c.new,()=>c.old);return value;};

describe('PEFT quantized-base worked gradient',()=>{
  it('reverses only the four recorded chapter, index and verifier deltas',()=>{
    expect(journal.files).toHaveLength(4);
    for(const row of journal.files){
      let value=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(value),row.source).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()){
        expect(value.split(c.new).length,row.source).toBe(2);
        value=value.replace(c.new,()=>c.old);
      }
      expect(hash(value),row.source).toBe(row.before_sha256);
    }
  });
  it('preserves the exact accepted explanation and all eighteen preceding programs',()=>{
    const old=original();
    expect(hash(old)).toBe('5102f8b7db11f1ce749c7263d75bb124cbeb6ed217e5d6762e10b6f9d96d24b0');
    expect(programs(body)).toHaveLength(19);
    expect(programs(old)).toHaveLength(18);
    expect(programs(body).filter(x=>!x.includes('q_original ='))).toEqual(programs(old));
    expect([...body.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]))
      .toEqual([...old.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]));
    expect(matter(body).data.source_language).toBe('en');
  });
  it('points both subject indexes to the same complete English explanation',()=>{
    const parsed=matter(body),terms=referenceTerms(owner.source,parsed.data.reference_terms,parseMarkdownHeadings(parsed.content,true));
    expect(terms).toHaveLength(18);
    for(const [p,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
      for(const t of terms)expect(read(p)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('preserves all ten source-image paths and registered byte hashes',()=>{
    expect(figs(reviewed())).toEqual(figs(original()));
    expect(figs(reviewed())).toHaveLength(10);
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const asset of figs(body)){
      const row=assets.find((x:any)=>x.asset===asset);
      expect(row,asset).toBeDefined();
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(row.sha256);
    }
  });
  it('labels the toy codebook, storage, compute precision and source separately',()=>{
    const section=body.split('<a id="qlora-frozen-gradient-example"></a>')[1]
      .split('### NF4, double quantization and paged optimizers')[0];
    for(const value of ['uniform signed four-bit','ordinary integer tensors',
      'not to reproduce NF4','FP64','not pack two codes','reconstructed',
      'https://arxiv.org/html/2305.14314v1#S3','2.025','q_x.grad','q_codes_before'])
      expect(section,value).toContain(value);
    expect(read('publishing/tools/check_peft_examples.py')).toContain('original_gradient');
  });
});

