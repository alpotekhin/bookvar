import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-rlvr-subject-systems-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0],body=read(owner.source);
const atReview=()=>beforeTopicConsolidation(body,owner.source,root,own);
const original=()=>{let s=atReview();for(const c of [...owner.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('RLVR subjects and source-scoped rollout/learner explanation',()=>{
  it('inverts only the three exact owner/index changes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const row of journal.files){
      let s=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(s),row.source).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()){
        expect(s.split(c.new).length,row.source).toBe(2);
        s=s.replace(c.new,()=>c.old);
      }
      expect(hash(s),row.source).toBe(row.before_sha256);
    }
  });
  it('preserves every preceding program, derivation, native-language block and anchor',()=>{
    const old=original();
    expect(hash(old)).toBe('e47cc4f42c05c58f9ff84aaba765cf21e6ad80f8c658efa159ea510ff5ff5828');
    expect(programs(body)).toEqual(programs(old));
    expect(programs(body)).toHaveLength(6);
    const math=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
    expect(math(body)).toEqual(math(old));
    expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
    expect(matter(body).data.source_language).toBe('mixed');
    for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))expect(body).toContain('id="'+id+'"');
  });
  it('links30 subjects per locale to their full canonical explanations',()=>{
    const p=matter(body),terms=referenceTerms(owner.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(30);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('retains the six reviewed figures and the later crossover plot with pinned bytes',()=>{
    const figs=figures(body);
    const reviewed=figures(atReview()).map(m=>m[1]);
    expect(reviewed).toHaveLength(6);
    expect(reviewed).toEqual(figures(original()).map(m=>m[1]));
    const crossover='02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-pass-k-2026-11-14/yue-figure-2.svg';
    expect(figs.filter(m=>m[1]===crossover)).toHaveLength(1);
    expect(figs.filter(m=>m[1]!==crossover).map(m=>m[1])).toEqual(reviewed);
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figs){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const row=assets.find((x:any)=>x.asset===path);
      expect(row,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path))),path).toBe(row.sha256);
    }
  });
  it('separates worker roles, overlap, lag and the sampling-policy denominator',()=>{
    const part=body.split('<a id="rlvr-rollout-learner-workers"></a>')[1].split('<a id="rlvr-failures"></a>')[0];
    for(const value of ['ZeRO Stage 3','value models','reference policy','not trained',
      'overlap these phases','older policy weights','second-latest inference',
      'does not guarantee on-policy','sampling-policy','updated learner',
      'https://arxiv.org/html/2411.15124v5#S6.SS3','#offpolicy-ratios|'])
      expect(part,value).toContain(value);
    expect(part).not.toMatch(/^\x60{3}/m);
    expect(part).not.toContain('25%');
  });
});
