import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-grpo-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0],body=read(owner.source);
const original=()=>{let s=beforeTopicConsolidation(body,owner.source,root,own);for(const c of [...owner.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('GRPO subject navigation without replacing the explanation',()=>{
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
  it('preserves all eleven programs, derivations, source units and legacy anchors',()=>{
    const old=original();
    expect(hash(old)).toBe('11087f7fbb1f65127313d6ba9a640079ace6002235bff2420e5dca41bb3ea217');
    expect(programs(body)).toEqual(programs(old));
    expect(programs(body)).toHaveLength(11);
    const math=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
    expect(math(body)).toEqual(math(old));
    expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
    for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))expect(body).toContain('id="'+id+'"');
    expect(body).toContain('GAE combines value predictions with rewards that include the KL penalty.');
    expect(body).toContain('unchanged; [MIT license notice](/licenses/rlhf-book-MIT.txt).');
  });
  it('links 34 subjects per locale to their full canonical explanations',()=>{
    const p=matter(body),terms=referenceTerms(owner.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(34);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('retains eight original figures and adds one source figure, with alternatives and pinned bytes',()=>{
    const figs=figures(body);
    expect(figs).toHaveLength(9);
    expect(figs.map(m=>m[1]).filter(path=>!path.endsWith('/wen-figure4.svg'))).toEqual(figures(original()).map(m=>m[1]));
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figs){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const row=assets.find((x:any)=>x.asset===path);
      expect(row,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path))),path).toBe(row.sha256);
    }
    expect(body).not.toContain('curated/grpo/ppo-vs-grpo.png');
  });
  it('reuses the historical source review only after an exact bounded-delta inversion',()=>{
    const previous=JSON.parse(read('docs/audit/2026-10-30-figure-replacements-additions.json')).files
      .find((x:any)=>x.source===owner.source);
    let s=original();
    expect(hash(s)).toBe(previous.after_sha256);
    for(const c of [...previous.changes].reverse()){
      expect(s.split(c.new).length).toBe(2);
      s=s.replace(c.new,()=>c.old);
    }
    expect(hash(s)).toBe('7173f3bbdb587d32f8d2f04c5663164c132563b6966ce7543ccc613f4e007e1a');
  });
});
