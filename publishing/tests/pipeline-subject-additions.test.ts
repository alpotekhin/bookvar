import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-pipeline-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0];
const programs=(s:string)=>[...s.matchAll(/```([^\n]+)\n([\s\S]*?)```/g)].map(m=>[m[1],m[2]]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('pipeline-parallel subject alignment',()=>{
  it('reverses all three files to their exact prior hashes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const row of journal.files){
      let current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(current),row.source).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()){
        expect(current.split(c.new).length,row.source).toBe(2);
        current=current.replace(c.new,()=>c.old);
      }
      expect(hash(current),row.source).toBe(row.before_sha256);
    }
  });
  it('preserves the complete explanation,all code/timelines,equations,figures and old anchors',()=>{
    const text=read(owner.source);let original=text;
    for(const c of [...owner.changes].reverse())original=original.replace(c.new,()=>c.old);
    expect(hash(original)).toBe('f622f710aff96bc0bbc4c4ae604e9a66d91db015c3c10428f35bc9c5280b0c33');
    expect(programs(text)).toEqual(programs(original));
    expect(programs(text).filter(p=>p[0]==='python')).toHaveLength(3);
    expect(formulas(text)).toEqual(formulas(original));
    expect(formulas(text)).toHaveLength(11);
    expect(figures(text).map(m=>m[1])).toEqual(figures(original).map(m=>m[1]));
    expect(figures(text)).toHaveLength(9);
    expect(figures(text).every(m=>m[2]?.trim())).toBe(true);
    expect(matter(text).data.source_unit_id).toEqual(matter(original).data.source_unit_id);
    for(const [,anchor]of original.matchAll(/<a id="([^"]+)"/g))expect(text).toContain('<a id="'+anchor+'"');
  });
  it('exposes all29 mechanisms through the same owner in both reference indexes',()=>{
    const page=matter(read(owner.source));
    const terms=referenceTerms(owner.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
    expect(terms).toHaveLength(29);
    expect(new Set(terms.map(t=>t.anchor)).size).toBe(29);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const term of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
  });
  it('pins all nine unchanged complete source figures to their existing hashes',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figures(read(owner.source))){
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===path);
      expect(rows.length,path).toBe(1);
      expect(hash(readFileSync(resolve(root,path))),path).toBe(rows[0].sha256);
    }
  });
});
