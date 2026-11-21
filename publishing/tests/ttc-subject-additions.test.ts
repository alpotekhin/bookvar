import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const own='2026-11-05-ttc-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const owner=journal.files[0],body=read(owner.source);
const original=()=>{
  let s=beforeTopicConsolidation(body,owner.source,root,own);
  for(const c of [...owner.changes].reverse())s=s.replace(c.new,()=>c.old);
  return s;
};
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const displays=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('TTC explanations reachable through the common subject index',()=>{
  it('inverts only three recorded source/index deltas',()=>{
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
  it('preserves seven programs, thirteen derivations and all compatibility anchors',()=>{
    const old=original();
    expect(hash(old)).toBe('0828f31f6c72d7d7f37692d171e1a9214f72865e9d0e8bb02420cf399687af03');
    expect(programs(body)).toHaveLength(7);
    expect(programs(body)).toEqual(programs(old));
    expect(displays(body)).toHaveLength(13);
    expect(displays(body)).toEqual(displays(old));
    expect(matter(body).data.source_unit_id).toHaveLength(15);
    expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
    expect(matter(body).data.source_language).toBe('mixed');
    for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))
      expect(body).toContain('id="'+id+'"');
    expect(body).not.toContain('Есть и другая ловушка:');
  });
  it('links36 distinct mechanisms in RU and EN to canonical explanations',()=>{
    const p=matter(body);
    const terms=referenceTerms(owner.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(36);
    expect(new Set(terms.map(t=>t.anchor)).size).toBe(36);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)
        expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('retains every original figure and adds descriptive alternatives without rewriting captions',()=>{
    const current=figures(body),old=figures(original());
    expect(current).toHaveLength(16);
    expect(current.map(m=>m[1])).toEqual(old.map(m=>m[1]));
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of current){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const row=assets.find((x:any)=>x.asset===path);
      expect(row,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path))),path).toBe(row.sha256);
    }
    const oldCaptions=original().split(/\n\n/)
      .filter(p=>/^\*(?!\*)/.test(p)&&p.endsWith('*'));
    expect(oldCaptions).toHaveLength(16);
    for(const caption of oldCaptions)expect(body).toContain(caption);
  });
});
