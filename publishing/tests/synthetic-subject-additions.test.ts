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
const own='2026-11-05-synthetic-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const owners=journal.files.slice(0,2);
const original=(row:any)=>{let s=read(row.source);for(const c of [...row.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const programs=(s:string)=>[...s.matchAll(/^(```|~~~)(python|bash)([^\n]*)\n([\s\S]*?)^\1\s*$/gm)].map(m=>m.slice(2));
const displays=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('synthetic-text and sim-to-real full explanations and subject links',()=>{
  it('inverts four exact owner/index deltas and rejects unknown revisions',()=>{
    expect(journal.files).toHaveLength(4);
    for(const row of journal.files){
      let s=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(s),row.source).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()){
        expect(s.split(c.new).length,row.source).toBe(2);
        s=s.replace(c.new,()=>c.old);
      }
      expect(hash(s),row.source).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(read(row.source)+'\nunrecorded\n',row.source,root,
        '2026-11-05-ttc-subject-additions.json')).toThrow();
    }
  });
  it('retains every executable example, source-only snippet and derivation',()=>{
    expect(owners.map((row:any)=>hash(original(row)))).toEqual([
      '41629ce92425dc01f122181499ddc21b186f0902d18450b140f1a4d91e281abd',
      'ffcc175ff7dcf3852f6d64ab9458033aa339bd4c30f38f55f3f87611af1cf5fc']);
    for(const row of owners){
      const body=read(row.source),old=original(row);
      expect(programs(body)).toEqual(programs(old));
      expect(displays(body)).toEqual(displays(old));
      expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
      for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))
        expect(body).toContain('id="'+id+'"');
    }
    expect(programs(read(owners[0].source))).toHaveLength(3);
    expect(programs(read(owners[1].source))).toHaveLength(12);
  });
  it('exposes37 text and14 image topics in both subject indexes',()=>{
    owners.forEach((row:any,i:number)=>{
      const p=matter(read(row.source));
      const terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
      expect(terms).toHaveLength([37,14][i]);
      expect(new Set(terms.map(t=>t.anchor)).size).toBe(terms.length);
      for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
        for(const t of terms)
          expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
    });
  });
  it('keeps nineteen byte-pinned original figures and every existing caption',()=>{
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    owners.forEach((row:any,i:number)=>{
      const body=read(row.source),old=original(row),images=figures(body);
      expect(images).toHaveLength([12,7][i]);
      expect(images.map(m=>m[1])).toEqual(figures(old).map(m=>m[1]));
      for(const image of images){
        expect(image[2]?.trim()).toBeTruthy();
        const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
        const asset=assets.find((x:any)=>x.asset===path);
        expect(asset,path).toBeDefined();
        expect(hash(readFileSync(resolve(root,path))),path).toBe(asset.sha256);
      }
      for(const caption of old.split(/\n\n/).filter(p=>/^\*(?!\*)/.test(p)&&p.endsWith('*')))
        expect(body).toContain(caption);
    });
  });
  it('states thematic prerequisites without sending synthetic text through TTC',()=>{
    const text=matter(read(owners[0].source)).data;
    expect(text.prerequisites).toEqual([
      '[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества]]']);
    expect(text.source_language).toBe('mixed');
    const image=matter(read(owners[1].source)).data;
    expect(image.source_language).toBe('en');
    expect(image.prerequisites).toEqual([
      '[[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений]]']);
    expect(image.next).toBe('[[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]');
    expect(read(owners[0].source)).not.toContain('Последнее решение не менее существенно:');
  });
});
