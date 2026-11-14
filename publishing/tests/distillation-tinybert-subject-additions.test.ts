import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-distillation-tinybert-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0],body=read(owner.source);
const original=()=>{let s=beforeTopicConsolidation(body,owner.source,root,own);for(const c of [...owner.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
const tinyPart=()=>body.split('<a id="tinybert-transformer-distillation"></a>')[1]
  .split('## Дистилляция на собственных генерациях ученика')[0];

describe('source-faithful TinyBERT explanation and distillation subjects',()=>{
  it('inverts all eight exact file deltas',()=>{
    expect(journal.files).toHaveLength(8);
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
  it('preserves the preceding eight programs, every derivation and legacy anchor',()=>{
    const old=original();
    expect(hash(old)).toBe('94becb658274ac9476ba8bb06c904b25ec6a71d5fe60e3453cb2884f186ad7b4');
    expect(programs(body)).toHaveLength(9);
    expect(programs(body).filter(s=>!s.startsWith('tinybert_teacher_scores ='))).toEqual(programs(old));
    const displays=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
    const withoutTiny=body.replace('<a id="tinybert-transformer-distillation"></a>'+tinyPart(),'');
    expect(displays(withoutTiny)).toEqual(displays(old));
    expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
    expect(matter(body).data.source_language).toBe('mixed');
    for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))expect(body).toContain('id="'+id+'"');
  });
  it('makes 38 explanations reachable through both subject indexes',()=>{
    const p=matter(body),terms=referenceTerms(owner.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(38);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('retains the earlier figures and verifies the added original panels',()=>{
    const figs=figures(body),old=figures(original());
    expect(figs).toHaveLength(10);
    expect(figs.filter(m=>!m[1].includes('distillation-tinybert-2026-11-05')&&!m[1].includes('distill-rlvr-2026-11-14')).map(m=>m[1]))
      .toEqual(old.map(m=>m[1]));
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figs){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const row=assets.find((x:any)=>x.asset===path);
      expect(row,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path))),path).toBe(row.sha256);
    }
    const tiny=JSON.parse(read('00 Учебник/Assets/Figures/curated/distillation-tinybert-2026-11-05/assets.json'));
    expect(tiny).toHaveLength(2);
    for(const row of tiny){
      expect(row.license).toBe('CC BY 4.0');
      expect(hash(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
      expect(row.modifications).toContain('byte-for-byte');
      expect(row.derivation).toBe('source-raster-copy');
      const bytes=readFileSync(resolve(root,row.asset));
      expect([bytes.readUInt32BE(16),bytes.readUInt32BE(20)]).toEqual(row.dimensions_px);
      expect(assets.find((x:any)=>x.id===row.id)).toEqual(row);
    }
  });
  it('separates scores from probabilities, mapping indices, projection sharing and stages',()=>{
    const s=tinyPart();
    for(const value of ['before softmax','4.25','17','312','768','twelve attention',
      '$2,5,8,11$','$0,3,6,9,12$','shared', 'without prediction-layer distillation',
      '--pred_distill','not a TinyBERT training run','sentence-level tasks',
      'separate validity mask','factor $1/C$','0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9'])
      expect(s,value).toContain(value==='shared'?'shares one':value);
    expect(read('05 Источники/Papers/Sequence-Level Knowledge Distillation.md'))
      .not.toContain('Рисунки статьи в учебник не переносились.');
    expect(read('publishing/navigation.yml')).toContain('route: sources/papers/tinybert');
    expect(read('05 Источники/Papers/TinyBERT.md')).toContain('#tinybert-transformer-distillation|');
  });
});
