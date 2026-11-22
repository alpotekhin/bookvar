import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-preference-reward-prm-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owners=journal.files.slice(0,2);
const authoredRevision=(row:any)=>beforeTopicConsolidation(read(row.source),row.source,root,own);
const original=(row:any)=>{let t=authoredRevision(row);for(const c of [...row.changes].reverse())t=t.replace(c.new,()=>c.old);return t;};
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const figs=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
const counts=[{terms:15,code:5,figs:3},{terms:21,code:7,figs:8}];

describe('preference/reward subject alignment and PRM boundary loss',()=>{
  it('inverts the five exact content/index/checker deltas',()=>{
    expect(journal.files).toHaveLength(5);
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
  for(const[i,row]of owners.entries()){
    it('preserves the accepted explanation, earlier programs and anchors: '+row.source,()=>{
      const value=read(row.source),old=original(row);
      expect(hash(old)).toBe(row.before_sha256);
      expect(programs(value)).toHaveLength(counts[i].code);
      expect(programs(value).filter(x=>!x.startsWith('prm_step_logits ='))).toEqual(programs(old));
      expect([...value.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]).sort())
        .toEqual([...old.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]).sort());
      expect(matter(value).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
      for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))expect(value).toContain('id="'+id+'"');
    });
    it('links both subject indexes to the full owner: '+row.source,()=>{
      const p=matter(read(row.source)),terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
      expect(terms).toHaveLength(counts[i].terms);
      for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
        for(const t of terms)expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
    });
  }
  it('preserves all eleven original figure paths/bytes with descriptive alternatives',()=>{
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const[i,row]of owners.entries()){
      const images=figs(authoredRevision(row));
      expect(images).toHaveLength(counts[i].figs);
      expect(images.map(m=>m[1])).toEqual(figs(original(row)).map(m=>m[1]));
      const currentImages=figs(read(row.source)).map(m=>m[1]);
      expect(currentImages.filter(path=>images.some(m=>m[1]===path)).sort())
        .toEqual(images.map(m=>m[1]).sort());
      for(const image of images){
        expect(image[2]?.trim()).toBeTruthy();
        const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
        const asset=assets.find((x:any)=>x.asset===path);
        expect(asset,path).toBeDefined();
        expect(hash(readFileSync(resolve(root,path))),path).toBe(asset.sha256);
      }
    }
  });
  it('distinguishes labels, missing annotation and inference aggregation',()=>{
    const t=read(owners[1].source).split('<a id="prm-boundary-loss"></a>')[1]
      .split('<a id="rm-verifiable"></a>')[0];
    for(const phrase of ['первой ошибки включительно','не получает автоматически отрицательные',
      'трёхклассовая голова','не воспроизведение языковой головы',
      'ignore_index=-100','No labeled steps','5/8',
      'при обучении они остаются разными','https://arxiv.org/html/2305.20050v1#S2.SS6'])
      expect(t,phrase).toContain(phrase);
    expect(authoredRevision(owners[1])).toContain('03 Reward modeling#rm-validation|разделу 7');
    const chapter=read(owners[1].source);
    expect(chapter.indexOf('id="rm-validation"')).toBeLessThan(chapter.indexOf('id="rm-self-rewarding"'));
    expect(read('publishing/tools/check_reward_modeling_examples.py')).toContain('wrong_slot_mean_gradient_factor');
  });
});
