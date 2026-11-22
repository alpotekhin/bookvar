import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-dpo-mass-kl-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0];
const body=read(owner.source);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
const reviewed=()=>beforeTopicConsolidation(body,owner.source,root,own);
const original=()=>{let s=reviewed();for(const c of [...owner.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};

describe('DPO probability-mass and KL worked example',()=>{
  it('inverts the four exact owner/index/checker deltas',()=>{
    expect(journal.files).toHaveLength(4);
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
  it('retains the accepted six programs, all display derivations and source units',()=>{
    const old=original();
    expect(hash(old)).toBe('fd7a08d5813a5427b74a943d8a57f00978877b8edce13ac1027ccc04cf172c06');
    expect(programs(body)).toHaveLength(7);
    expect(programs(body).filter(s=>!s.startsWith('mass_reference ='))).toEqual(programs(old));
    const displays=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
    expect(displays(body)).toEqual(displays(old));
    expect(matter(body).data.source_unit_id).toEqual(matter(old).data.source_unit_id);
    for(const[,id]of old.matchAll(/<(?:a|span) id="([^"]+)"/g))
      expect(body).toContain('id="'+id+'"');
  });
  it('links26 subjects in both locales to the same full owner',()=>{
    const p=matter(body),terms=referenceTerms(owner.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(26);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('preserves all five original image paths and registry byte hashes',()=>{
    const figs=images(reviewed());
    expect(figs).toHaveLength(5);
    expect(figs.map(m=>m[1])).toEqual(images(original()).map(m=>m[1]));
    const oldPaths=figs.map(m=>m[1]);
    expect(images(body).map(m=>m[1]).filter(path=>oldPaths.includes(path))).toEqual(oldPaths);
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figs){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const asset=assets.find((x:any)=>x.asset===path);
      expect(asset,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path))),path).toBe(asset.sha256);
    }
  });
  it('distinguishes a stipulated redistribution, a pair loss and full-policy KL',()=>{
    const t=body.split('<a id="dpo-normalized-mass-kl"></a>')[1].split('<span id="16-варианты-dpo">')[0];
    for(const value of ['это не утверждение','конкретный шаг оптимизатора',
      '0.599675','0.534800','70%','Одиночная','из-за случайности',
      'Систематическое усреднение по другому распределению','https://arxiv.org/html/2305.18290v3#S3'])
      expect(t,value).toContain(value);
    const q=[.4,.4,.2],p=[.2,.1,.7],r=p.map((v,i)=>Math.log(v/q[i]));
    expect(Math.log1p(Math.exp(-.5*(r[0]-r[1])))).toBeCloseTo(.5347999967395703,14);
    expect(p.reduce((s,v,i)=>s+v*r[i],0)).toBeCloseTo(.5996752057227794,14);
    expect((r[0]+r[1])/2).toBeLessThan(0);
    expect(read('publishing/tools/check_dpo_examples.py')).toContain('F.kl_div(mass_ref.log(), mass_after');
  });
});
