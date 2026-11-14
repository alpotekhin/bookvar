import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-pretext-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const chapters=journal.files.slice(0,4);
const counts=[
  {terms:25,programs:14,figures:8,displays:22},
  {terms:9,programs:5,figures:4,displays:2},
  {terms:10,programs:6,figures:2,displays:3},
  {terms:11,programs:6,figures:3,displays:2},
];
const programs=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const mathTokens=(s:string)=>s.replace(/\\begin\{aligned\}|\\end\{aligned\}|\\\\|\\qquad|&|\s/g,'');
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);

describe('pretext and semi-supervised subject-route alignment',()=>{
  it('retains all gate decisions,masked losses and reported protocol values in the narrow tables',()=>{
    const text=read(chapters[0].source);
    const rows=(header:string)=>text.slice(text.indexOf(header)).split('\n').slice(2)
      .filter((line:string,index:number,lines:string[])=>line.startsWith('|')&&!lines.slice(0,index).some(l=>!l.startsWith('|')))
      .map((line:string)=>line.split('|').slice(1,-1).map(c=>c.trim()));
    const weak=rows('| Row | Weak $q$ |'),strong=rows('| Example | Strong $p(0)$ |');
    expect(weak).toHaveLength(4);
    expect(strong).toHaveLength(4);
    const probabilities=[[.97,.03],[.75,.25],[.08,.92],[.95,.05]],p0=[.6,.4,.8,.2];
    for(let i=0;i<4;i++){
      const target=probabilities[i][0]>=probabilities[i][1]?0:1;
      const accepted=Math.max(...probabilities[i])>=.95;
      expect(weak[i][1].match(/[\d.]+/g)!.map(Number)).toEqual(probabilities[i]);
      expect(Number(weak[i][2])).toBe(target);
      expect(weak[i][3]).toBe(accepted?'yes':'no');
      expect(Number(strong[i][1])).toBe(p0[i]);
      expect(strong[i][2]).toBe(accepted?'$-\\log'+(target===0?p0[i]:1-p0[i])+'$':'0');
    }
    const results=rows('| Labels | RandAugment | CTAugment |');
    expect(results).toEqual([
      ['40','$13.81\\pm3.37$','$11.39\\pm3.35$'],
      ['250','$5.07\\pm0.65$','$5.07\\pm0.33$'],
      ['4,000','$4.26\\pm0.05$','$4.31\\pm0.15$'],
    ]);
  });
  it('reverses all six files to their exact preceding revision',()=>{
    expect(journal.files).toHaveLength(6);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(current),row.source).toBe(row.after_sha256);
      let before=current;
      for(const change of [...row.changes].reverse()){
        expect(before.split(change.new).length,row.source).toBe(2);
        before=before.replace(change.new,()=>change.old);
      }
      expect(hash(before),row.source).toBe(row.before_sha256);
    }
  });
  for(const [i,row] of chapters.entries()){
    it('retains every explanation,program,equation and original figure in '+row.source,()=>{
      const current=read(row.source);
      let original=current;
      for(const change of [...row.changes].reverse())
        original=original.replace(change.new,()=>change.old);
      expect(hash(original)).toBe(row.before_sha256);
      expect(programs(current)).toEqual(programs(original));
      expect(programs(current)).toHaveLength(counts[i].programs);
      expect(formulas(current).map(mathTokens)).toEqual(formulas(original).map(mathTokens));
      expect(formulas(current)).toHaveLength(counts[i].displays);
      expect(figures(current)).toEqual(figures(original));
      expect(figures(current)).toHaveLength(counts[i].figures);
      let body=matter(current).content;
      for(const change of [...row.changes].reverse())
        if(!change.old.startsWith('last_updated:'))body=body.replace(change.new,()=>change.old);
      expect(body).toBe(matter(original).content);
      expect(matter(current).data.source_language).toBe('en');
    });
    it('resolves every subject to an actual heading in both indexes for '+row.source,()=>{
      const page=matter(read(row.source));
      const terms=referenceTerms(row.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
      expect(terms).toHaveLength(counts[i].terms);
      expect(new Set(terms.map(t=>t.anchor)).size).toBe(terms.length);
      for(const [path,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
        for(const term of terms)
          expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    });
  }
  it('checks the original file bytes for all seventeen figure placements',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const placements=chapters.flatMap((row:any)=>figures(read(row.source)));
    expect(placements).toHaveLength(17);
    expect(new Set(placements.map((p:string)=>p.replace(/^02 Areas\/ML & DL\//,''))).size).toBe(16);
    for(const path of placements){
      const asset=path.replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
});
