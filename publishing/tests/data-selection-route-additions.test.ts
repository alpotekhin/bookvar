import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-04-data-selection-route-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const paths=["00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md","00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров.md","00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding.md"];
const counts=[{terms:16,programs:8,figures:4},{terms:10,programs:5,figures:2},{terms:12,programs:6,figures:3}];
const programs=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);

describe('whole-owner review of labelled,active and geometric data selection',()=>{
  it('transposes the coverage table without losing any nearest distance or radius',()=>{
    const source=read(paths[2]).split('<a id="coverage-trace"></a>')[1];
    const table=source.match(/\| Point [\s\S]*?\| Radius [^\n]+/)!;
    const rows=table[0].split('\n').slice(2).map(line=>line.split('|').slice(1,-1).map(c=>c.trim()));
    const points=[0,1,4,5,9,10];
    expect(rows.slice(0,6).map(row=>Number(row[0]))).toEqual(points);
    for(const [col,selected] of [[0,[0]],[1,[0,10]],[2,[0,10,5]]] as const){
      const distances=points.map(p=>Math.min(...selected.map(c=>Math.abs(p-c))));
      expect(rows.slice(0,6).map(row=>Number(row[col+1]))).toEqual(distances);
      expect(Number(rows[6][col+1])).toBe(Math.max(...distances));
    }
  });
  it('pins both reused MOOC originals and their public CC BY4.0 notice',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const pin='0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6';
    for(const [file,sha] of [
      ['clustering_quiz_kmeans_not_scaled.svg','04c1dd673b6b7690a0bc9e7c06358a43ec7b38e64c7cac401180bb9cf37ec2c8'],
      ['clustering_quiz_kmeans_scaled.svg','bd2aae6c6eca8e5951324b268c312dc52263a697e7de1d7b3040bc3df2310694']
    ]){
      const path='00 Учебник/Assets/Figures/curated/selection-metric-2026-11-04/'+(file.includes('not_scaled')?'mooc-kmeans-not-scaled.svg':'mooc-kmeans-scaled.svg');
      const row=registry.find((r:any)=>r.asset===path);
      expect(row.sha256).toBe(sha);
      expect(row.commit).toBe(pin);
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/'+pin+'/figures/'+file);
      expect(row.source_viewbox).toEqual([0,0,375.6672,325.18365]);
      expect(row.license).toBe('CC BY 4.0');
      expect(hash(readFileSync(resolve(root,row.public_license_copy)))).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    }
  });
  it('preserves all five preceding files with exact reversible navigation-only changes',()=>{
    expect(journal.files).toHaveLength(6);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(current),row.source).toBe(row.after_sha256);
      let before=current;
      for(const c of [...row.changes].reverse()){
        expect(before.split(c.new).length,row.source).toBe(2);
        before=before.replace(c.new,()=>c.old);
      }
      expect(hash(before),row.source).toBe(row.before_sha256);
    }
  });
  for(const [i,path] of paths.entries()){
    it('retains the complete explanation,examples,source figures and English in '+path,()=>{
      const row=journal.files.find((r:any)=>r.source===path),current=read(path);
      let original=current;
      for(const c of [...row.changes].reverse())original=original.replace(c.new,()=>c.old);
      expect(programs(current)).toEqual(programs(original));
      expect(programs(current)).toHaveLength(counts[i].programs);
      expect(formulas(current)).toEqual(formulas(original));
      const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
      const originalPaths=figures(current).map(p=>{
        const asset=p.replace(/^02 Areas\/ML & DL\//,'');
        const entry=registry.find((r:any)=>r.asset===asset);
        return asset.includes('/selection-metric-2026-11-04/')?entry.source_asset:p;
      });
      expect(originalPaths).toEqual(figures(original));
      expect(figures(current)).toHaveLength(counts[i].figures);
      expect(matter(current).data.source_language).toBe('en');
      // Only navigation, alts, exact image copies and the recorded table transpose.
      let body=matter(current).content;
      for(const c of [...row.changes].reverse())if(!c.old.startsWith('last_updated:'))
        body=body.replace(c.new,()=>c.old);
      expect(body).toBe(matter(original).content);
    });
    it('provides unique subject anchors and matching RU/EN index links in '+path,()=>{
      const page=matter(read(path));
      const terms=referenceTerms(path,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
      expect(terms).toHaveLength(counts[i].terms);
      expect(new Set(terms.map(t=>t.anchor)).size).toBe(terms.length);
      for(const [index,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
        for(const term of terms)
          expect(read(index)).toContain('[['+path.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    });
  }
  it('keeps every original figure bound to its recorded file bytes',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const path of paths)for(const image of figures(read(path))){
      const asset=image.replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
});

