import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {loadManifest} from '../adapter/manifest.js';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-kernel-compiler-split.json';
const audit=JSON.parse(read('docs/audit/'+own));
const kernel='00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const compiler='00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs.md';
const original=audit.files.find((r:any)=>r.source===kernel).changes[0].old;
const currentKernel=read(kernel),currentCompiler=read(compiler);
const select=(s:string)=>s.slice(s.indexOf(audit.moved.start),s.indexOf(audit.moved.end)+audit.moved.end.length);
const python=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
const sources=(s:string)=>[...s.matchAll(/<!-- BEGIN SOURCE: ([^>]+) -->([\s\S]*?)<!-- END SOURCE: \1 -->/g)]
  .map(m=>[m[1],m[2]]);

describe('kernels and graph compilation have separate canonical owners',()=>{
  it('binds every changed content object to an exact reversible journal',()=>{
    expect(audit.files).toHaveLength(12);
    for(const row of audit.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(current),row.source).toBe(row.after_sha256);
      if(row.created)continue;
      let before=current;
      for(const change of [...row.changes].reverse()){
        expect(before.split(change.new).length,row.source).toBe(2);
        before=before.replace(change.new,()=>change.old);
      }
      expect(hash(before),row.source).toBe(row.before_sha256);
    }
  });
  it('moves all1375 lines byte-for-byte apart from fifteen explicit navigation aliases',()=>{
    let moved=select(currentCompiler);
    for(const anchor of audit.moved.addedAliases)
      moved=moved.replace('<a id="'+anchor+'"></a>\n\n','');
    expect(moved).toBe(select(original));
    expect(python(currentCompiler)).toEqual(python(select(original)));
    expect(python(currentCompiler)).toHaveLength(35);
    expect(python(currentKernel)).toHaveLength(10);
    expect([...python(currentKernel),...python(currentCompiler)]).toEqual(python(original));
    expect([...sources(currentKernel),...sources(currentCompiler)]).toEqual(sources(original));
    expect([...images(currentKernel),...images(currentCompiler)]).toEqual(images(original));
    expect(images(currentKernel)).toHaveLength(11);
    expect(images(currentCompiler)).toHaveLength(5);
  });
  it('keeps old fragment destinations as exact links instead of duplicating their explanation',()=>{
    for(const anchor of audit.moved.compatibilityAnchors)
      expect(currentKernel).toContain('<a id="'+anchor+'"></a>');
    expect(currentKernel).not.toContain('<!-- BEGIN REVIEW: JAX runtime');
    expect(currentKernel).not.toContain('<!-- BEGIN REVIEW: CUDA Graphs');
    for(const anchor of ['torch-compile-graphs-guards-recompilation','cuda-graphs-capture-replay',
      'Tracing a data-dependent branch','Device placement and collective communication'])
      expect(currentKernel).toContain('00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#'+anchor+'|');
  });
  it('indexes12 kernel subjects and17 compiler subjects with unique actual anchors in both locales',()=>{
    for(const [path,n] of [[kernel,12],[compiler,17]] as const){
      const p=matter(read(path)),terms=referenceTerms(path,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
      expect(terms).toHaveLength(n);
      for(const [index,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
        for(const term of terms)
          expect(read(index)).toContain('[['+path.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    }
  });
  it('partitions source units and updates original image provenance without modifying images',()=>{
    const oldUnits=matter(original).data.source_unit_id;
    const currentUnits=[...matter(currentKernel).data.source_unit_id,...matter(currentCompiler).data.source_unit_id];
    expect(currentUnits).toHaveLength(14);
    expect(new Set(currentUnits)).toEqual(new Set(oldUnits));
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of images(currentCompiler)){
      const path=image.replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===path);
      expect(rows).toHaveLength(1);
      expect(rows[0].used_in).toContain(compiler);
      expect(rows[0].used_in).not.toContain(kernel);
      expect(hash(readFileSync(resolve(root,path)))).toBe(rows[0].sha256);
    }
    const coverage=JSON.parse(read('05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml'));
    const entries=coverage.rows;
    const row=entries.find((r:any)=>r.source_unit==='assignment-02-task-torch-compile');
    expect(row.destination).toBe(compiler);
    expect(row.destination_anchor).toBe('torch-compile-graphs-guards-recompilation');
  });
  it('places the new chapter after Triton within the same thematic route',()=>{
    const m=loadManifest(resolve(root,'publishing/navigation.yml'));
    const pages=m.sections.find(s=>s.id==='textbook')!.pages;
    const at=pages.findIndex(p=>p.source===kernel);
    expect(pages[at+1].source).toBe(compiler);
    expect(pages[at+1].route).toBe('textbook/ml-systems/graph-compilation-pytorch-jax');
    for(const index of ['00 Учебник/_index.md','en/00 Textbook/_index.md'])
      expect(read(index)).toContain('00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs|');
  });
});

