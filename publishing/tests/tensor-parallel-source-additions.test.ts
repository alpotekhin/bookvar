import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-tensor-parallel-source-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0];
const programs=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('TP/SP/CP source integration and subject alignment',()=>{
  it('reverses all four files to their exact prior hashes',()=>{
    expect(journal.files).toHaveLength(4);
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
  it('retains every previous explanation,example,equation,figure and legacy anchor',()=>{
    const text=read(owner.source);let original=text;
    for(const c of [...owner.changes].reverse())original=original.replace(c.new,()=>c.old);
    expect(hash(original)).toBe('19a681a8a334905e3c2b65b66757b4dcff58086e7c2e4c48a3717882767da30a');
    expect(programs(text).slice(0,3)).toEqual(programs(original));
    expect(programs(text)).toHaveLength(4);
    expect(formulas(text)).toEqual(formulas(original));
    expect(figures(text).map(m=>m[1])).toEqual(figures(original).map(m=>m[1]));
    expect(figures(text)).toHaveLength(6);
    expect(figures(text).every(m=>m[2]?.trim())).toBe(true);
    expect(matter(text).data.source_unit_id).toEqual(matter(original).data.source_unit_id);
    for(const [,anchor]of original.matchAll(/<a id="([^"]+)"/g))
      expect(text).toContain('<a id="'+anchor+'"');
  });
  it('connects all31 subjects to one owner in both reference indexes',()=>{
    const page=matter(read(owner.source));
    const terms=referenceTerms(owner.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
    expect(terms).toHaveLength(31);
    expect(new Set(terms.map(t=>t.anchor)).size).toBe(31);
    for(const [path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const term of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    expect(page.data.source_language).toBe('mixed');
  });
  it('preserves the three selected EDLS classes byte-for-byte with their MIT notice',()=>{
    const original=programs(read(journal.verification.sourceClass.source))
      .find(s=>s.startsWith('%%writefile tensor_parallel_mlp.py'))!
      .split('\n').slice(1).join('\n').split('\n\nif __name__')[0];
    const actual=programs(read(owner.source))[3];
    expect(actual).toBe(original);
    expect(hash(actual)).toBe('92414271c13361505eaa4929140e5c452b05a986808c22676c600ea3b1e29cb8');
    expect(read(journal.verification.sourceClass.licenseCopy)).toContain('Copyright (c) 2021 course authors');
    expect(read(owner.source)).toContain('/licenses/edls-e632aa89-MIT.txt');
  });
  it('pins all six unchanged original figures to registry hashes',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figures(read(owner.source))){
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===path);
      expect(rows.length,path).toBe(1);
      expect(hash(readFileSync(resolve(root,path))),path).toBe(rows[0].sha256);
    }
  });
  it('explains and independently checks the original boundary limitations',()=>{
    const text=read(owner.source).split('<a id="tp-edls-distributed-mlp"></a>')[1]
      .split('## Практика и первоисточники')[0];
    for(const statement of ['[3,4]','[6,4]','[4,6]','дважды','requires_grad=False',
      'не сохраняет состояние RNG','разные части весов не смешиваются',
      'полный и одинаковый','Ещё одна TP-сумма','группа процессов\nпо умолчанию'])
      expect(text).toContain(statement);
    const checker=read('publishing/tools/check_tensor_parallel_edls.py');
    for(const statement of ['dist.init_process_group("gloo"','nprocs=2','reductions.call_count == 2',
      'reference.backward(incoming)','close(actual_grad, expected)','not actual.requires_grad',
      'not torch.equal(masks[0], masks[1])','local_norm.weight.grad * size'])expect(checker).toContain(statement);
  });
});
