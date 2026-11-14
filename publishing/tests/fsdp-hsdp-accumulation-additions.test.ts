import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-fsdp-hsdp-accumulation-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owner=journal.files[0];
const programs=(s:string)=>[...s.matchAll(/^~~~python\n([\s\S]*?)^~~~/gm)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('FSDP owner and HSDP accumulation',()=>{
  it('reverses all three content files to their exact preceding hashes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const row of journal.files){
      let text=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(text),row.source).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()){
        expect(text.split(c.new).length,row.source).toBe(2);
        text=text.replace(c.new,()=>c.old);
      }
      expect(hash(text),row.source).toBe(row.before_sha256);
    }
  });
  it('retains all prior explanations,eight programs,equations,six originals and anchors',()=>{
    const text=read(owner.source);let original=text;
    for(const c of [...owner.changes].reverse())original=original.replace(c.new,()=>c.old);
    expect(hash(original)).toBe('183de5525e52384c034fc3996acbd74d20dad9a64560f7a65f24406c15da0ac8');
    expect(programs(text).slice(0,8)).toEqual(programs(original));
    expect(programs(text)).toHaveLength(9);
    expect(formulas(text).slice(0,-1)).toEqual(formulas(original));
    expect(formulas(text)).toHaveLength(4);
    expect(figures(text).map(m=>m[1])).toEqual(figures(original).map(m=>m[1]));
    expect(figures(text)).toHaveLength(6);
    expect(figures(text).every(m=>m[2]?.trim())).toBe(true);
    expect(matter(text).data.source_unit_id).toEqual(matter(original).data.source_unit_id);
    for(const [,a]of original.matchAll(/<a id="([^"]+)"/g))expect(text).toContain('<a id="'+a+'"');
  });
  it('resolves26 unique subjects into the same owner from RU and EN indexes',()=>{
    const page=matter(read(owner.source));
    const terms=referenceTerms(owner.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
    expect(terms).toHaveLength(26);
    expect(new Set(terms.map(t=>t.anchor)).size).toBe(26);
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const term of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
  });
  it('retains all six exact source-image files and their existing rights records',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const image of figures(read(owner.source))){
      const asset=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
  it('separates HSDP reduction axes,loss scaling and weight lifetime in the new worked example',()=>{
    const text=read(owner.source).split('<a id="fsdp-hsdp-accumulation"></a>')[1]
      .split('<a id="fsdp-custom-correctness"></a>')[0];
    for(const value of ['set_requires_all_reduce(False)','set_requires_gradient_sync',
      'set_reshard_after_backward','G_0=[12,18]','G_1=[36,42]','[6,7.5]',
      'scale = dist.get_world_size(data_group) / count.item()',
      'model.set_requires_gradient_sync(True)',
      'model.set_requires_all_reduce(index == len(window) - 1)',
      'Коллективный\nподсчёт числа целей','пик GPU-памяти проверяются отдельно'])expect(text).toContain(value);
    expect(programs(text)).toHaveLength(1);
    const checker=read('publishing/tools/check_fsdp_examples.py');
    for(const value of ['for length in (1, 2, 3)','for update in range(2)',
      'compare(model, reference, optimizer, reference_optimizer)',
      'partially_synchronized_accumulation(rank, namespace, hybrid_mesh)',
      '[[12.], [18.], [36.], [42.]]','accumulated /= 8'])expect(checker).toContain(value);
  });
});
