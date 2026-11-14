import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {parseMarkdownHeadings} from '../adapter/build.js';
import {referenceTerms} from '../adapter/reference-terms.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),own='2026-11-05-network-fault-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own)),owners=journal.files.slice(0,2);
const programs=(s:string)=>[...s.matchAll(/^(\x60{3}|~{3})([^\n]*)\n([\s\S]*?)^\1\s*$/gm)].map(m=>[m[2],m[3]]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$([\s\S]*?)\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];
const counts=[{code:8,math:4,figures:4,terms:36},{code:2,math:9,figures:6,terms:31}];

describe('network, checkpoint and recovery subject alignment',()=>{
  it('reverses four content files and the registry to their exact preceding hashes',()=>{
    expect(journal.files).toHaveLength(5);
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
  for(const [index,owner]of owners.entries()){
    it('preserves the full prior explanation, code, math and original figures: '+owner.source,()=>{
      const text=read(owner.source);let original=text;
      for(const c of [...owner.changes].reverse())original=original.replace(c.new,()=>c.old);
      expect(hash(original)).toBe(owner.before_sha256);
      expect(programs(text)).toEqual(programs(original));
      expect(programs(text).filter(p=>p[0]==='python')).toHaveLength(counts[index].code);
      expect(formulas(text)).toEqual(formulas(original));
      expect(formulas(text)).toHaveLength(counts[index].math);
      expect(figures(text).map(m=>m[1])).toEqual(figures(original).map(m=>m[1]));
      expect(figures(text)).toHaveLength(counts[index].figures);
      expect(figures(text).every(m=>m[2]?.trim())).toBe(true);
      expect(matter(text).data.source_unit_id).toEqual(matter(original).data.source_unit_id);
      for(const [,anchor]of original.matchAll(/<(?:a|span) id="([^"]+)"/g))
        expect(text).toContain('id="'+anchor+'"');
      if(index===1){
        const beforeRoute=original.replace(
          '← [[44g Network, storage и distributed checkpoints]] · Далее: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]',
          ()=>'← [[44g Network, storage и distributed checkpoints]] · Далее: [[01 SFT и instruction data]]');
        expect(hash(beforeRoute)).toBe('1d9f922994ef80300a2f796e620f1b66ad9faa80c66c97689731cdf46909900e');
      }
    });
    it('resolves both locale indexes to the same detailed owner: '+owner.source,()=>{
      const page=matter(read(owner.source));
      const terms=referenceTerms(owner.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
      expect(terms).toHaveLength(counts[index].terms);
      expect(new Set(terms.map(t=>t.anchor)).size).toBe(counts[index].terms);
      for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
        for(const term of terms)expect(read(path)).toContain('[['+owner.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    });
  }
  it('retains all ten source-image bytes and their existing rights records',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for(const owner of owners)for(const image of figures(read(owner.source))){
      const asset=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
  it('separates upload completion from caller-owned stager cleanup',()=>{
    const text=read(owners[0].source).split('<a id="checkpoint-stager-lifetime"></a>')[1]
      .split('<a id="checkpoint-process-group"></a>')[0];
    for(const value of ['async_stager','close()','upload_completion',
      'кэшированные хранилища тензоров','пул фоновых потоков',
      'библиотека выполняет очистку сама',
      'https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save'])
      expect(text).toContain(value);
  });
});

