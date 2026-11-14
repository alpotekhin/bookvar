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
const own='2026-11-05-ddp-checkpoint-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const owners=journal.files.slice(0,2);
const counts=[{subjects:33,programs:15,figures:14,displays:16},
  {subjects:21,programs:9,figures:5,displays:6}];
const programs=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('DDP and checkpointing subject alignment',()=>{
  it('retains every send count,row total and receive total in the narrow matrix',()=>{
    const text=read(owners[0].source);
    const table=text.slice(text.indexOf('| Source | To 0 |')).split('\n').slice(0,7);
    expect(table[0]).toBe('| Source | To 0 | To 1 | To 2 | To 3 | Sent |');
    const rows=table.slice(2).map(line=>line.split('|').slice(1,-1).map(x=>x.trim()));
    const matrix=[[2,2,1,1],[3,2,2,2],[2,1,1,1],[2,2,2,1]];
    for(let rank=0;rank<4;rank++){
      expect(rows[rank][0]).toBe(String(rank));
      expect(rows[rank].slice(1,5).map(Number)).toEqual(matrix[rank]);
      expect(Number(rows[rank][5])).toBe(matrix[rank].reduce((a,b)=>a+b,0));
    }
    expect(rows[4]).toEqual(['Total','9','7','6','5','27']);
    expect(matrix[0].map((_,c)=>matrix.reduce((sum,row)=>sum+row[c],0)))
      .toEqual(rows[4].slice(1,5).map(Number));
  });
  it('reverses all six files to their exact prior revision',()=>{
    expect(journal.files).toHaveLength(6);
    for(const row of journal.files){
      let current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(current),row.source).toBe(row.after_sha256);
      for(const change of [...row.changes].reverse()){
        expect(current.split(change.new).length,row.source).toBe(2);
        current=current.replace(change.new,()=>change.old);
      }
      expect(hash(current),row.source).toBe(row.before_sha256);
    }
  });
  for(const [i,row] of owners.entries()){
    it('retains all prior explanations,programs,equations and source figures in '+row.source,()=>{
      const text=beforeTopicConsolidation(read(row.source),row.source,root,own);
      let original=text;
      for(const change of [...row.changes].reverse())
        original=original.replace(change.new,()=>change.old);
      expect(hash(original)).toBe(row.before_sha256);
      const actual=programs(text);
      expect(i===0?actual:actual.filter((_,n)=>n!==7)).toEqual(programs(original));
      expect(actual).toHaveLength(counts[i].programs);
      expect(formulas(text)).toEqual(formulas(original));
      expect(formulas(text)).toHaveLength(counts[i].displays);
      expect(figures(text).map(m=>m[1])).toEqual(figures(original).map(m=>m[1]));
      expect(figures(text)).toHaveLength(counts[i].figures);
      expect(figures(text).every(m=>m[2]?.trim())).toBe(true);
      if(i===0)expect(text).toContain('collective-primitives-overview.svg|Входы, выходы и размещение результатов четырёх коллективных операций');
      expect(matter(text).data.source_unit_id).toEqual(matter(original).data.source_unit_id);
      for(const [,anchor] of original.matchAll(/<a id="([^"]+)"/g))
        expect(text).toContain('<a id="'+anchor+'"');
    });
    it('resolves all subjects in both indexes without inventing an English translation for '+row.source,()=>{
      const text=read(row.source),page=matter(text);
      const subjects=referenceTerms(row.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
      expect(subjects).toHaveLength(counts[i].subjects);
      expect(new Set(subjects.map(t=>t.anchor)).size).toBe(subjects.length);
      for(const [path,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
        for(const subject of subjects)
          expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+subject.anchor+'|'+(en?subject.label_en:subject.label)+']]');
      expect(page.data.source_language).not.toBe('en');
    });
  }
  it('pins all nineteen original figure files to their existing registry hashes',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const placements=owners.flatMap((r:any)=>figures(read(r.source)).map(m=>m[1]));
    expect(placements).toHaveLength(19);
    for(const path of placements){
      const asset=path.replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
  it('explains and checks the selective-cache hook route without claiming GPU offload',()=>{
    const text=read(owners[1].source).split('<a id="checkpoint-sac-hooks"></a>')[1]
      .split('<a id="checkpoint-transfer-time"></a>')[0];
    for(const statement of ['в собственном кеше','[3,4]','[4,5]','[3,5]',
      'respect_saved_tensors_hooks=False','передаётся в `pack`',
      'в `unpack`','120 байт','не экономию GPU-памяти',
      'по-прежнему не равна пику памяти','без\nCUDA-передач',
      'torch/utils/checkpoint.py#L1545-L1651'])expect(text).toContain(statement);
    expect(programs(text)).toHaveLength(1);
    const checker=read('publishing/tools/check_checkpoint_offload_examples.py');
    expect(checker).toContain('execute("checkpoint-sac-hooks")');
    expect(checker).toContain('preactivation.cos() @ weight.T');
    expect(checker).toContain('input_value.T @ preactivation.cos()');
    expect(checker).toContain('shape_policy_cases');
  });
});
