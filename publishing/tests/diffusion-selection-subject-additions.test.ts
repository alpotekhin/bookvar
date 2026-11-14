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
const own='2026-11-05-diffusion-selection-subject-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const chapters=journal.files.slice(0,2);
const counts=[{terms:13,programs:4,figures:4,displays:8},
  {terms:20,programs:10,figures:5,displays:8}];
const programs=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);

describe('diffusion and selection-engineering subject alignment',()=>{
  it('preserves all four trajectory states and ordered tokens in the narrow table',()=>{
    const text=read(chapters[0].source);
    const table=text.slice(text.indexOf('| Reverse-process state |')).split('\n').slice(0,6);
    expect(table[0]).toBe('| Reverse-process state | Sequence (positions 1–4) |');
    const states=['Initial','One possible step','Later','Complete'];
    const tokens=[['[MASK]','[MASK]','[MASK]','[MASK]'],
      ['[MASK]','[MASK]','sleeps','[MASK]'],['The','[MASK]','sleeps','here'],
      ['The','cat','sleeps','here']];
    for(let i=0;i<4;i++){
      expect(table[i+2].split('|')[1].trim()).toBe(states[i]);
      expect([...table[i+2].matchAll(/`([^`]+)`/g)].map(m=>m[1])).toEqual(tokens[i]);
    }
  });
  it('reverses all four files to their exact preceding revision',()=>{
    expect(journal.files).toHaveLength(4);
    for(const row of journal.files){
      let text=beforeTopicConsolidation(read(row.source),row.source,root,own);
      expect(hash(text),row.source).toBe(row.after_sha256);
      for(const change of [...row.changes].reverse()){
        expect(text.split(change.new).length,row.source).toBe(2);
        text=text.replace(change.new,()=>change.old);
      }
      expect(hash(text),row.source).toBe(row.before_sha256);
    }
  });
  for(const [i,row] of chapters.entries()){
    it('retains every original program,equation,figure and surrounding explanation in '+row.source,()=>{
      const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
      let original=current;
      for(const change of [...row.changes].reverse())
        original=original.replace(change.new,()=>change.old);
      expect(hash(original)).toBe(row.before_sha256);
      const actual=programs(current);
      expect(i===0?actual:actual.filter((_,index)=>index!==7)).toEqual(programs(original));
      expect(actual).toHaveLength(counts[i].programs);
      expect(formulas(current)).toEqual(formulas(original));
      expect(formulas(current)).toHaveLength(counts[i].displays);
      expect(figures(current)).toEqual(figures(original));
      expect(figures(current)).toHaveLength(counts[i].figures);
      let body=matter(current).content;
      for(const change of [...row.changes].reverse())
        body=body.replace(change.new,()=>change.old);
      expect(body).toBe(matter(original).content);
      expect(matter(current).data.source_language).toBe('en');
    });
    it('links all subjects to their actual explanation in both indexes for '+row.source,()=>{
      const page=matter(read(row.source));
      const terms=referenceTerms(row.source,page.data.reference_terms,parseMarkdownHeadings(page.content,true));
      expect(terms).toHaveLength(counts[i].terms);
      expect(new Set(terms.map(t=>t.anchor)).size).toBe(terms.length);
      for(const [path,en] of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]] as const)
        for(const term of terms)
          expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|'+(en?term.label_en:term.label)+']]');
    });
  }
  it('checks all nine complete original image files against the asset registry',()=>{
    const registry=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const placements=chapters.flatMap((row:any)=>figures(read(row.source)));
    expect(placements).toHaveLength(9);
    for(const path of placements){
      const asset=path.replace(/^02 Areas\/ML & DL\//,'');
      const rows=registry.filter((r:any)=>r.asset===asset);
      expect(rows.length,asset).toBe(1);
      expect(hash(readFileSync(resolve(root,asset))),asset).toBe(rows[0].sha256);
    }
  });
  it('works through both update schedules and distinguishes their budgets',()=>{
    const section=read(chapters[1].source).split('<a id="echoed-optimizer-updates"></a>')[1]
      .split('<a id="selection-quality-measurement"></a>')[0];
    for(const explanation of ['their second gradient need not be the same',
      'The first echoed update has gradient $-1$',
      'The next evaluation of the same example has gradient $-0.5$',
      'their mean gradient is still $-1$',
      'two optimizer\nupdates for echoing and one for accumulation',
      'Summing instead of averaging','learning-rate schedule',
      'https://research.google/blog/speeding-up-neural-network-training-with-data-echoing/'])
      expect(section).toContain(explanation);
    expect(programs(section)).toHaveLength(1);
    const checker=read('publishing/tools/check_selection_engineering_examples.py');
    expect(checker).toContain('expected = 1 + (initial - 1) * (1 - rate) ** repeats');
    expect(checker).toContain('Fraction(3, 2), Fraction(2)');
    expect(checker).toContain('echoed_optimizer_step_cases');
  });
});
