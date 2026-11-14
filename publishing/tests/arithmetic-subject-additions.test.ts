import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-arithmetic-subject-additions.json';
const priorJournal='2026-11-04-roofline-block-additions.json';
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0],current=read(row.source);
const body=beforeTopicConsolidation(current,row.source,root,own);
const prior=beforeTopicConsolidation(current,row.source,root,priorJournal);
const removeAliases=(s:string)=>s.replace(/!\[\[([^\]|]+)\|[^\]]+\]\]/g,'![[$1]]');

describe('arithmetic subject navigation preserves the complete accepted explanation',()=>{
  it('reverses only recorded metadata,anchor and alt additions in three files',()=>{
    expect(journal.files).toHaveLength(3);
    expect(hash(prior)).toBe('2d49342ee5b2998d93980151ff95f2fdbf5a464c1cbb6911f9e7c3c4dce0cafa');
    for(const f of journal.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,priorJournal))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,priorJournal)).toThrow();
    }
    const clean=(s:string)=>removeAliases(matter(s).content)
      .replace(/<a id="arith-[^"]+"><\/a>\n\n/g,'');
    expect(clean(body)).toBe(matter(prior).content);
    expect(body.match(/!\[\[/g)).toHaveLength(9);
    expect(body.match(/!\[\[[^\]|]+\|[^\]]+\]\]/g)).toHaveLength(9);
    expect(body.match(/^```python$/gm)).toHaveLength(1);
  });
  it('maps31 actual subjects in both indexes without claiming an English translation',()=>{
    const meta=matter(body).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.status).toBe('draft');
    expect(meta.reference_terms).toHaveLength(31);
    expect(new Set(meta.reference_terms.map((t:any)=>t.anchor)).size).toBe(31);
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(body.match(/id="cs336-systems-arithmetic"/g)).toHaveLength(1);
    expect(body).toContain('03 Измерение производительности и roofline|');
    expect(body).toContain('05 Численные форматы и mixed precision|');
  });
});
