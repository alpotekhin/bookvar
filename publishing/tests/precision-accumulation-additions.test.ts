import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-precision-accumulation-additions.json';
const stop='2026-11-04-arithmetic-subject-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const body=read(row.source);
const prior=beforeTopicConsolidation(body,row.source,root,stop);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const programs=(s:string)=>[...s.matchAll(/^(?:```|~~~)python\n([\s\S]*?)^(?:```|~~~)\s*$/gm)].map(x=>x[1]);
const formulas=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);

describe('precision accumulation and quantization orientation',()=>{
  it('restores the complete421-line explanation and preceding indexes/registry/notice',()=>{
    expect(journal.files).toHaveLength(5);
    expect(hash(prior)).toBe('9a72e3c171518af749c26271e183bf35c55d2f98f76a146cf93c8d8404efa02d');
    for(const f of journal.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(programs(prior)).toHaveLength(2);
    expect(programs(body)).toHaveLength(4);
    expect(programs(body)).toEqual(expect.arrayContaining(programs(prior)));
    expect(images(prior)).toHaveLength(6);
    expect(images(body)).toHaveLength(7);
    expect(images(body)).toEqual(expect.arrayContaining(images(prior)));
    expect(formulas(body)).toHaveLength(formulas(prior).length+1);
    expect(formulas(body)).toEqual(expect.arrayContaining(formulas(prior)));
    expect(body).not.toContain('\\(');
  });
  it('connects concrete gradient and quantization states to the actual API limits',()=>{
    for(const fact of ['1024','960.5','544','8.5','4.25','0.075,0.1','1/S',
      'GradScaler("cpu", init_scale=128.0)','torch.float8_e4m3fn',
      'rowwise[0, 1].item() == 0.0','columnwise[0, 1].item() == 1.0',
      'nonzero finite groups','hardware-packed','не проверяет произвольно изменённые градиенты повторно'])
      expect(body).toContain(fact);
    expect(body).not.toContain('Mixed Precision Training — справочная карточка');
    expect(body).toContain('650×450');
    expect(body).toContain('| Position | Input | By row | By column |');
    const manifest=JSON.parse(read('publishing/tools/precision-accumulation-sources.json'));
    expect(body).toContain('> '+manifest.quote);
  });
  it('maps21 actual subjects in both indexes and preserves canonical-language status',()=>{
    const meta=matter(body).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.status).toBe('draft');
    expect(meta.reference_terms).toHaveLength(21);
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(body.match(/id="cs336-systems-precision"/g)).toHaveLength(1);
    expect(body).toContain('04 Арифметика Transformer и MoE|');
    expect(body).toContain('06 Data pipeline, padding и packing|');
  });
});
