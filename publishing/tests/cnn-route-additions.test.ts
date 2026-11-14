import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-cnn-route-additions.json',stop='2026-11-03-augmentation-composition-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const marked=(s:string,start:string,end:string)=>s.split(start)[1].split(end)[0];

describe('CNN core route and worked causal convolution',()=>{
  it('restores the exact preceding owner and both reference indexes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
  });
  it('retains all twenty-one original programs, fifteen figures and both full extensions',()=>{
    expect(programs(current)).toHaveLength(22);
    expect(programs(current)).toEqual(expect.arrayContaining(programs(prior)));
    const figures=(s:string)=>[...s.matchAll(/!\[\[[^\]]+\]\]/g)].map(m=>m[0]).sort();
    expect(figures(current)).toHaveLength(15);
    expect(figures(current)).toEqual(figures(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]).sort();
    expect(math(current)).toEqual(math(prior));
    for(const [start,end]of [
      ['<!-- group-equivariance 2026-10-26 -->','<!-- /group-equivariance -->'],
      ['<!-- BEGIN REVIEW: Winograd 2026-10-27 -->','<!-- END REVIEW: Winograd 2026-10-27 -->']]){
      const stripAnchors=(s:string)=>s.replace(/<a id="cnn-[^"]+"><\/a>\n\n/g,'');
      expect(stripAnchors(marked(current,start,end))).toBe(marked(prior,start,end));
    }
    const positions=['## Pooling','## Receptive field','## Свёртка как матричное умножение',
      '## LeNet:','## AlexNet, VGG','## ResNet\n','## Свёртки в последовательностях',
      '## Проверка собственной реализации','## Rotation-equivariant feature maps','## Winograd:','## Источники']
      .map(s=>current.indexOf(s));
    expect(positions.every(p=>p>=0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a,b)=>a-b));
    expect(current.indexOf('Основной курс продолжается')).toBeLessThan(current.indexOf('## Rotation-equivariant'));
  });
  it('indexes nineteen mechanisms and explains the current-target leakage case',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(19);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+term.anchor+'|');
    }
    for(const phrase of ['[100., 210., 321., 432., 543.]','[1., 10., 100., 0., 0.]',
      '[100., 200., 310., 420., 531.]','dilation * (weight.shape[-1] - 1)',
      'модель увидит целевое значение','https://arxiv.org/html/1609.03499v1#S2.SS1'])
      expect(current).toContain(phrase);
  });
});
