import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-optimizer-route-additions.json';
const stop='2026-11-03-backprop-route-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60\x60\x60python\n([\s\S]*?)^\x60\x60\x60\s*$/gm)].map(m=>m[1]).sort();
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]).sort();

describe('optimizer chapter reading route',()=>{
  it('restores the exact preceding chapter and both reference indexes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
  });
  it('keeps all nine programs, thirteen source images and complete English source additions',()=>{
    expect(programs(current)).toHaveLength(9);expect(programs(current)).toEqual(programs(prior));
    expect(figures(current)).toHaveLength(13);expect(figures(current)).toEqual(figures(prior));
    for(const [name,date] of [['adaptive clipping','2026-10-29'],['optimizer memory','2026-10-28']]){
      const pattern=new RegExp('<!-- BEGIN REVIEW: '+name+' '+date+' -->\\n([\\s\\S]*?)<!-- END REVIEW: '+name+' '+date+' -->');
      const plain=(s:string)=>s.replace(/<a id="[^"]+"><\/a>\n/g,'').replace(/!\[\[([^\]|]+)\|[^\]]+\]\]/g,'![[$1]]');
      expect(plain(current.match(pattern)![1])).toBe(plain(prior.match(pattern)![1]));
    }
    const captions=(s:string)=>[...s.matchAll(/^\*(?!\*)[\s\S]*?\*(?=\n\n|$)/gm)].map(m=>m[0]).sort();
    expect(captions(current)).toEqual(captions(prior));
  });
  it('finishes the core loop before adaptive clipping and matrix methods',()=>{
    const order=['## Ограничение нормы градиента','## Смешанная точность',
      '### Контрольная точка должна продолжать тот же процесс','## Чеклист перед большим запуском',
      'Следующая глава:','## Адаптивные пороги ограничения градиента',
      '### AGC:','### AutoClip:','## Muon и SOAP:','## Optimizer memory:'];
    const offsets=order.map(s=>current.indexOf(s));
    expect(offsets.every(n=>n>=0)).toBe(true);
    expect(offsets).toEqual([...offsets].sort((a,b)=>a-b));
    expect(current.indexOf('<a id="adamw"></a>')).toBeGreaterThan(current.indexOf('### Adam:'));
    expect(current).toContain('до $4{,}68$ или $5{,}22$');
    expect(current).toContain('возможным исходам даёт $4{,}95$');
  });
  it('indexes the same twenty mechanisms in both locales',()=>{
    const anchors=["minibatch-sgd","learning-rate-curvature","momentum","nesterov","adagrad-rmsprop","adam-moments","adamw","optimizer-step-comparison","schedule-boundary","critical-batch-size","gradient-norm-clipping","optimizer-loss-scaling","optimizer-checkpoint","agc-unitwise","autoclip","matrix-optimizers","adafactor","optimizer-8bit-states","galore-projection","optimizer-memory-accounting"];
    for(const anchor of anchors){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+anchor+'|');
    }
  });
});
