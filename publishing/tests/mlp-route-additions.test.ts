import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-mlp-route-additions.json', stop='2026-11-03-kernels-course-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)), row=journal.files[0];
const raw=read(row.source);
const current=beforeTopicConsolidation(raw,row.source,root,own);
const prior=beforeTopicConsolidation(raw,row.source,root,stop);
const implementation=/<!-- BEGIN REVIEW: Module state 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: Module hooks 2026-10-27 -->\n\n/;

describe('continuous MLP route and original source provenance',()=>{
  it('restores only the five exact preceding artifacts',()=>{
    expect(row.before_sha256).toBe('17e763a94b26580e6c4d015c6ab092c1d33d57a9f6c77e34a598fdc27379507b');
    expect(journal.files).toHaveLength(5);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });
  it('preserves all eleven programs, four image paths and complete English implementation prose',()=>{
    const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
    const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(programs(current)).toHaveLength(11);
    expect(programs(current)).toEqual(programs(prior));
    expect(figures(current)).toHaveLength(4);
    expect(figures(current)).toEqual(figures(prior));
    const old=prior.match(implementation)![0], changed=current.match(implementation)![0]
      .replace(/<a id="pytorch-module-(?:state|hooks)"><\/a>\n/g,'')
      .replace(/^(#{3,4}) /gm,(_,marks)=>marks.slice(1)+' ');
    expect(changed).toBe(old);
    expect(current.indexOf('## MLP внутри Transformer')).toBeLessThan(current.indexOf('<!-- BEGIN REVIEW: Module state'));
    expect(current.indexOf('## Ширина, глубина и число параметров')).toBeLessThan(current.indexOf('## MLP внутри Transformer'));
    expect(current.indexOf('Продолжение основного курса:')).toBeLessThan(current.indexOf('## Дополнение:'));
  });
  it('indexes eight canonical topics in both locales without creating another explanation',()=>{
    const stem=row.source.replace(/\.md$/,'');
    for(const anchor of ['neuron-affine-activation','mlp-batch-shapes','mlp-xor','mlp-training-step',
      'mlp-width-depth-parameters','mlp-positionwise-ffn','pytorch-module-state','pytorch-module-hooks']){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+stem+'#'+anchor+'|');
    }
    expect(current.match(/\/licenses\/cs231n-MIT\.txt/g)).toHaveLength(3);
    expect(journal.sourceComparison.fullyRead).toHaveLength(5);
  });
});
