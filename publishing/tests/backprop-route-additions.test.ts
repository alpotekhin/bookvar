import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-backprop-route-additions.json',stop='2026-11-03-activation-course-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const raw=read(row.source),current=beforeTopicConsolidation(raw,row.source,root,own);
const prior=beforeTopicConsolidation(raw,row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
const marker=(name:string)=>new RegExp('<!-- BEGIN REVIEW: '+name+' 2026-10-27 -->\\n([\\s\\S]*?)<!-- END REVIEW: '+name+' 2026-10-27 -->\\n\\n');
const plain=(s:string)=>s.replace(/<a id="[^"]+"><\/a>\n/g,'')
  .replace(/!\[\[([^\]|]+)\|[^\]]+\]\]/g,'![[$1]]');

describe('complete backpropagation reading route',()=>{
  it('restores the exact preceding owner and both reference indexes',()=>{
    expect(row.before_sha256).toBe('8795146abe1f9c7a8de8a232c958aae472e3788d963f99aba557b90690fee2c0');
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });
  it('retains every program, source figure, caption and displayed equation',()=>{
    expect(programs(current)).toHaveLength(20);
    expect([...programs(current)].sort()).toEqual([...programs(prior)].sort());
    expect(figures(current)).toHaveLength(7);expect(figures(current)).toEqual(figures(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]).sort();
    const captions=(s:string)=>[...s.matchAll(/^\*[\s\S]*?\*\s*$/gm)].map(m=>m[0]);
    expect(math(current)).toEqual(math(prior));expect(captions(current)).toEqual(captions(prior));
    expect(programs(raw)).toEqual(programs(current));
    expect(figures(raw)).toEqual(figures(current));
    for(const formula of math(current))expect(math(raw)).toContain(formula);
    for(const name of ['forward mode','JAX transforms','custom backward']){
      expect(plain(current.match(marker(name))![1])).toBe(prior.match(marker(name))![1]);
      expect(raw.match(marker(name))![1]).toBe(current.match(marker(name))![1]);
    }
  });
  it('finishes reverse-mode failure analysis and gradcheck before advanced transforms',()=>{
    const before=(a:string,b:string)=>expect(current.indexOf(a)).toBeLessThan(current.indexOf(b));
    before('## 9. Автоматическое','## 10. Ошибки');before('## 10. Ошибки','## 11. Как проверить');
    before('## 11. Как проверить','Далее по основному курсу:');
    before('Далее по основному курсу:','## 12. Прямой режим');
    before('## 12. Прямой режим','<!-- BEGIN REVIEW: forward mode');
    before('<!-- END REVIEW: JAX transforms','<!-- BEGIN REVIEW: custom backward');
  });
  it('indexes seventeen mechanisms into the canonical chapter in both locales',()=>{
    for(const anchor of ["backprop-chain-rule","backprop-shared-nodes","affine-layer-backward","micrograd-topological-backward","backward-versus-step","gradient-accumulation-weighted","reverse-vjp","autograd-gradient-stability","autograd-invalid-operations","autograd-recorded-graph","finite-difference-gradcheck","forward-jvp-dual","forward-reverse-traces","hessian-vector-product","gradient-function-transforms","custom-backward-contract","custom-forward-transform-rules"]){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+anchor+'|');
    }
  });
});

