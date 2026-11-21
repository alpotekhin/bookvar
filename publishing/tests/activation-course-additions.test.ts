import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-activation-course-additions.json',stop='2026-11-03-mlp-route-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const raw=read(row.source),current=beforeTopicConsolidation(raw,row.source,root,own);
const prior=beforeTopicConsolidation(raw,row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);

describe('activation mechanism additions and canonical reference',()=>{
  it('restores exact preceding chapter and both indexes, rejects unreviewed edits',()=>{
    expect(row.before_sha256).toBe('467efffdbfaad2a74bb9e31ff86fed9553cc0478664407efd34d2c678e4b7696');
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });
  it('preserves all prior programs, image paths, captions and equations',()=>{
    const old=programs(prior),now=programs(current);
    expect(old).toHaveLength(3);expect(now).toHaveLength(5);
    expect([now[0],now[1],now[4]]).toEqual(old);
    expect(figures(current)).toHaveLength(8);expect(figures(current)).toEqual(figures(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    for(const formula of math(prior))expect(math(current)).toContain(formula);
    const captions=(s:string)=>[...s.matchAll(/^\*[\s\S]*?\*\s*$/gm)].map(m=>m[0]);
    expect(captions(current)).toEqual(captions(prior));
    expect(programs(raw)).toEqual(programs(current));
    expect(figures(raw).slice(0, 8)).toEqual(figures(current));
    expect(figures(raw).at(-1)).toContain('goujon-maxout-figure6');
    for (const formula of math(current)) expect(math(raw)).toContain(formula);
    expect(current).toContain('max(dim=-1)');
    expect(journal.remaining[0].status).toBe('linked-not-reproduced');
  });
  it('indexes twelve mechanisms in both locales, preserving one canonical owner',()=>{
    const anchors=["activation-chain-rule","sigmoid-saturation","sigmoid-bce-logits","tanh-centering","relu-inactive-leaky","elu-slopes","gelu-exact-approximation","silu-swish","swiglu-branch-gradients","maxout-affine-branches","activation-moments","activation-output-logits"];
    for(const anchor of anchors){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+anchor+'|');
    }
    expect(journal.sourceComparison.fullyRead).toHaveLength(4);
  });
});

