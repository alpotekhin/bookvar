import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
import {beforeReaderRouteNavigation} from '../tools/reader-route-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-gan-gradient-additions.json',stop='2026-11-03-autoencoder-course-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);

describe('GAN gradient-path explanation and subject links',()=>{
  it('restores the complete owner and both indexes, then the exact historical source review boundary',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('30f6c8d2da4c7fcf3607ef4649eb74e077286b5e1e538c18e7cad6dee3791874');
    expect(sha(beforeReaderRouteNavigation(read(row.source),row.source,root)))
      .toBe('a55940b1d05c1544877d2779656b388a769edb0fdd07996f2275865f8aa86c98');
  });
  it('preserves all fourteen prior programs, nine display equations and five original figures',()=>{
    expect(programs(current)).toHaveLength(15);
    expect(programs(current).filter(x=>!x.includes('frozen_routes'))).toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
    expect(math(prior)).toHaveLength(9);
    expect(math(current)).toHaveLength(10);
    expect(math(current)).toEqual(expect.arrayContaining(math(prior)));
    expect(images(current)).toEqual(images(prior));
    expect(images(current)).toHaveLength(5);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const prefixed of images(current)){
      const path=prefixed.replace(/^02 Areas\/ML & DL\//,'');
      const entry=registry.find((x:any)=>x.asset===path);
      expect(entry).toBeTruthy();
      expect(sha(readFileSync(resolve(root,path)))).toBe(entry.sha256);
    }
  });
  it('indexes eighteen mechanisms in both languages without replacing original English with a translation',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('en');
    expect(meta.reference_terms).toHaveLength(18);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(read('en/01 Reference/_index.md')).toContain('|GAN and DCGAN: generator and discriminator]] · **EN original**');
  });
  it('explains the full derivative, all four graph routes, old gradients and train/eval state',()=>{
    for(const phrase of ['(q-1)vz','-3.592126','−0.119738','frozen_routes',
      'no optimizer step',"restore D's parameter flags",
      'does not clear\ngradients left by an earlier backward pass',
      'eval()§ does not itself disable autograd'.replace('§',String.fromCharCode(96)),
      'three BatchNorm visits','not an image-quality demonstration'])
      expect(current).toContain(phrase);
    const position=current.indexOf('### Input gradients when discriminator weights are frozen');
    expect(position).toBeGreaterThan(current.indexOf('def update_G'));
    expect(position).toBeLessThan(current.indexOf('### Check parameter ownership'));
    expect(current).toContain('https://docs.pytorch.org/docs/2.14/notes/autograd.html#setting-requires-grad');
  });
});
