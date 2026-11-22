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
const own='2026-11-06-embeddings-gradcache-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/'+own));
const row=journal.files[0],body=read(row.source);
const undo=(s:string,r:any)=>{for(const c of [...r.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const original=undo(beforeTopicConsolidation(body,row.source,root,own),row);
const programs=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```/gm)].map(m=>m[1]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('full embeddings explanation and GradCache addition',()=>{
  it('inverts five exact owner/index/navigation changes and rejects drift',()=>{
    expect(journal.files).toHaveLength(5);
    for(const r of journal.files){
      const s=beforeTopicConsolidation(read(r.source),r.source,root,own);
      expect(hash(s)).toBe(r.after_sha256);
      expect(hash(undo(s,r))).toBe(r.before_sha256);
      expect(()=>beforeTopicConsolidation(s+'\nunreviewed\n',r.source,root,
        '2026-11-05-synthetic-subject-additions.json')).toThrow();
    }
  });
  it('keeps all three earlier programs and eleven derivations intact',()=>{
    expect(hash(original)).toBe('bc9d7e31337190371294760004dbb10b113fedfdb161aeda2a790631d699d1fd');
    const code=programs(body),oldCode=programs(original);
    expect(code).toHaveLength(4);
    expect(code.filter(x=>!x.includes('def cached_encoder_grad'))).toEqual(oldCode);
    expect(math(body)).toHaveLength(12);
    expect(math(body).filter(x=>!x.includes('J_i^\\top'))).toEqual(math(original));
    expect(body).toContain('The short example does not implement that mechanism');
    expect(body).toContain('Training-mode BatchNorm');
    expect(body).toContain('O(B^2)');
  });
  it('links21 explicit subjects in each locale and marks mixed source language',()=>{
    const p=matter(body);
    const terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(21);
    expect(p.data.source_language).toBe('mixed');
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)
        expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('preserves all eight original figures, captions and source notices',()=>{
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const images=figures(body);
    const originalPaths=figures(original).map(m=>m[1]);
    expect(originalPaths).toHaveLength(8);
    expect(images.map(m=>m[1]).filter(path=>originalPaths.includes(path))).toEqual(originalPaths);
    for(const image of images){
      expect(image[2]?.trim()).toBeTruthy();
      const path=image[1].replace(/^02 Areas\/ML & DL\//,'');
      const asset=assets.find((a:any)=>a.asset===path);
      expect(asset,path).toBeDefined();
      expect(hash(readFileSync(resolve(root,path)))).toBe(asset.sha256);
    }
    for(const caption of original.split(/\n\n/).filter(p=>/^\*(?!\*)/.test(p)&&p.endsWith('*')))
      expect(body).toContain(caption.replace(/^\*Tong Gao, Xingcheng Yao, Danqi Chen,/,
        '*Tianyu Gao, Xingcheng Yao, Danqi Chen,'));
    const fixtures=JSON.parse(read('publishing/fixtures/gradcache-906f0383/sources.json'));
    expect(fixtures.commit).toBe('906f03835fbc183132a9db32612a9e8f180ca3b4');
    for(const f of fixtures.files)
      expect(hash(readFileSync(resolve(root,'publishing/fixtures/gradcache-906f0383',f.file)))).toBe(f.sha256);
  });
});
