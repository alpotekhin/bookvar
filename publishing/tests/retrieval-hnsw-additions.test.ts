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
const own='2026-11-06-retrieval-hnsw-additions.json';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const j=JSON.parse(read('docs/audit/'+own)),row=j.files[0],body=read(row.source);
const undo=(s:string,r:any)=>{for(const c of [...r.changes].reverse())s=s.replace(c.new,()=>c.old);return s;};
const old=undo(body,row);
const blocks=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```/gm)].map(m=>m[1]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)];

describe('retrieval explanation and HNSW base-layer reasoning',()=>{
  it('reverses only three hash-pinned chapter/index deltas and rejects drift',()=>{
    expect(j.files).toHaveLength(3);
    for(const r of j.files){
      const s=beforeTopicConsolidation(read(r.source),r.source,root,own);
      expect(hash(s)).toBe(r.after_sha256);
      expect(hash(undo(s,r))).toBe(r.before_sha256);
      expect(()=>beforeTopicConsolidation(s+'\nunreviewed\n',r.source,root,
        '2026-11-06-embeddings-gradcache-additions.json')).toThrow();
    }
  });
  it('retains six earlier programs and every old derivation, adding one search demo',()=>{
    expect(hash(old)).toBe('c2f49438a49ecdf7aac70a44605763ebdcf3311781fb4b10c4514470ceab25bc');
    expect(blocks(body)).toHaveLength(7);
    expect(blocks(body).filter(x=>!x.includes('def search_layer_demo'))).toEqual(blocks(old));
    expect(math(body)).toEqual(math(old));
    expect(math(body)).toHaveLength(8);
    expect(body).toContain('its own distance is not');
    expect(body).toContain('The width limits retained results');
    expect(body).toContain('does not implement insertion, upper layers, deletion, filtering');
  });
  it('publishes27 anchored subjects per locale and identifies mixed-language content',()=>{
    const p=matter(body),terms=referenceTerms(row.source,p.data.reference_terms,parseMarkdownHeadings(p.content,true));
    expect(terms).toHaveLength(27);
    expect(p.data.source_language).toBe('mixed');
    for(const[path,en]of [['01 Справочник/_index.md',false],['en/01 Reference/_index.md',true]]as const)
      for(const t of terms)
        expect(read(path)).toContain('[['+row.source.slice(0,-3)+'#'+t.anchor+'|'+(en?t.label_en:t.label)+']]');
  });
  it('preserves original figure bytes, clarifies the HNSW caption and pins source fixtures',()=>{
    const assets=YAML.parse(read('05 Источники/asset-registry.yml')).assets,figures=images(body);
    expect(figures).toHaveLength(8);
    expect(figures.map(m=>m[1])).toEqual(images(old).map(m=>m[1]));
    for(const f of figures){
      expect(f[2]?.trim()).toBeTruthy();
      const a=assets.find((a:any)=>a.asset===f[1]);
      expect(a,f[1]).toBeDefined();
      expect(hash(readFileSync(resolve(root,f[1])))).toBe(a.sha256);
    }
    expect(body).toContain('Пунктир между жёлтыми точками');
    const sources=JSON.parse(read('publishing/fixtures/hnswlib-3f342966/sources.json'));
    expect(sources.commit).toBe('3f3429661187e4c24a490a0f148fc6bc89042b3d');
    for(const f of sources.files)
      expect(hash(readFileSync(resolve(root,'publishing/fixtures/hnswlib-3f342966',f.file)))).toBe(f.sha256);
  });
});
