import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString(),sha=(s:Buffer|string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-ngram-makemore-additions.json',stop='2026-11-03-cnn-route-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const excerpts=(s:string)=>[...s.matchAll(/^\x60{3}python title=[^\n]+\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const folder='00 Учебник/Assets/Figures/curated/ngram-makemore-2026-11-03/';
const [asset]=JSON.parse(text(folder+'assets.json'));

describe('N-gram source count matrix and normalization',()=>{
  it('restores the exact whole preceding chapter, source registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('032dc6d57c8399255ec427be82cf59fd2fe30885f218abe7083a5596901ef8ea');
  });
  it('retains the six complete examples, equations and four originals, then adds the source at section ends',()=>{
    expect(programs(current)).toHaveLength(7);
    expect(programs(current)).toEqual(expect.arrayContaining(programs(prior)));
    expect(excerpts(current)).toHaveLength(2);
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(math(current)).toEqual(math(prior));
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
    expect(images(current)).toHaveLength(5);
    expect(images(current)).toEqual(expect.arrayContaining(images(prior)));
    expect(current.indexOf('Биграмма объединяла оба случая')).toBeLessThan(current.indexOf('id="ngram-makemore-matrix"'));
    expect(current.indexOf('Правила замены затем не меняют при оценивании.')).toBeLessThan(current.indexOf('id="ngram-makemore-normalization"'));
    expect(current.indexOf('id="ngram-makemore-normalization"')).toBeLessThan(current.indexOf('## Backoff'));
  });
  it('pins the published count matrix, MIT notice and notebook explanation',()=>{
    expect(parse(text('05 Источники/asset-registry.yml')).assets.filter((a:any)=>a.asset===asset.asset)).toEqual([asset]);
    expect(sha(read(asset.asset))).toBe(asset.sha256);
    expect(sha(read(asset.public_license_copy))).toBe(asset.license_evidence_sha256);
    expect(asset.dimensions_px).toEqual([884,884]);
    expect(asset.license).toBe('MIT');
    expect(asset.crop_box_px).toBeUndefined();
    expect(current).toContain('cell 12 prints a row\nof ones');
  });
  it('compares the count matrix, author code and MIT notice with the archive', { tags: ['source-archive'] }, ()=>{
    for(const [p,h]of [[asset.source_asset,asset.source_asset_sha256],
      [asset.source_data,asset.source_data_sha256],[asset.source_page_snapshot,asset.source_page_sha256],
      [asset.license_evidence,asset.license_evidence_sha256]]) expect(sha(read(p))).toBe(h);
    const nb=JSON.parse(text(asset.source_asset));
    const encoded=nb.cells[11].outputs[0].data['image/png'];
    expect(read(asset.asset)).toEqual(Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64'));
    const source=(i:number)=>nb.cells[i].source.join('');
    expect(excerpts(current)[0].trimEnd()).toBe([7,8,9,10].map(source).join('\n\n').trimEnd());
    expect(excerpts(current)[1].trimEnd()).toBe(source(23).trimEnd());
    expect(read(asset.public_license_copy)).toEqual(read(asset.license_evidence));
  });
  it('indexes twenty-one mechanisms without translating or compressing the explanation',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(21);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    for(const phrase of ['4410/32033','6640/33885','4411/32060','1/32060',
      'row_denominators','[.8, 1.2]']) expect(current).toContain(phrase);
  });
  it('preserves the n-gram examples and replaces the four copied course images with licensed source figures',()=>{
    const note=text(row.source);
    expect(programs(note)).toEqual(programs(current));
    expect(excerpts(note)).toEqual(excerpts(current));
    const entries=parse(text('05 Источники/asset-registry.yml')).assets;
    const authors=entries.filter((entry:any)=>entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ngram-author-figures-2026-11-10/'));
    expect(authors).toHaveLength(3);
    expect(authors.map((entry:any)=>entry.license).sort())
      .toEqual(['CC BY-NC-SA 3.0','CC BY-NC-SA 3.0','CC BY-NC-SA 4.0']);
    for(const entry of authors){
      expect(entry).toMatchObject({metadata_status:'verified',rights_status:'source-license-verified',used_in:[row.source]});
      const png=read(entry.asset);
      expect(sha(png)).toBe(entry.sha256);
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual(entry.dimensions_px);
      expect(note).toContain(entry.asset+'|');
      expect(note).toContain(entry.license_url);
    }
    const archived=entries.filter((entry:any)=>entry.asset.includes('/voita-language-modeling/'));
    expect(archived).toHaveLength(4);
    for(const entry of archived){
      expect(entry).toMatchObject({used_in:[],publication_status:'retired',metadata_status:'needs-confirmation'});
      expect(note).not.toContain(entry.asset);
      expect(sha(read(entry.asset))).toBe(entry.sha256);
    }
  });
  it('matches all three figures to the source and rights evidence', { tags: ['source-archive'] }, ()=>{
    const entries=parse(text('05 Источники/asset-registry.yml')).assets;
    for(const entry of entries.filter((a:any)=>a.asset.startsWith('00 Учебник/Assets/Figures/curated/ngram-author-figures-2026-11-10/'))){
      expect(sha(read(entry.source_asset))).toBe(entry.source_asset_sha256);
      expect(sha(read(entry.license_evidence))).toBe(entry.license_evidence_sha256);
      if(entry.source_page_snapshot){
        expect(sha(read(entry.source_page_snapshot))).toBe(entry.source_page_sha256);
        expect(sha(read(entry.source_asset))).toBe(entry.sha256);
      }else{
        expect(entry.crop_box_px.slice(2).map((edge:number,i:number)=>edge-entry.crop_box_px[i]))
          .toEqual(entry.dimensions_px);
      }
    }
  });
});
