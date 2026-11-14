import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-seq2seq-state-additions.json',stop='2026-11-03-lstm-gru-contract-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);

describe('complete seq2seq state-interface and beam-state explanations',()=>{
  it('restores the exact preceding whole owner, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('f8f44caa2a641441cedbc99c9419de2bbb2ce2567585e2ccde65e48c5f02535b');
  });
  it('retains all three old programs, display equations and seven original diagrams',()=>{
    expect(programs(current)).toHaveLength(5);
    expect(programs(current).filter(x=>!['state_bridge','selected_parents'].some(k=>x.includes(k))))
      .toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
    expect(math(current)).toEqual(math(prior));
    expect(math(current)).toHaveLength(11);
    expect(images(current).filter(x=>!x.includes('seq2seq-bridge-2026-11-03'))).toEqual(images(prior));
    expect(images(current)).toHaveLength(8);
    const assets=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const prefixed of images(current)){
      const path=prefixed.replace(/^02 Areas\/ML & DL\//,'');
      const entries=assets.filter((x:any)=>x.asset===path);
      expect(entries).toHaveLength(1);
      expect(sha(readFileSync(resolve(root,path)))).toBe(entries[0].sha256);
      expect(entries[0].license).toBe('CC BY-SA 4.0');
    }
  });
  it('retains the exact licensed init_state excerpt and complete original SVG', { tags: ['source-archive'] }, ()=>{
    const source=read('05 Источники/Originals/seq2seq-2026-10-21/d2l-encoder-decoder.md');
    expect(sha(source)).toBe('ad89663c337fc3aed42c26fb53365bb249a33ac96e45828adfb9ae0072cd0b05');
    const excerpt=current.split('<!-- source-excerpt: d2l-decoder-init-state -->\n')[1]
      .split('\n<!-- /source-excerpt -->')[0].replace(/^> /gm,'');
    const expected=source.slice(source.indexOf('In the following decoder interface,'),
      source.indexOf('into the encoded state.')+'into the encoded state.'.length);
    expect(excerpt).toBe(expected);
    const asset='00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg';
    expect(read(asset)).toBe(read('05 Источники/Originals/seq2seq-2026-10-21/encoder-decoder.svg'));
  });
  it('pins the published encoder–decoder illustration and license',()=>{
    const asset='00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg';
    expect(sha(read(asset))).toBe('bf5083530e10aff826bfc6be26b4b32ccd8ade3e7ad86cc7b7ee66dc5601ddaf');
    expect(read(asset)).toContain('viewBox="0 0 304 34"');
    expect(sha(read('site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt')))
      .toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
  });
  it('indexes twenty-three mechanisms and explains both state-mapping operations completely',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(23);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    for(const phrase of ['mix the batch and direction axes','28$\nparameters',
      'both $h_0$ and $c_0$','both inherit the state\nafter C',
      'one state-routing operation','axis 1','An EOS child',
      'not choose the beam or compute sequence probabilities',
      'fixed number of source coordinates'])
      expect(current).toContain(phrase);
  });
});
