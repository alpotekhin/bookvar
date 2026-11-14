import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-lstm-gru-contract-additions.json';
const stop='2026-11-03-rnn-library-contract-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);

describe('complete LSTM/GRU state and parameterization explanations',()=>{
  it('restores the exact preceding chapter, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('c0afc59f36af36d3616792c6dc8873cdee8ce419fa8cf0db000698f2393283f4');
  });
  it('retains all five prior programs, every display equation and eleven original pictures',()=>{
    expect(programs(current)).toHaveLength(8);
    expect(programs(current).filter(x=>!['biased_gru','bidirectional_lstm','projected_lstm'].some(k=>x.includes(k))))
      .toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
    expect(math(current)).toEqual(math(prior));
    expect(math(current)).toHaveLength(22);
    expect(images(current).filter(x=>!x.endsWith('/d2l-birnn.svg'))).toEqual(images(prior));
    expect(images(current)).toHaveLength(12);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const prefixed of images(current)){
      const p=prefixed.replace(/^02 Areas\/ML & DL\//,'');
      const entry=registry.find((x:any)=>x.asset===p);
      expect(sha(readFileSync(resolve(root,p))),p).toBe(entry.sha256);
    }
    expect(registry.filter((x:any)=>images(current).some(p=>p.endsWith(x.asset)))
      .filter((x:any)=>x.author==='Chris Olah').every((x:any)=>x.metadata_status==='needs-confirmation')).toBe(true);
  });
  it('copies the complete licensed bidirectional original and the exact direction-order excerpt', { tags: ['source-archive'] }, ()=>{
    const p='05 Источники/Originals/lstm-gru-contracts-2026-11-03/d2l-bi-rnn.md';
    const source=read(p);
    expect(sha(source)).toBe('6de8c8124497b0c69703ba3805559d51eb57c96d93096b70e2d5b6e133373975');
    const excerpt=current.split('<!-- source-excerpt: d2l-bidirectional-order -->\n')[1]
      .split('\n<!-- /source-excerpt -->')[0].replace(/^> /gm,'');
    expect(excerpt).toBe(source.slice(source.indexOf('For the first RNN layer,'),
      source.indexOf('of the two underlying unidirectional RNN layers.')+'of the two underlying unidirectional RNN layers.'.length));
    const asset='00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg';
    expect(read(asset)).toBe(read('05 Источники/Originals/lstm-gru-contracts-2026-11-03/d2l-birnn.svg'));
  });
  it('pins the published bidirectional illustration and license',()=>{
    const asset='00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg';
    expect(sha(read(asset))).toBe('aa4e678afa0e31c160374974709512d4d7e0121eeab7a1af1ad7725403afc018');
    expect(read(asset)).toContain('viewBox="0 0 240 141"');
    expect(sha(read('site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt')))
      .toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
  });
  it('indexes all twenty-one mechanisms and retains each contract limitation',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(21);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    for(const phrase of ['A single constant bias cannot reproduce both',
      'first computed reverse state','actual boundary','not execution time',
      'm=6,p=5,d=2','246 rather than 240','projection participates in the\nrecurrence',
      'manual_gradients','library_gradients','отсоединяются оба вектора'])
      expect(current).toContain(phrase);
  });
});
