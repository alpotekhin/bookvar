import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTensorFactorization} from '../tools/tensor-factorization-baseline.mjs';
import {beforeSystemsCompletion} from '../tools/systems-completion-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const row=JSON.parse(read('docs/audit/2026-10-29-tensor-factorization-addition.json'));
const chapter=beforeSystemsCompletion(read(row.source),row.source,root);
const packet='05 Источники/Originals/tensor-factorization-2026-10-29/';

describe('Tensor decomposition explanation',()=>{
  it('preserves the previously accepted chapter outside the bounded addition',()=>{
    expect(sha(chapter)).toBe(row.after_sha256);
    expect(sha(beforeTensorFactorization(chapter,row.source))).toBe(row.before_sha256);
    expect(()=>beforeTensorFactorization(chapter.replace('one component is an outer product',
      'unreviewed change'),row.source)).toThrow();
    expect(sha(beforeTensorFactorization(chapter+'\nUnreviewed outside change.',row.source)))
      .not.toBe(row.before_sha256);
  });
  it('retains the original TensorLy explanations and source snapshots',()=>{
    const sources=JSON.parse(read(packet+'sources.json'));
    for(const s of sources.sources) expect(sha(read(packet+s.file))).toBe(s.sha256);
    const normalize=(s:string)=>s.replace(/\s+/g,' ').trim();
    const guide=normalize(read(packet+'tensorly-guide.rst'));
    for(const id of ['cp','tucker']){
      const quote=chapter.match(new RegExp(`<!-- tensorly-excerpt: ${id}:start -->([\\s\\S]*?)<!-- tensorly-excerpt: ${id}:end -->`))![1];
      expect(guide).toContain(normalize(quote.replace(/^> ?/gm,'')));
    }
    for(const [file,copy]of [['tensorly-LICENSE.txt','tensorly-BSD-3-Clause.txt'],
      ['torch-LICENSE','tensorly-torch-BSD-3-Clause.txt']])
      expect(read('site/public/licenses/'+copy)).toBe(read(packet+file));
  });
  it('keeps the complete original diagram and all eleven runnable examples',()=>{
    const [asset]=JSON.parse(read('00 Учебник/Assets/Figures/curated/tensor-factorization-2026-10-29/assets.json'));
    expect(sha(read(asset.asset))).toBe(asset.sha256);
    expect(sha(read(asset.license_evidence))).toBe(asset.license_evidence_sha256);
    expect(sha(read(asset.source_page_snapshot))).toBe(asset.source_page_sha256);
    expect(chapter).toContain(asset.asset);
    expect(chapter.match(/<!-- checked-example:/g)).toHaveLength(11);
    expect(chapter.match(/!\[\[/g)).toHaveLength(9);
    for(const phrase of ['super-diagonal core','There is no activation',
      'intermediate channels is one CP component','only while the mode sizes',
      "implementation='reconstructed'",'1_000_000']) expect(chapter).toContain(phrase);
  });
});
