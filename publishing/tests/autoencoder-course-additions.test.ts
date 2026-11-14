import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-autoencoder-course-additions.json',stop='2026-11-03-seq2seq-state-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);

describe('complete autoencoder course comparisons and retrieval explanation',()=>{
  it('restores the complete preceding owner, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('0fa503dade5bacee4bf88d40a4aa799474328586332fe92fc6a28538dca9b4fd');
  });
  it('preserves all fourteen prior programs, seventeen equations and six illustrations',()=>{
    expect(programs(current)).toHaveLength(15);
    expect(programs(current).filter(x=>!x.includes('retrieval_dtype'))).toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
    expect(math(prior)).toHaveLength(17);
    expect(math(current)).toEqual(expect.arrayContaining(math(prior)));
    expect(math(current)).toHaveLength(19);
    expect(images(current).filter(x=>!x.includes('autoencoder-course-2026-11-03'))).toEqual(images(prior));
    expect(images(current)).toHaveLength(10);
    const strip=(s:string)=>s.replace(/<a id="(?:ae|vae)-[^"]+"><\/a>\n\n/g,'');
    for(const section of ['## ELBO и два разных KL','## Reparameterization trick','## Исходный VAE из PyTorch examples']){
      const extract=(s:string)=>strip(s).split(section)[1].split('\n## ')[0];
      expect(extract(current)).toBe(extract(prior));
    }
  });
  it('pins the four published original SVGs and their license',()=>{
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    const rows=registry.filter((x:any)=>x.asset.includes('/autoencoder-course-2026-11-03/'));
    expect(rows).toHaveLength(4);
    for(const entry of rows){
      expect(sha(read(entry.asset))).toBe(entry.sha256);
      expect(entry.license).toBe('MIT');
      expect(entry.derivation).toBe('notebook-svg-output-copy');
      expect(sha(read(entry.license_evidence))).toBe(entry.license_evidence_sha256);
    }
  });
  it('compares the four SVGs with the archived notebook outputs', { tags: ['source-archive'] }, ()=>{
    expect(()=>execFileSync(process.execPath,['publishing/tools/prepare_autoencoder_course_figures.mjs','--check'],{cwd:root})).not.toThrow();
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    const rows=registry.filter((x:any)=>x.asset.includes('/autoencoder-course-2026-11-03/'));
    expect(rows).toHaveLength(4);
    const notebook=JSON.parse(read(rows[0].source_asset));
    for(const entry of rows){
      const svg=notebook.cells[entry.source_cell].outputs[0].data['image/svg+xml'].join('');
      expect(read(entry.asset)).toBe(svg);
    }
  });
  it('indexes twenty-four mechanisms and explains the metric counterexample without claiming a training run',()=>{
    const meta=matter(current).data;
    expect(meta.reference_terms).toHaveLength(24);
    expect(meta.source_language).toBe('mixed');
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    for(const phrase of ['not a measured 48-fold file compression ratio','horizontal axis is logarithmic',
      'a calibrated\nout-of-distribution detector','first coordinate counts much more',
      '[[1, 0]]','[[0, 1]]','not samples from a VAE','not a local\ntraining result'])
      expect(current).toContain(phrase);
  });
});
