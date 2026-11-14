import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString();
const sha=(s:Buffer|string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-normalization-course-additions.json',stop='2026-11-03-optimizer-route-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60\x60\x60python\n([\s\S]*?)^\x60\x60\x60\s*$/gm)].map(m=>m[1]);
const figures=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1].replace(/^02 Areas\/ML & DL\//,''));
const folder='00 Учебник/Assets/Figures/curated/normalization-course-2026-11-03/';
const assets=JSON.parse(text(folder+'assets.json'));

describe('normalization original-course additions',()=>{
  it('restores exact prior chapter, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
  });
  it('keeps three original programs, every old figure and all displayed derivations',()=>{
    expect(programs(current)).toHaveLength(4);
    expect(programs(current).slice(0,3)).toEqual(programs(prior));
    expect(figures(current)).toHaveLength(6);
    expect(figures(current)).toEqual(expect.arrayContaining(figures(prior)));
    const equations=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(equations(current)).toEqual(equations(prior));
    expect(current).toContain('dividing by $B-1$');
    expect(current).toContain('training functions identical');
    expect(current).toContain('does not inherit');
  });
  it('pins the published PNGs, registry and MIT license',()=>{
    const registry=parse(text('05 Источники/asset-registry.yml')).assets;
    expect(assets).toHaveLength(2);
    for(const asset of assets){
      expect(registry.filter((x:any)=>x.asset===asset.asset)).toEqual([asset]);
      expect(sha(read(asset.asset))).toBe(asset.sha256);
      expect(sha(read(asset.public_license_copy))).toBe(asset.license_evidence_sha256);
      expect(text(asset.public_license_copy)).toContain('Copyright (c) 2022 Andrej Karpathy');
      expect(current).toContain(asset.asset);
    }
    const xiong=registry.find((x:any)=>x.asset.endsWith('/xiong-pre-post-norm.png'));
    expect(xiong.used_in).toContain(row.source);
    expect(xiong.license).toBe('CC BY 4.0');
    expect(sha(read(xiong.asset))).toBe(xiong.sha256);
  });
  it('compares notebook code, saved figures, license and experiment conditions to the archive', { tags: ['source-archive'] }, ()=>{
    const notebook=JSON.parse(text(assets[0].source_asset));
    const cell=notebook.cells[13].source.join('');
    const original=cell.slice(cell.indexOf('class BatchNorm1d:'),cell.indexOf('class Tanh:')).trimEnd();
    expect(programs(current)[3]).toContain(original.replace(/^[ \t]+$/gm,''));
    for(const asset of assets){
      expect(sha(read(asset.source_asset))).toBe(asset.source_asset_sha256);
      const saved=notebook.cells[asset.source_cell_index].outputs[asset.source_output_index].data['image/png'];
      expect(read(asset.asset)).toEqual(Buffer.from(Array.isArray(saved)?saved.join(''):saved,'base64'));
      expect(sha(read(asset.license_evidence))).toBe(asset.license_evidence_sha256);
      expect(read(asset.public_license_copy)).toEqual(read(asset.license_evidence));
    }
    const xiong=parse(text('05 Источники/asset-registry.yml')).assets.find((x:any)=>x.asset.endsWith('/xiong-pre-post-norm.png'));
    expect(sha(read(xiong.source_asset))).toBe(xiong.source_asset_sha256);
    expect(notebook.cells[14].source.join('')).toContain('if i >= 1000:');
    for(const cell of [15,16])expect(notebook.cells[cell].source.join('')).toContain('density=True');
  });
  it('does not describe a trained notebook snapshot as an initialization experiment',()=>{
    expect(current).toContain('iterations 0 through 1000');
    expect(current).toContain('not fresh measurements');
    expect(current).toContain('not a controlled comparison');
    expect(current).toContain('1.952382');
    expect(current).toContain('0.0591');
  });
  it('indexes seventeen explanations in both locales and links directly to modern applications',()=>{
    for(const anchor of ["initialization-symmetry","initialization-second-moment","xavier-initialization","he-initialization","normalization-axes","batchnorm-train-eval","normalization-causal-leakage","token-layernorm","rmsnorm-forward-precision","normalization-backward","residual-addition","residual-gradients","residual-zero-projection","pre-post-normalization","residual-variance","normalization-diagnostics","batchnorm-source-implementation"]){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(p)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+anchor+'|');
    }
    expect(current).not.toContain('Формулы для быстрого обращения');
    expect(current).toContain('главе о нормализации и gating в LLM');
  });
});
