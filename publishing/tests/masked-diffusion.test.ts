import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { readingPaths } from '../adapter/reading-paths.js';

const root=resolve(import.meta.dirname,'../..');
const path='00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA.md';
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(p:string)=>createHash('sha256').update(readFileSync(resolve(root,p))).digest('hex');
const body=read(path);

describe('masked diffusion language models',()=>{
  it('executes all four printed programs and independent probability controls',()=>{
    const report=JSON.parse(execFileSync('python3',['publishing/tools/check_masked_diffusion_examples.py'],{cwd:root,encoding:'utf8'}));
    expect(report.python_blocks).toBe(4);
    expect(report.exact_finite_state_checks).toHaveLength(7);
    expect(report.remasking_marginal_cases).toBe(120);
    expect(report.negative_controls).toBe(6);
    expect(report.training_stream_leakage_control).toBe(true);
    const receipt=JSON.parse(read('docs/audit/2026-10-29-final-closure-masked-examples.json'));
    expect(receipt.chapter_sha256).toBe(hash(path));
    expect(receipt.checker_sha256).toBe(hash('publishing/tools/check_masked_diffusion_examples.py'));
  });
  it('preserves four complete licensed original figures with their primary bytes',()=>{
    const specs=JSON.parse(read('00 Учебник/Assets/Figures/curated/masked-diffusion-2026-10-29/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(specs).toHaveLength(4);
    for(const spec of specs){
      expect(body).toContain(spec.asset);
      expect(spec.license).toBe('CC BY 4.0');
      expect(existsSync(resolve(root,spec.source_asset))).toBe(true);
      expect(hash(spec.asset)).toBe(spec.sha256);
      expect(hash(spec.source_asset)).toBe(spec.sha256);
      expect(hash(spec.license_evidence)).toBe(spec.license_evidence_sha256);
      expect(read(spec.license_evidence)).toMatch(/License: CC BY 4\.0/);
      expect(registry.find((r:any)=>r.asset===spec.asset)?.used_in).toContain(path);
    }
  });
  it('integrates the explanation in the existing module and both subject indexes',()=>{
    const book=loadManifest(resolve(root,'publishing/navigation.yml')).sections.find(s=>s.id==='textbook')!;
    const route='textbook/pretraining/masked-diffusion-language-models';
    expect(book.pages.find(p=>p.route===route)?.source).toBe(path);
    expect(readingPaths(book).find(p=>p.id==='representation-learning')?.routes.slice(-2))
      .toEqual(['textbook/pretraining/mae-masked-autoencoders',route]);
    for(const p of ['00 Учебник/_index.md','en/00 Textbook/_index.md','01 Справочник/_index.md','en/01 Reference/_index.md'])
      expect(read(p)).toContain(path.replace(/\.md$/,''));
    for(const anchor of ['masking-process','weighted-mask-loss','reverse-transition','complete-toy-sampler',
      'llada-training-and-selection','block-diffusion-and-kv','remasking-and-refinement','likelihood-and-serving'])
      expect(body).toContain('id="'+anchor+'"');
    expect(body).toContain('not a complete ReMDM sampler');
    expect(body).toContain('a different estimator');
    expect(body).toContain('not just the word “block”');
  });
});
