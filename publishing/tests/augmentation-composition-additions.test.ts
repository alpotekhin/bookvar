import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString();
const sha=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const own='2026-11-03-augmentation-composition-additions.json',stop='2026-11-03-dropout-route-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const archival=(s:string)=>[...s.matchAll(/^\x60{3}python title=[^\n]+\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);
const folder='00 Учебник/Assets/Figures/curated/augmentation-composition-2026-11-03/';
const [asset]=JSON.parse(text(folder+'assets.json'));

describe('augmentation composition original and subject index',()=>{
  it('restores the exact preceding owner, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
  });
  it('keeps all nine runnable programs, prior excerpts, derivations and originals',()=>{
    expect(programs(current)).toHaveLength(9);
    expect(programs(current)).toEqual(programs(prior));
    expect(archival(current)).toHaveLength(3);
    expect(archival(current)).toEqual(expect.arrayContaining(archival(prior)));
    const maths=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(maths(current)).toEqual(maths(prior));
    const figures=(s:string)=>[...s.matchAll(/!\[\[[^\]]+\]\]/g)].map(m=>m[0]);
    expect(figures(current)).toHaveLength(4);
    expect(figures(current)).toEqual(expect.arrayContaining(figures(prior)));
  });
  it('pins the published PyTorch gallery, complete license and explanatory boundary',()=>{
    const registry=parse(text('05 Источники/asset-registry.yml')).assets;
    expect(registry.filter((a:any)=>a.asset===asset.asset)).toEqual([asset]);
    expect(sha(read(asset.asset))).toBe(asset.sha256);
    expect(sha(read(asset.public_license_copy))).toBe(asset.license_evidence_sha256);
    expect(text(asset.asset).match(/<image\b/g)).toHaveLength(8);
    expect(text(asset.asset)).not.toMatch(/(?:href|src)="https?:/);
    expect(asset.license).toBe('CC BY-SA 4.0');
    expect(current).toContain('not an\nAutoAugment search result');
  });
  it('compares the saved PyTorch gallery, excerpts and license with the archive', { tags: ['source-archive'] }, ()=>{
    for(const [p,h]of [[asset.source_asset,asset.source_asset_sha256],
      [asset.source_page_snapshot,asset.source_page_sha256],[asset.source_code_snapshot,asset.source_code_sha256],
      [asset.license_evidence,asset.license_evidence_sha256]])expect(sha(read(p))).toBe(h);
    expect(read(asset.asset)).toEqual(read(asset.source_asset));
    const html=text(asset.source_page_snapshot);
    const panel=html.split('id="pytorch-17-0"')[1].split('id="mxnet-17-1"')[0];
    expect(panel).toContain('output_image-augmentation_7d0887_81_0.svg');
    expect(panel).not.toContain('output_image-augmentation_7d0887_84_0.svg');
    const original=text(asset.source_code_snapshot),added=archival(current).find(s=>s.includes('color_aug ='));
    expect(added).toBeDefined();
    const parts=added!.trimEnd().split('\naugs =');
    expect(original).toContain(parts[0]);
    expect(original).toContain('augs ='+parts[1]);
    expect(read(asset.public_license_copy)).toEqual(read(asset.license_evidence));
  });
  it('indexes sixteen full explanations and retains the English original',()=>{
    const metadata=matter(current).data;
    expect(metadata.source_language).toBe('en');
    expect(metadata.reference_terms).toHaveLength(16);
    for(const term of metadata.reference_terms){
      expect(current).toContain('<a id="'+term.anchor+'"></a>');
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(index)).toContain('[['+row.source.replace(/\.md$/,'')+'#'+term.anchor+'|');
    }
    expect(current.indexOf('### Compose')).toBeLessThan(current.indexOf('### AutoAugment'));
    expect(current.indexOf('### AutoAugment')).toBeLessThan(current.indexOf('### RandAugment'));
    expect(text('en/01 Reference/_index.md')).toContain('Аугментация данных — Mixup, CutMix и политики преобразований|');
    expect(current).toContain('The three labelled source excerpts');
  });
});
