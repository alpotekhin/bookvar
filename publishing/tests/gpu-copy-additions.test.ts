import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-04-gpu-copy-additions.json';
const stop='2026-11-03-systems-timeline-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const body=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const programs=(s:string)=>[...s.matchAll(/^(~~~|```)python\n([\s\S]*?)^\1\s*$/gm)].map(x=>x[2]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);

describe('GPU copy timelines and independently protected buffer lifetimes',()=>{
  it('reversibly preserves the preceding complete chapter, registry and both indexes',()=>{
    expect(journal.files).toHaveLength(4);
    for(const f of journal.files){
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,own))).toBe(f.after_sha256);
      expect(hash(beforeTopicConsolidation(read(f.source),f.source,root,stop))).toBe(f.before_sha256);
      expect(()=>beforeTopicConsolidation(read(f.source)+'UNREVIEWED',f.source,root,stop)).toThrow();
    }
    expect(hash(prior)).toBe('d700f60713f8d39ff010f0c7f7daeaa2aa3a0c1b3125ef63aad0858e29f9db1c');
    expect(images(prior)).toHaveLength(18);
    expect(images(body)).toHaveLength(22);
    expect(images(body)).toEqual(expect.arrayContaining(images(prior)));
    expect([...body.matchAll(/!\[\[[^\]|]+\|([^\]]+)\]\]/g)]).toHaveLength(22);
    expect(programs(prior)).toHaveLength(6);
    expect(programs(body)).toHaveLength(8);
    expect(programs(body)).toEqual(expect.arrayContaining(programs(prior)));
    expect(math(body)).toEqual(math(prior));
    for(const name of ['layout','buffer-reuse','soc','portability']){
      const marker='<!-- hardware-excerpt: '+name+':start -->';
      expect(body.split(marker)[1].split('<!-- hardware-excerpt: '+name+':end -->')[0])
        .toBe(prior.split(marker)[1].split('<!-- hardware-excerpt: '+name+':end -->')[0]);
    }
  });
  it('explains the actual independent operation, unequal scales and three distinct dependencies',()=>{
    for(const text of ['The multiplication does not read the','horizontal scales differ',
      'not a Bookvar benchmark','not dedicated','a separate copy stream',
      'It does not\norder the sum after initialization',
      'A.record_stream(s)','s.wait_stream(torch.cuda.default_stream(cuda))',
      'CUDA stream example executed:', 'that branch does not test a GPU',
      '| Pinned CPU input | H2D copy | 3 ms |',
      '| GPU input | Kernel | 10 ms |','teaching interleaving',
      'assert read_interleaving(early_host) == [1, 2, 0, 0]',
      'assert read_interleaving(early_device) == [1, 2, 0, 0]'])
      expect(body).toContain(text);
    expect(body.indexOf('id="gpu-copy-overlap"')).toBeLessThan(body.indexOf('id="gpu-buffer-lifetimes"'));
    expect(body.indexOf('id="gpu-buffer-lifetimes"')).toBeLessThan(body.indexOf('id="gpu-hardware-mapping"'));
  });
  it('copies four complete pinned PNGs and the entire BSD notice without raster alteration',()=>{
    const manifest=JSON.parse(read('publishing/tools/gpu-copy-source-figures.json'));
    const registry=YAML.parse(read('05 Источники/asset-registry.yml'),{merge:true}).assets;
    const license=manifest.files.find((x:{name:string})=>x.name==='LICENSE');
    expect(hash(readFileSync(resolve(root,'site/public/licenses/pytorch-tutorials-5672cd98-BSD-3-Clause.txt'))))
      .toBe(license.sha256);
    const figures=manifest.files.filter((x:{dimensions:number[]|null})=>x.dimensions);
    expect(figures).toHaveLength(4);
    for(const f of figures){
      const asset=manifest.folder+f.name,b=readFileSync(resolve(root,asset));
      expect(hash(b)).toBe(f.sha256);
      expect(b.length).toBe(f.bytes);
      expect([b.readUInt32BE(16),b.readUInt32BE(20)]).toEqual(f.dimensions);
      const entry=registry.find((x:{asset:string})=>x.asset===asset);
      expect(entry.source_asset_sha256).toBe(f.sha256);
      expect(entry.license).toBe('BSD-3-Clause');
      expect(entry.commit).toBe(manifest.pin);
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.used_in).toContain(row.source);
    }
  });
  it('links 28 actual subjects in both locales without calling the mixed original an English edition',()=>{
    const meta=matter(body).data;
    expect(meta.reference_terms).toHaveLength(28);
    expect(meta.source_language).toBe('mixed');
    for(const term of meta.reference_terms){
      expect(body.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(body.match(/id="cs336-systems-gpu"/g)).toHaveLength(1);
    expect(body).toContain('01 Модель как часть системы|Модель как часть системы');
    expect(body).toContain('03 Измерение производительности и roofline|Измерение производительности и roofline');
  });
});

