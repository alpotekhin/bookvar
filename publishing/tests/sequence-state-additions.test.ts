import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';
import {beforeReaderRouteNavigation} from '../tools/reader-route-baseline.mjs';

const root=resolve(import.meta.dirname,'../..'),read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-sequence-state-additions.json',stop='2026-11-03-moe-gate-additions.json';
const journal=JSON.parse(read('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(read(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^~~~python\n([\s\S]*?)^~~~\s*$/gm)].map(x=>x[1]);
const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(x=>x[1]);
const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(x=>x[1]);
const figures=JSON.parse(read('publishing/tools/sequence-state-figures.json'));

describe('chunkwise gated-delta explanation and original state figures',()=>{
  it('restores the complete owner, registry and both indexes without accepting an unrecorded edit',()=>{
    expect(journal.files).toHaveLength(4);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(read(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('783cab0ddb3b947eecd310c420898a0acf1ff6a7088e45f98ef1176de4ba4aea');
    expect(sha(beforeReaderRouteNavigation(read(row.source),row.source,root)))
      .toBe('4f78170b76794b013076afc363b9df813563ac0df59e0465bd0d1c74633bb803');
  });
  it('preserves six complete programs, all earlier math and thirty-one unchanged originals',()=>{
    expect(programs(prior)).toHaveLength(6);
    expect(programs(current)).toHaveLength(7);
    expect(programs(current).filter(x=>!x.includes('def gated_delta_chunked('))).toEqual(programs(prior));
    expect(math(current)).toEqual(expect.arrayContaining(math(prior)));
    const old='/recurrent-update-rule-comparison.png';
    expect(images(current).filter(x=>!x.includes('/sequence-state-2026-11-03/')))
      .toEqual(images(prior).filter(x=>!x.endsWith(old)));
    expect(images(current)).toHaveLength(33);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const f of figures){
      const entry=registry.find((x:any)=>x.asset===f.asset);
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.used_in).toContain(row.source);
      expect(entry.source_sha256).toBe('bfca970ee2a85ed15b7e94adfd4e246366988ff1d927a861cfffa9cd591b23f7');
      expect(entry.crop_box_px).toEqual(f.crop_pixels);
      expect(entry.dimensions_px).toEqual(f.dimensions);
      expect(sha(readFileSync(resolve(root,f.asset)))).toBe(f.sha256);
      expect(f.dimensions[0]).toBe(1988);
    }
    const archived=registry.find((x:any)=>x.asset.endsWith(old));
    expect(archived.used_in).not.toContain(row.source);
    expect(readFileSync(resolve(root,archived.asset)).length).toBeGreaterThan(1000);
    expect(current).toContain('full English caption');
    expect(current).toContain('300 dpi');
  });
  it('derives the causal triangular solve, incoming state, readout and outgoing state',()=>{
    for(const phrase of ['(I+L)U&=R','strictly lower triangular','u_2=18.9-0.3u_1',
      'wrong second output 22.7','[14.72, 13.96]','corrections_block',
      'Unlike the TTT mini-batch rule','Their intermediate variables differ',
      'np.linalg.solve','not\ntheir GPU kernel'])
      expect(current).toContain(phrase);
    const at=current.indexOf('<a id="seq-delta-chunks"></a>');
    expect(at).toBeGreaterThan(current.indexOf('def gated_delta('));
    expect(at).toBeLessThan(current.indexOf('def ttt_linear('));
  });
  it('provides twenty-seven real subject destinations in both indexes, keeping the canonical language',()=>{
    const meta=matter(current).data;
    expect(meta.reference_terms).toHaveLength(27);
    expect(meta.source_language).toBeUndefined();
    expect(meta.previous).toContain('02 Mixture of Experts');
    expect(meta.next).toContain('_index');
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
  });
});
