import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe, expect, it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(x:string|Buffer)=>createHash('sha256').update(x).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции.md';
const prefix='00 Учебник/Assets/Figures/curated/sim-to-real-2026-10-28/';
const packet='05 Источники/Originals/sim-to-real-2026-10-28/';

describe('synthetic images and sim-to-real reader',()=>{
  it('historical explanation preservation: inserts a thematic route without rewriting accepted synthetic-text explanations',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/synthetic-images-sim-to-real$/gm)).toHaveLength(2);
    expect(nav).toContain('source: '+chapter);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('53a Синтетические изображения');
    let previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md');
    previous=previous.replace('11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции]]',
      '14 Inference и оптимизация/54 Декодирование и выбор следующего токена]]');
    previous=previous.replace('11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции|Синтетические изображения и перенос из симуляции',
      '14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Декодирование и выбор следующего токена');
    expect(hash(previous)).toBe('8b162e5380772ff0066fc7241061f4d6af8aa7e536d1b0172a753211bf6e855e');
  });

  it('retains source code excerpts and complete annotation, risk and gradient examples',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(9);
    expect(text.match(/^```python title=/gm)).toHaveLength(3);
    expect(text.match(/!\[\[/g)).toHaveLength(7);
    expect(text).toContain('source_language: en');
    for(const anchor of ['renderer-annotations','randomized-scenes','domain-shift-risk',
      'domain-adversarial-learning','alignment-limits','synthetic-real-validation','synthetic-privacy'])
      expect(text).toContain('id="'+anchor+'"');
    const snippets=[...text.matchAll(/^```python title="Original BlenderProc[^\n]*\n([\s\S]*?)^```/gm)].map(m=>m[1].trimEnd());
    const sources=['examples_basics_semantic_segmentation_main.py',
      'examples_advanced_material_randomizer_main.py','examples_basics_camera_sampling_main.py'];
    snippets.forEach((snippet,i)=>expect(read(packet+'blenderproc/'+sources[i])).toContain(snippet));
    const normalized=text.replace(/\s+/g,' ');
    for(const qualification of ['not the ordinary derivative of the identity',
      'fixed multiplier 1 for the domain head','does not assign the same extraction rate',
      'off-diagonal formula','not a local Blender run','target mass lies outside source support'])
      expect(normalized).toContain(qualification);
    expect(text).not.toMatch(/```(?:mermaid|svg)/);
  });

  it('binds all original figures to pinned sources and actual permission scope',()=>{
    const rows=JSON.parse(read(prefix+'assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(7);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.used_in).toEqual([chapter]);
      for(const [path,expected] of [[row.asset,row.sha256],[row.source_asset,row.source_asset_sha256],
        [row.source_page_snapshot,row.source_page_sha256],[row.license_evidence,row.license_evidence_sha256]])
        expect(hash(readFileSync(resolve(root,path)))).toBe(expected);
      expect(read(row.public_license_copy)).toBe(read(row.license_evidence));
      if(row.derivation==='source-jpeg-copy'){
        expect(row.commit).toBe('76bd8b82ee19489843578ff045298642197bf883');
        expect(row.license).toBe('GPL-3.0');
        expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
      }else{
        expect(row.license).toBe('CC BY (version unspecified in JMLR publication policy)');
        expect(row.source_page).toBe(12);
        expect(row.crop_box_px).toEqual([640,470,2840,1410]);
      }
    }
  });
});
