import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe, expect, it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const chapter='00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований.md';
const prefix='00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/';
const packet='05 Источники/Originals/data-augmentation-2026-10-28/';

describe('data augmentation source reader',()=>{
  it('adds the route after Dropout without rewriting its accepted explanation',()=>{
    const nav=read('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/neural-networks\/data-augmentation$/gm)).toHaveLength(2);
    expect(nav).toContain('source: '+chapter);
    expect(read('00 Учебник/_index.md')).toContain('06a Аугментация данных');
    const path='00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md';
    const previous=beforeTopicConsolidation(read(path),path,root,
      '2026-10-29-adaptive-clipping-additions.json');
    const restored=previous.replace(
      '[[00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Следующая глава]]\nразбирает преобразования обучающих примеров, смешивание изображений и меток,\nа также выбор и проверку политик аугментации.',
      '[[00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|Следующая глава]]\nразбирает локальные связи и общие веса свёрточной сети. Эти ограничения\nзадаются её архитектурой, а не случайной маской обучающего режима.');
    expect(hash(restored)).toBe('ca5d07f0d27f27f8143f53882014a629e0f50c936e448ed8658be64acde7863c');
  });

  it('retains source excerpts alongside complete coordinate, loss and policy examples',()=>{
    const text=read(chapter);
    expect(text).toContain('source_language: en');
    expect(text.match(/^```python$/gm)).toHaveLength(9);
    expect(text.match(/^```python title=/gm)).toHaveLength(2);
    expect(text.match(/^!\[\[/gm)).toHaveLength(3);
    for(const anchor of ['augmentation-target','augmentation-objective','augmentation-views',
      'mixup','cutmix','augmentation-policies','text-audio-augmentation','augmentation-evaluation'])
      expect(text).toContain('id="'+anchor+'"');
    const d2l=read(packet+'d2l/image-augmentation.md');
    const helper='def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):\n    Y = [aug(img) for _ in range(num_rows * num_cols)]\n    d2l.show_images(Y, num_rows, num_cols, scale=scale)';
    expect(text).toContain(helper); expect(d2l).toContain(helper);
    const original=read(packet+'cutmix/train.py').split('\n').slice(229,240).join('\n');
    expect(text).toContain(original);
    for(const caveat of ['synthetic English source and the original German target',
      'not a measurement of visible object', 'not 196 hyperparameter trials',
      'does not disable a random crop', 'Von.grzanka', 'obsolete `np.int`'])
      expect(text.replace(/\s+/g,' ')).toContain(caveat);
    expect(text).not.toMatch(/```(?:mermaid|svg)/);
  });

  it('binds two unchanged originals and the reused SimCLR image to their licenses',()=>{
    const rows=JSON.parse(read(prefix+'assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(2);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.used_in).toEqual([chapter]);
      const bytes=readFileSync(resolve(root,row.asset));
      expect(bytes).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect(hash(bytes)).toBe(row.sha256);
      expect(hash(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
      expect(read(row.public_license_copy)).toBe(read(row.license_evidence));
    }
    const sim=JSON.parse(read('00 Учебник/Assets/Figures/curated/simclr-2026-10-28/assets.json'))
      .find((r:any)=>r.asset.endsWith('/augmentations.png'));
    expect(sim.used_in).toContain(chapter);
    expect(registry.find((r:any)=>r.id===sim.id)).toEqual(sim);
    expect(sim.sha256).toBe('fcca5e8050e7c00364ecc4a838721bf7049f957709ab5d8aab75efc190d45969');
  });
});
