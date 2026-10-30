import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(t:string)=>createHash('sha256').update(t).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md';

describe('semi-supervised explanation and source reuse',()=>{
  it('historical explanation preservation: preserves accepted adjacent chapter explanations and inserts the route',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/semi-supervised-fixmatch$/gm)).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41e Обучение с частичной разметкой');
    const edge='41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Обучение с частичной разметкой — pseudo-labeling и FixMatch';
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding.md');
    expect(hash(previous.replace(edge,'42 Next-token prediction|Целевая функция предобучения')))
      .toBe('db32ea0fad9b98fe920150fda9408c857678c629f6f962df0878e1235f5590c8');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md')
      .replace('41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества','41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей');
    expect(hash(next.replace('41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей','41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding')))
      .toBe('36663ad6d1759134b00f1aebf4d8e9d7538513109e1749b64f5bb3a37fcd13af');
  });
  it('keeps complete programs, figures, assumptions and exact protocol distinctions',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(14);
    expect(text.match(/^!\[\[/gm)).toHaveLength(8);
    for(const anchor of ['pseudo-labels','consistency','fixmatch','training-step','confirmation-bias',
      'label-propagation','published-results','cost-accounting','self-supervised-boundary'])
      expect(text).toContain(`id="${anchor}"`);
    expect(text).toContain('`tf.stop_gradient`');
    expect(text).toContain('full unlabelled batch size');
    expect(text).toContain('despite the broader');
    expect(text.replace(/\s+/g,' ')).toContain('they are not image augmentations or a reproduction');
    expect(text).toContain('scenario assumption');
  });
  it('binds all complete author images to sources and license copies',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/assets.json'));
    expect(rows).toHaveLength(3);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      if(row.derivation==='source-raster-copy')
        expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
      else {
        expect(row.derivation).toBe('pdf-page-render-crop');
        expect(row.dimensions_px).toEqual([1850,700]);
        expect(row.permission_asset).toContain('FixMatch-diagram-white.png');
      }
      expect(readFileSync(resolve(root,row.public_license_copy))).toEqual(readFileSync(resolve(root,row.license_evidence)));
    }
  });
});
