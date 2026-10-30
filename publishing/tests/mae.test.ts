import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей.md';
const edge='41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей';

describe('MAE explanation and source integration',()=>{
  it('historical explanation preservation: adds the route and changes only accepted neighbouring navigation edges',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/mae-masked-autoencoders$/gm)).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41h Masked autoencoders');
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder.md');
    expect(hash(previous.replace(edge,'42 Next-token prediction|Целевая функция предобучения')))
      .toBe('fe649bdd8a8fc010337edbcc942a8c639664694483e854ad5c893426c1aada67');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md')
      .replace('41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества','41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей');
    expect(hash(next.replace(edge,'41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding')))
      .toBe('36663ad6d1759134b00f1aebf4d8e9d7538513109e1749b64f5bb3a37fcd13af');
  });
  it('retains executable mechanisms, original figures and experimental limits',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(6);
    expect(text.match(/^!\[\[/gm)).toHaveLength(3);
    for(const anchor of ['asymmetric-mae','patch-layout','mask-and-restore','masked-pixel-loss',
      'normalized-targets','mae-training-step','reading-reconstructions','mae-cost-and-evaluation'])
      expect(text).toContain(`id="${anchor}"`);
    expect(text).not.toContain('SOURCE_PATCH_HELPERS');
    expect(text).not.toContain('SOURCE_FORWARD_METHODS');
    for(const term of ['sample variance','different checkpoint','copying the visible input','no MoCo-style',
      'historical source measurements','not that architecture'])
      expect(text.replace(/\n/g,' ')).toContain(term);
  });
  it('binds all original assets and the public license to the saved source packet',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/mae-2026-10-28/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(3);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.used_in).toEqual([chapter]);
    }
    expect(read('site/public/licenses/mae-efb2a80-CC-BY-NC-4.0.txt'))
      .toBe(read('05 Источники/Originals/mae-2026-10-28/mae/LICENSE'));
  });
});
