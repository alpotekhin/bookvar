import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding.md';
const route='textbook/pretraining/geometric-coresets';

describe('geometric selection explanation and publication',()=>{
  it('historical explanation preservation: adds the reader route and changes only adjacent navigation edges',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(new RegExp(`route: ${route}$`,'gm'))).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}\n        route: ${route}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41d Геометрический отбор');
    const prefix='**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/';
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров.md');
    const restored=previous.replace(prefix+'41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding]]',
      prefix+'42 Next-token prediction|Целевая функция предобучения]]');
    expect(hash(restored)).toBe('c1970f259e599a045fb49e6f47745a80e7e4d7adca764e4ff500b1bec7862f81');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md')
      .replace('41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества','41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей');
    const before=next.replace('41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей',
      '41c Active learning — запрос меток и выбор примеров|Active learning — запрос меток и выбор примеров');
    expect(hash(before)).toBe('3889d5f75e7b62c9ff21dc127638200efb97133d8aa75ea1024845bda8da028c');
  });

  it('retains each complete mechanism, original proof and executable example',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(6);
    expect(text.match(/^!\[\[/gm)).toHaveLength(3);
    for(const anchor of ['coverage-objective','farthest-first','approximation-guarantee',
      'implementation','representation-and-metric','outliers-and-labels',
      'coreset-versus-centres','herding']) expect(text).toContain(`id="${anchor}"`);
    expect(text).toContain('at least two of the points $c_1,c_2,\\ldots,c_{k+1}$');
    expect(text).toContain('factor-four bound');
    expect(text).toContain('Exact distance ties go to the smallest ID');
    expect(text).toContain('No figures or extended excerpts');
    expect(text).toContain('source_language: en');
  });

  it('binds the whole six-panel source figure and actual permission snapshot',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/geometric-coresets-2026-10-28/assets.json'));
    expect(rows).toHaveLength(1);
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(registry.find((r:any)=>r.id===rows[0].id)).toEqual(rows[0]);
    expect(rows[0].dimensions_px).toEqual([1500,856]);
    expect(rows[0].source_page).toBe(14);
  });
});
