import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const chapter='00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров.md';
const previous='00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md';
const route='textbook/pretraining/active-learning';

describe('active-learning source excerpts and publication boundaries',()=>{
  it('historical explanation preservation: adds a thematic reader chapter while preserving the previously accepted explanation',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(new RegExp(`route: ${route}$`,'gm'))).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}\n        route: ${route}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41c Active learning — запрос меток');
    const current=readBeforeReaderRouteNavigation(previous);
    const edge='**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning — запрос меток и выбор примеров]]';
    expect(current).toContain(edge);
    const restored=current.replace(edge,'**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Целевая функция предобучения]]');
    expect(createHash('sha256').update(restored).digest('hex'))
      .toBe('e9b361be8d70274f446dd3f2c7607c96e2c26b3cf74862942e5db5d3bc05def2');
  });

  it('retains original notebook calculations and all complete programs',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(5);
    expect(text.match(/^!\[\[/gm)).toHaveLength(2);
    const nb=JSON.parse(read('05 Источники/Originals/active-learning-2026-10-28/modal/uncertainty_sampling.ipynb'));
    for(const index of [3,12]) expect(text).toContain(nb.cells[index].source.join(''));
    for(const anchor of ['annotation-loop','uncertainty','committee','expected-gradient',
      'executable-loop','batch-and-diversity','evaluation-and-cost']) expect(text).toContain(`id="${anchor}"`);
    expect(text).toContain('not a learning\ncurve');
    expect(text).toContain('model-training procedure');
    expect(text).toContain('rtol=0., atol=1e-12');
  });

  it('binds unchanged author images and their MIT notice to the reader',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/active-learning-2026-10-28/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(2);
    for(const row of rows){
      expect(registry.find((x:any)=>x.id===row.id)).toEqual(row);
      expect(row.license).toBe('MIT');
      expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect(readFileSync(resolve(root,row.public_license_copy))).toEqual(readFileSync(resolve(root,row.license_evidence)));
    }
  });
});
