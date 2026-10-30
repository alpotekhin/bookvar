import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(t:string)=>createHash('sha256').update(t).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder.md';
const edge='41g Momentum contrast — MoCo, очередь и key encoder|Momentum contrast — MoCo, очередь и key encoder';

describe('MoCo source explanation',()=>{
  it('historical explanation preservation: inserts the route while preserving accepted explanation bodies',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/moco-queue-momentum$/gm)).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41g Momentum contrast');
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений.md');
    expect(hash(previous.replace(edge,'42 Next-token prediction|Целевая функция предобучения')))
      .toBe('b973c7b056664c775eca19f06e448c3572264447bbc04c74f06da82aa89818f2');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md')
      .replace('41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества','41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей');
    expect(hash(next.replace('41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей','41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding')))
      .toBe('36663ad6d1759134b00f1aebf4d8e9d7538513109e1749b64f5bb3a37fcd13af');
  });
  it('keeps worked mechanisms, original diagrams and explicit scope',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(6);
    expect(text.match(/^!\[\[/gm)).toHaveLength(2);
    for(const anchor of ['dictionary','momentum-encoder','fifo-queue','moco-loss',
      'queue-snapshot','complete-step','distributed-shuffle','cost-and-versions'])
      expect(text).toContain(`id="${anchor}"`);
    expect(text).toContain('shares its storage');
    expect(text).toContain('33,554,432');
    expect(text).toContain('67,109,888');
    expect(text).toContain('without image IDs');
    expect(text).toContain('batch-norm running');
  });
  it('binds the native diagram and the reused SimCLR comparison to provenance',()=>{
    const row=JSON.parse(read('00 Учебник/Assets/Figures/curated/moco-2026-10-28/assets.json'))[0];
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
    const reused=JSON.parse(read('00 Учебник/Assets/Figures/curated/simclr-2026-10-28/assets.json'))
      .find((r:any)=>r.asset.endsWith('/framework.png'));
    expect(reused.used_in).toContain(chapter);
    expect(registry.find((r:any)=>r.id===reused.id)).toEqual(reused);
    expect(read('site/public/licenses/moco-132aea9-MIT.txt'))
      .toBe(read('05 Источники/Originals/moco-2026-10-28/moco/LICENSE'));
    expect(read('site/public/licenses/moco-aea97d4-CC-BY-NC-4.0.txt'))
      .toBe(read('05 Источники/Originals/moco-2026-10-28/diagram-release/LICENSE'));
  });
});
