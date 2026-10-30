import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(t:string)=>createHash('sha256').update(t).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений.md';
const edge='41f Контрастивное предобучение — SimCLR и проверка представлений|Контрастивное предобучение — SimCLR и проверка представлений';

describe('SimCLR source explanation and navigation',()=>{
  it('historical explanation preservation: inserts a thematic route without rewriting accepted neighbours',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/simclr-representations$/gm)).toHaveLength(2);
    expect(nav).toContain(`source: ${chapter}`);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41f Контрастивное предобучение');
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md')
      .replace(/<a id="pretraining-amortization-figure"><\/a>\n[\s\S]*?(?=The same accounting applies to a reusable pretrained model\.)/,'');
    expect(hash(previous.replace(edge,'42 Next-token prediction|Целевая функция предобучения')))
      .toBe('7a35177afdf70c69f947345525297e3fd9bf7aa85933d646032f6d41c4a48ce6');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md')
      .replace('41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества','41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей');
    expect(hash(next.replace('41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей','41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding')))
      .toBe('36663ad6d1759134b00f1aebf4d8e9d7538513109e1749b64f5bb3a37fcd13af');
  });
  it('preserves the source wrapper and supplies full worked programs and caveats',()=>{
    const text=read(chapter);
    const n=JSON.parse(read('05 Источники/Originals/simclr-2026-10-28/uva/SimCLR.ipynb'));
    expect(text).toContain(n.cells[8].source.join(''));
    expect(text.match(/^```python$/gm)).toHaveLength(5);
    expect(text.match(/^!\[\[/gm)).toHaveLength(4);
    for(const anchor of ['two-view-encoder','augmentation-task','positive-index','nt-xent',
      'shared-training-step','collapse-and-negatives','projection-versus-representation','linear-evaluation'])
      expect(text).toContain(`id="${anchor}"`);
    expect(text).toContain('test loader as Lightning');
    expect(text).toContain('268,435,456');
    expect(text).toContain('offset-$N$ rule');
    expect(text).toContain('cosine derivatives can vanish');
  });
  it('binds all four native original figures to the central registry',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/simclr-2026-10-28/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(4);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.derivation).toBe('pdf-page-render-crop');
      expect(row.license).toBe('CC BY 4.0');
    }
    expect(read('site/public/licenses/uva-c7d9e18-MIT.txt'))
      .toBe(read('05 Источники/Originals/simclr-2026-10-28/uva/LICENSE.md'));
    expect(read('site/public/licenses/simclr-PMLR-CC-BY-4.0.txt'))
      .toBe(read('05 Источники/Originals/simclr-2026-10-28/CC-BY-4.0.txt'));
  });
});
