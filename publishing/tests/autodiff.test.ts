import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {withoutAutodiffAdditions} from './helpers/autodiff-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const digest=(value:Buffer|string)=>createHash('sha256').update(value).digest('hex');
const chapter='00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md';
const text=read(chapter).toString();
const qmd=read('05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').toString();
const fences=Array.from(text.matchAll(/^```python\n([\s\S]*?)^```/gm),r=>r[1].trim());

describe('source-faithful AD and custom differentiation',()=>{
  it('preserves every byte of the accepted backpropagation chapter outside two additions',()=>{
    expect(digest(withoutAutodiffAdditions(text)))
      .toBe('d329325d295f0fc50ddbf013412c96e0f24f8a31bd6733754e08ae23a5baa198');
    expect(fences).toHaveLength(20);
    expect(text.match(/!\[\[/g)).toHaveLength(7);
  });

  it('retains both complete original Harvard examples without translating comments',()=>{
    for(const [label,imports] of [['lst-forward_mode_ad','from math import sin, cos\n\n'],
      ['lst-custom-autograd-function','import torch\n\n']]) {
      const code=qmd.split('{#'+label)[1].split('```{.python}\n')[1].split('```')[0].trim();
      expect(fences).toContain(imports+code);
    }
    expect(text).toContain('The example hardcodes the unit seed');
    expect(text).toContain('The following editorial variant');
    expect(text).toContain('does not supply it');
    expect(text).toContain('do not establish a `vmap` rule');
    expect(text).toContain('gradgradcheck(MultiplyAdd.apply, broadcast_inputs)');
  });

  it('pins the contract documentation and preserves complete graph/trace images',()=>{
    expect(digest(read('05 Источники/Originals/frameworks-2026-10-27/pytorch-5c488690-extending.md')))
      .toBe('adf937440d119ed1b23952e76d784b76db349ce990068ae8b9a41453eb2a6a78');
    expect(digest(read('05 Источники/Originals/frameworks-2026-10-27/pytorch-5c488690-extending.func.md')))
      .toBe('a9ba3badd4dc5627b9470fe7536f39bbbd08272b65f3922bae23d5ae015d8f48');
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/assets.json').toString());
    const registry=parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(rows.map((r:{source_page:number})=>r.source_page)).toEqual([9,10,13]);
    expect(rows.map((r:{dimensions_px:number[]})=>r.dimensions_px))
      .toEqual([[2476,1015],[2476,1237],[2476,1543]]);
    for(const row of rows) {
      expect(registry.find((r:{id:string})=>r.id===row.id)).toEqual(row);
      expect(text).toContain(row.asset);
      for(const [path,sha] of [[row.asset,row.sha256],
        [row.source_asset,row.source_asset_sha256],
        [row.license_evidence,row.license_evidence_sha256]]) expect(digest(read(path))).toBe(sha);
    }
  });
});
