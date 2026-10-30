import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const sha=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md';
const current=read(chapter).toString();
const marker=/<!-- BEGIN REVIEW: placement dimensions 2026-10-28 -->\n[\s\S]*?<!-- END REVIEW: placement dimensions 2026-10-28 -->\n\n/;

describe('original Harvard placement diagram in the distributed narrative',()=>{
  it('preserves all 917 accepted chapter lines outside the addition and two dates',()=>{
    const baseline=beforeReaderRouteNavigation(current, chapter).replace(marker,'').replace('last_updated: 2026-10-28','last_updated: 2026-10-23')
      .replace('last_verified: 2026-10-28','last_verified: 2026-10-23');
    expect(sha(baseline)).toBe('fe1a7b12bc09cabd25750a0f14ca805217c05f72359549d3cbd611e3906a02a9');
    expect(current.match(marker)).toHaveLength(1);
    expect(current.match(marker)![0].match(/^```python$/gm)).toHaveLength(1);
  });
  it('retains the full original caption and exact source SVG rather than its preview',()=>{
    const qmd=read('05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').toString();
    const caption=qmd.match(/\{#fig-3d-parallelism[^\n]*fig-cap="([^"]*)"/)![1];
    expect(current.split('<!-- BEGIN SOURCE: Harvard device placement caption -->')[1]
      .split('<!-- END SOURCE: Harvard device placement caption -->')[0].trim()).toBe(caption);
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/frameworks-native-2026-10-27/assets.json').toString());
    const entry=rows.find((r:any)=>r.id==='curated-frameworks-placement-6c3e1e350e56');
    expect(read(entry.asset)).toEqual(read(entry.source_asset));
    expect(sha(read(entry.asset))).toBe('6c3e1e350e5653c517ac8b82ec4a7184eceb7573ae8320b56c4ca222861c6b12');
    expect(entry.source_viewbox).toEqual([0,0,274.96933,168.46268]);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(parse(read('05 Источники/asset-registry.yml').toString()).assets.find((r:any)=>r.id===entry.id)).toEqual(entry);
    expect(entry.used_in).toEqual([chapter]);
    const css=read('site/src/styles/custom.css').toString();
    expect(css).toContain("/frameworks-native-2026-10-27/device-placement-dimensions.svg");
    expect(css.split("/frameworks-native-2026-10-27/device-placement-dimensions.svg")[1]
      .split('}')[0]).toContain('width: min(100%, 40rem)');
  });
  it('keeps the physical-topology, group-ownership and batch boundaries',()=>{
    const addition=current.match(marker)![0];
    for(const text of ['32 devices, not eight','8p+4r+t','[0, 8, 16, 24]',
      '[7, 15, 23, 31]','30 sequences','480', 'no network topology or pipeline schedule']) {
      if(text==='no network topology or pipeline schedule') expect(addition).toContain('not a network\ntopology or pipeline schedule');
      else expect(addition).toContain(text);
    }
    expect(addition).toContain('not a claim\nthat a framework assigns physical GPU IDs');
    expect(addition).toContain('must not average the weights of different layers');
  });
});
