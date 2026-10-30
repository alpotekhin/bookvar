import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {withoutNativeWindowExample} from './helpers/nn-native-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const page='00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md';
const prefix='00 Учебник/Assets/Figures/curated/group-equivariance-2026-10-26/';
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
describe('source-original rotation-equivariance extension',()=>{
  it('preserves the previously accepted whole CNN chapter outside the bounded insertion',()=>{
    const chapter=readFileSync(resolve(root,page),'utf8');
    const restored=withoutNativeWindowExample(chapter).replace(/<!-- group-equivariance 2026-10-26 -->\n[\s\S]*?<!-- \/group-equivariance -->\n\n/,'');
    expect(hash(restored)).toBe('c6cceb20910882ae1408dc7013c286ce77c3e00b0ab32070505e7df6b6bf5355');
    expect(chapter.match(/^```python$/gm)).toHaveLength(21);
    const original=readFileSync(resolve(root,'05 Источники/Originals/group-equivariance-2026-10-26/escnn-README.md'),'utf8');
    for(const name of ['escnn-feature-fields','escnn-demo']) {
      const excerpt=chapter.split('<!-- source-excerpt: '+name+' -->\n')[1].split('\n<!-- /source-excerpt -->')[0]
        .split('\n').map(line=>line.replace(/^> ?/,'')).join('\n');
      expect(original.replace(/\s+/g,' ')).toContain(excerpt.replace(/\s+/g,' '));
    }
  });
  it('preserves the original figures, source hashes and full BSD Clear notice',()=>{
    const assets=JSON.parse(readFileSync(resolve(root,prefix+'assets.json'),'utf8'));
    const central=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
    expect(assets).toHaveLength(3);
    for(const asset of assets) {
      expect(central.find((row:{id:string})=>row.id===asset.id)).toEqual(asset);
      expect(asset.license).toBe('BSD-3-Clause-Clear');
      expect(asset.commit).toBe('9ad44cc37d694d9c805c4fef4d16722a254b3bf2');
      expect(asset.crop_box_px).toBeUndefined();
      for(const [file,expected] of [[asset.asset,asset.sha256],
        [asset.source_asset,asset.source_asset_sha256],
        [asset.source_page_snapshot,asset.source_page_sha256],
        [asset.license_evidence,asset.license_evidence_sha256]])
        expect(hash(readFileSync(resolve(root,file)))).toBe(expected);
      expect(readFileSync(resolve(root,asset.asset)).equals(readFileSync(resolve(root,asset.source_asset)))).toBe(true);
    }
    expect(readFileSync(resolve(root,prefix+'LICENSE'))).toEqual(readFileSync(resolve(root,
      '05 Источники/Originals/group-equivariance-2026-10-26/escnn-LICENSE')));
  });
});
