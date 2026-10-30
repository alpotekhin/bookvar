import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {beforeSelectionGapsAddition} from '../tools/selection-gaps-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const page='00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md';
const prefix='00 Учебник/Assets/Figures/curated/dlrm-2026-10-26/';
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
describe('DLRM integration into the existing ranking chapter',()=>{
  it('preserves all accepted BPR, AutoRec, NCF, FM and DeepFM explanations',()=>{
    const chapter=readFileSync(resolve(root,page),'utf8');
    const restored=beforeSelectionGapsAddition(chapter,page).replace(/<!-- dlrm 2026-10-26 -->\n[\s\S]*?<!-- \/dlrm -->\n\n/,'')
      .replace('last_updated: 2026-10-26','last_updated: 2026-10-25');
    expect(hash(restored)).toBe('ecec78266fddd5dcfa2463d9c448f013ae63e7101e8f50052fe4e3895c092f25');
    expect(chapter.match(/^```python$/gm)).toHaveLength(23);
    const normalize=(text:string)=>text.replace(/`/g,'').replace(/\s+/g,' ');
    for(const [name,file] of [['torchrec-pooling','torchrec-high-level-arch.rst'],
      ['torchrec-communication','torchrec-concepts.rst']]) {
      const excerpt=chapter.split('<!-- source-excerpt: '+name+' -->\n')[1].split('\n<!-- /source-excerpt -->')[0]
        .split('\n').map(line=>line.replace(/^> ?/,'')).join('\n');
      const original=readFileSync(resolve(root,'05 Источники/Originals/dlrm-2026-10-26/'+file),'utf8');
      expect(normalize(original)).toContain(normalize(excerpt));
    }
  });
  it('keeps both complete TorchRec PNGs and their redistribution notice',()=>{
    const assets=JSON.parse(readFileSync(resolve(root,prefix+'assets.json'),'utf8'));
    const central=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
    expect(assets).toHaveLength(3);
    for(const asset of assets) {
      expect(central.find((row:{id:string})=>row.id===asset.id)).toEqual(asset);
      expect(asset.license).toBe('BSD-3-Clause');
      expect(asset.commit).toBe('36ae8ed89ac78e34a18ab00a08b93afe19410b55');
      expect(asset.crop_box_px).toBeUndefined();
      for(const [file,expected] of [[asset.asset,asset.sha256],
        [asset.source_asset,asset.source_asset_sha256],
        [asset.source_page_snapshot,asset.source_page_sha256],
        [asset.license_evidence,asset.license_evidence_sha256]])
        expect(hash(readFileSync(resolve(root,file)))).toBe(expected);
      expect(readFileSync(resolve(root,asset.asset)).equals(readFileSync(resolve(root,asset.source_asset)))).toBe(true);
    }
    expect(readFileSync(resolve(root,prefix+'LICENSE')).equals(readFileSync(resolve(root,
      'site/public/licenses/torchrec-BSD-3-Clause.txt')))).toBe(true);
    expect(hash(readFileSync(resolve(root,'site/public/licenses/dlrm-MIT.txt'))))
      .toBe('52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85');
  });
});
