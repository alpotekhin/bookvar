import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';

const root=resolve(import.meta.dirname,'../..');
const page='00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md';
const prefix='00 Учебник/Assets/Figures/curated/rnn-execution-2026-10-26/';
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
describe('recurrent execution extension',()=>{
  it('preserves the accepted RNN chapter and reuses the exact D2L explanation',()=>{
    const chapter=readFileSync(resolve(root,page),'utf8');
    const restored=chapter.replace(/<!-- rnn-execution 2026-10-26 -->\n[\s\S]*?<!-- \/rnn-execution -->\n\n/,'')
      .replace('last_updated: 2026-10-26','last_updated: 2026-10-21');
    expect(hash(restored)).toBe('d9cb27f7d76de93f41fe3f252ce1da1bcf53897197decbabaf257dbddfabaa25');
    expect(chapter.match(/^```python$/gm)).toHaveLength(6);
    const original=readFileSync(resolve(root,'05 Источники/Originals/rnn-execution-2026-10-26/d2l-deep-rnn.md'),'utf8');
    const excerpt=chapter.split('<!-- source-excerpt: d2l-stacked-inputs -->\n')[1].split('\n<!-- /source-excerpt -->')[0]
      .split('\n').map(line=>line.replace(/^> ?/,'')).join('\n');
    expect(original.replace(/\s+/g,' ')).toContain(excerpt.replace(/\s+/g,' '));
  });
  it('registers the complete original figure with pinned source and license',()=>{
    const assets=JSON.parse(readFileSync(resolve(root,prefix+'assets.json'),'utf8'));
    const central=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
    expect(assets).toHaveLength(1);
    const asset=assets[0];
    expect(central.find((row:{id:string})=>row.id===asset.id)).toEqual(asset);
    expect(asset.license).toBe('CC BY-SA 4.0');
    expect(asset.commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
    expect(asset.crop_box_px).toBeUndefined();
    for(const [file,expected] of [[asset.asset,asset.sha256],
      [asset.source_asset,asset.source_asset_sha256],
      [asset.source_page_snapshot,asset.source_page_sha256],
      [asset.license_evidence,asset.license_evidence_sha256]])
      expect(hash(readFileSync(resolve(root,file)))).toBe(expected);
    expect(readFileSync(resolve(root,asset.asset)).equals(readFileSync(resolve(root,asset.source_asset)))).toBe(true);
  });
});
