import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { collectiveStart, collectiveEnd, withoutCollectiveDestinations } from './helpers/collective-destinations-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md';
const originals = '05 Источники/Originals/collective-destinations-2026-10-27';
const read = (path: string) => readFile(join(root, path), 'utf8');
const digest = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

describe('collective destinations', () => {
  it('preserves every byte of accepted DDP outside the bounded addition and source/date metadata', async () => {
    const text = await read(chapter);
    expect(text.split(collectiveStart)).toHaveLength(2);
    expect(text.split(collectiveEnd)).toHaveLength(2);
    expect(digest(withoutCollectiveDestinations(text)))
      .toBe('e33ee25e5047a10c3814e0f41c38fef164abe5364c1a30c7407a8fb3a91a9211');
  });

  it('retains the original English gather/all-gather rules with code-markup conversion only', async () => {
    const source = await read(`${originals}/dist_tuto.rst`);
    const selected = source.slice(source.indexOf('-  ``dist.gather('), source.indexOf('-  ``dist.barrier('))
      .replaceAll('``', '`').trim();
    const text = (await read(chapter)).split('<!-- BEGIN SOURCE: PyTorch tutorial destination rules -->')[1]
      .split('<!-- END SOURCE: PyTorch tutorial destination rules -->')[0].trim();
    const normalized = (value: string) => value.replace(/\s+/g, ' ');
    expect(normalized(text)).toBe(normalized(selected));
  });

  it('retains all complete native PNGs and the full public source license', async () => {
    const assets = '00 Учебник/Assets/Figures/curated/collective-destinations-2026-10-27';
    const rows = JSON.parse(await read(`${assets}/assets.json`));
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      const source = await readFile(join(root, row.source_asset));
      expect(digest(source)).toBe(row.sha256);
      expect(await readFile(join(root, row.asset))).toEqual(source);
      expect(source.readUInt32BE(16)).toBe(row.dimensions_px[0]);
      expect(source.readUInt32BE(20)).toBe(row.dimensions_px[1]);
    }
    expect(await read('site/public/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt'))
      .toBe(await read(`${originals}/LICENSE`));
  });

  it('retains examples and separates ownership, local indexing and performance evidence', async () => {
    const text = (await read(chapter)).split(collectiveStart)[1].split(collectiveEnd)[0];
    for (const fact of ['[2,12,22,32]', '[4,15,16,23,34,35]', 'sends five', 'receives six',
      'CPU/Gloo', 'not a prescribed physical network topology', 'experimental',
      'not the list', 'different address spaces']) expect(text).toContain(fact);
    expect(text.match(/^```python$/gm)).toHaveLength(2);
  });
});
