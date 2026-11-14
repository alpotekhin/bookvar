import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const asset = '00 Учебник/Assets/Figures/curated/responsible-audit-2026-11-09/gender-shades-figure3.png';

describe('responsible ML evaluation examples', () => {
  it('exhibits two outcome completions with the same observed log and opposite TPR gaps', () => {
    expect(72 / 80).toBe(18 / 20);
    const gaps: number[] = [];
    for (let repayA = 0; repayA <= 20; repayA++)
      for (let repayB = 0; repayB <= 80; repayB++)
        gaps.push(72 / (72 + repayA) - 18 / (18 + repayB));
    expect(gaps.length).toBe(1701);
    expect(Math.min(...gaps)).toBeCloseTo(-.2173913043, 9);
    expect(Math.max(...gaps)).toBeCloseTo(.8163265306, 9);
    expect(gaps[20 * 81]).toBe(Math.min(...gaps));
    expect(gaps[80]).toBe(Math.max(...gaps));
  });

  it('reconstructs the weighted historical error from the rounded subgroup rates', () => {
    const shares = [.213, .250, .233, .303];
    const errors = [.347, .120, .071, .003];
    const total = shares.reduce((sum, weight, i) => sum + weight * errors[i], 0);
    expect(500 * .044).toBe(22);
    expect(shares[0] * errors[0]).toBeCloseTo(.073911, 10);
    expect((100 * total).toFixed(1)).toBe('12.1');
  });

  it('pins the full original plot crop and its PMLR attribution', () => {
    const png = read(asset);
    const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());
    const row = registry.assets.find((x: { asset: string }) => x.asset === asset);
    expect(row).toMatchObject({
      source_url: 'https://proceedings.mlr.press/v81/buolamwini18a.html',
      source_asset: '05 Источники/Originals/responsible-2026-10-20/buolamwini18a.pdf',
      source_page: 6,
      render_dpi: 360,
      render_page_size_px: [3060, 3960],
      crop_box_px: [1544, 445, 2621, 825],
      license: 'CC BY 4.0',
      rights_status: 'source-license-verified',
      used_in: ['00 Учебник/18 Evaluation и методология/59a Responsible systems.md'],
      sha256: '3c0aed571f22a84f1c1171acf44a46fdf56636855af75fcbb094c1e28821543f',
    });
    expect(createHash('sha256').update(png).digest('hex')).toBe(row.sha256);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1077, 380]);
  });
});
