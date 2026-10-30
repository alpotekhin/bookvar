import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const chapter = '00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md';
const assetRoot = '00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/';
const read = (path: string) => readFileSync(resolve(root, path));
const sha = (path: string) => createHash('sha256').update(read(path)).digest('hex');
const assets = JSON.parse(read(assetRoot + 'assets.json').toString());

describe('Plan-Seq-Learn original source transfer', () => {
  it('keeps transfer commentary in a closed disclosure without hiding the explanation', () => {
    const text = read(chapter).toString();
    const disclosure = text.match(/<details>\n<summary>Source text and attribution<\/summary>\n\n([\s\S]*?)\n<\/details>/);
    expect(disclosure).not.toBeNull();
    expect(disclosure![1].includes('Ниже сохранены полные английские объяснения')).toBe(true);
    expect(disclosure![1].includes('Карточка статьи и авторские материалы')).toBe(true);
    expect(disclosure![1].match(/^>|!\[\[|^####/gm) ?? []).toEqual([]);
    expect(text.indexOf('</details>', text.indexOf('<summary>Source text and attribution')))
      .toBeLessThan(text.indexOf(assetRoot + 'method.png'));
    const prior = beforeTopicConsolidation(text, chapter, root,
      '2026-10-30-modern-route-editorial-additions.json');
    expect(createHash('sha256').update(prior).digest('hex')).toBe(
      '718e68fb6cf58e941e10d76e02acc0761e06a19d8c4bb8496ac6dd3353bcd628');
    for (const pattern of [/^>[^\n]*$/gm, /^~~~python\n[\s\S]*?^~~~$/gm,
      /^\$\$\n[\s\S]*?^\$\$$/gm, /!\[\[[^\]]+\]\]/g])
      expect(text.match(pattern)).toEqual(prior.match(pattern));
  });
  it('keeps the original chapter outside the bounded addition', () => {
    const text = read(chapter).toString();
    const reviewed = beforeReaderRouteNavigation(text, chapter, root);
    const baseline = reviewed.replace(/<!-- BEGIN SOURCE: Plan-Seq-Learn 2405\.01534v1 -->[\s\S]*?<!-- END SOURCE: Plan-Seq-Learn 2405\.01534v1 -->\n\n/, '')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-10-20')
      .replace('  - https://arxiv.org/abs/2405.01534v1\n', '')
      .replace('  - meeting-06-slides-plan-sequence-learn\n', '');
    expect(createHash('sha256').update(baseline).digest('hex')).toBe(
      '15a26693fe0f017f3d59be94319205afe7b510051a81b63d362d504acec17562');
    for (const section of ['S3.SS2', 'S3.SS3', 'S3.SS4', 'S3.SS5', 'A4.SS3.SSS1', 'A2.SS3'])
      expect(text).toContain('<!-- source-psl:' + section + ' -->');
    expect(text).toContain('без перевода и стилистической правки');
    expect(text).toContain('несогласованность текста с его же планом');
    expect(text).toContain('0{,}9^4=0{,}6561');
    expect(text).toContain('в симуляторах, а не на физическом роботе');
  });

  it('does not hide an unrelated prose change when restoring historical navigation', () => {
    const text = read(chapter).toString();
    expect(() => beforeReaderRouteNavigation(text + '\nUnreviewed explanation.\n', chapter, root))
      .toThrow('Unreviewed topic consolidation revision: ' + chapter);
  });

  it('pins three untouched PNGs and their article-license evidence', () => {
    expect(assets).toHaveLength(3);
    const registry = parse(read('05 Источники/asset-registry.yml').toString(), { merge: true });
    expect(registry.assets.filter((row: any) => row.asset.startsWith(assetRoot))).toEqual(assets);
    const text = read(chapter).toString();
    for (const row of assets) {
      expect(row.license).toBe('CC BY 4.0');
      expect(row.derivation).toBe('source-raster-copy');
      expect(sha(row.asset)).toBe(row.sha256);
      expect(sha(row.source_page_snapshot)).toBe(row.source_page_sha256);
      expect(sha(row.license_evidence)).toBe(row.license_evidence_sha256);
      const png = read(row.asset);
      expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual(row.dimensions_px);
      expect(text).toContain(row.asset);
      expect(read(row.license_evidence).toString()).toContain('creativecommons.org/licenses/by/4.0/');
      expect(read(row.public_license_copy).toString()).toContain('Original English retained');
    }
  });
});
