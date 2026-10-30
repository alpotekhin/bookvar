import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const sha = (path: string) => createHash('sha256').update(read(path)).digest('hex');
const intro = '00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training.md';
const collection = '00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md';
const asset = '00 Учебник/Assets/Figures/curated/berkeley-agents-2025/pretraining/m04-p147-two-stage-training.png';

describe('OLMo original recipe and corpus attribution correction', () => {
  it('preserves the accepted introduction outside the explicit OLMo expansion', () => {
    const previousParagraph = `В рецепте OLMo 2 поздняя стадия сопровождалась линейным снижением скорости
обучения до нуля. Кроме того, финальные веса получали усреднением нескольких
моделей, обученных с разным порядком данных. Поэтому сравнение результата
до и после этой стадии нельзя выдавать за чистый эксперимент «поменяли только
корпус». Чтобы измерить именно влияние смеси, нужно сравнить варианты с одним
начальным состоянием, бюджетом, расписанием и протоколом оценивания.

`;
    const baseline = beforeReaderRouteNavigation(read(intro).toString(), intro)
      .replace(/<a id="olmo-two-stage-recipe"><\/a>[\s\S]*?(?=Для OLMo 2-Instruct далее)/, previousParagraph)
      .replaceAll('2026-10-26', '2026-10-23')
      .replace('  - meeting-04-slides-olmo-pretraining\n', '');
    expect(createHash('sha256').update(baseline).digest('hex')).toBe(
      '01c2164280d1f9b7c31a2db26e32b60f184dc1b8947fa5c62b64a80004810df0');
  });

  it('changes only the mistaken author, pinned citation and verification dates in chapter 41', () => {
    const baseline = beforeReaderRouteNavigation(read(collection).toString(), collection)
      .replaceAll('2026-10-26', '2026-10-23')
      .replace('[William Held и соавторы, Optimizing Pretraining Data Mixtures with LLM-Estimated Utility, §3.2](https://arxiv.org/html/2501.11747v1#S3.SS2)',
        '[Shen и соавторы, Optimizing Pretraining Data Mixtures](https://arxiv.org/abs/2501.11747)');
    expect(createHash('sha256').update(baseline).digest('hex')).toBe(
      'aa947518e59dfddd31b1c1c58bfdeaf364708108965b04272cad62f4951e2397');
  });

  it('pins the full non-confidential slide without assigning the paper license to it', () => {
    const registry = parse(read('05 Источники/asset-registry.yml').toString(), { merge: true });
    const row = registry.assets.find((candidate: any) => candidate.asset === asset);
    expect(row).toBeTruthy();
    expect(sha(asset)).toBe(row.sha256);
    expect(sha(row.source_asset)).toBe(row.source_asset_sha256);
    expect(row.source_page).toBe(147);
    expect(row.derivation).toBe('pdf-page-render');
    expect(row.license).toBe('User permission record (not a license)');
    expect(row.provenance_confirmation).toBe('user-confirmed-open-materials');
    const png = read(asset);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([2400, 1350]);
    expect(row.crop_box_px).toBeUndefined();
    expect(read(intro).toString()).toContain(asset);
    const notice = read(row.public_license_copy).toString();
    expect(notice).toContain('No CC BY license is asserted for the course slide');
    expect(notice).toContain('Three complete paragraphs');
    expect(sha('05 Источники/Originals/olmo-pretraining-2026-10-26/article.html')).toBe(
      '68e5fd8ffb899793b8c517770f9e8611946bb12a2a23c84e3641615c3a4d67d0');
  });
});
