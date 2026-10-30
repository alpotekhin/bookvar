import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const chapter = '00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts.md';
const read = (p: string) => readFileSync(resolve(root, p), 'utf8');
const hash = (b: string | Buffer) => createHash('sha256').update(b).digest('hex');
const body = read(chapter);
const assets = JSON.parse(read('00 Учебник/Assets/Figures/curated/retention-2026-10-29/assets.json'));

describe('retention mechanisms inside the existing fine-tuning chapter', () => {
  it('publishes native-English reasoning, four CPU cases and both locale topic links', () => {
    expect(body).toContain('source_language: en');
    expect(body.match(/^`{3}python$/gm)).toHaveLength(18);
    for (const anchor of ['retaining-capabilities', 'fine-tuning-replay', 'l2-sp',
      'elastic-weight-consolidation', 'fisher-for-ewc']) {
      expect(body).toContain('id="' + anchor + '"');
      for (const p of ['01 Справочник/_index.md', 'en/01 Reference/_index.md'])
        expect(read(p)).toContain('#' + anchor + '|');
    }
    const prose = body.replace(/\s+/g, ' ');
    for (const fact of ['mean of their squared gradients is 0.25',
      'A penalty on the frozen base alone is identically zero',
      'supplied toy coefficients', 'no old-task retraining',
      '62.6%, 72.0% and 72.5%', 'not measurements on LLMs',
      'different algorithms', 'unbiased gradient estimate'])
      expect(prose).toContain(fact);
  });

  it('keeps the original English passage and verifies the PMLR permission evidence', () => {
    const normalized = (s: string) => s.replace(/^> ?/gm, '')
      .replace(/-\s*\n\s*/g, '-').replace(/\s+/g, ' ').trim();
    const original = execFileSync('pdftotext', ['-f', '1', '-l', '1',
      resolve(root, assets[0].source_asset), '-'], {encoding: 'utf8'});
    const quote = body.match(/(?:^>[^\n]*(?:\n|$))+/gm)!
      .find(s => s.startsWith('> We thus advocate for a coherent'));
    expect(quote).toBeDefined();
    expect(normalized(original)).toContain(normalized(quote!));
    const permission = execFileSync('pdftotext',
      [resolve(root, assets[0].license_evidence), '-'], {encoding:'utf8'});
    expect(normalized(permission)).toContain('Creative Commons Attribution 4.0 International');
  });

  it('retains complete original panels and captions with byte-exact PDF crops', () => {
    expect(assets).toHaveLength(2);
    const registry = parse(read('05 Источники/asset-registry.yml')).assets;
    const expected = [
      ['37ee7a43639cb4abda0d3ad54b63c18d89ce0dff7a09ad4d9bcec2236c33bcf1',
        [205,710,1190,1950], [985,1240]],
      ['e344b0d6cb6ce3feb7e5a601fdd13517b8dc9e0e3beda52ef8d68053b90fed25',
        [1220,730,2195,1305], [975,575]],
    ];
    for (const [i, asset] of assets.entries()) {
      expect(registry.filter((a: {asset: string}) => a.asset === asset.asset)).toEqual([asset]);
      expect(asset.sha256).toBe(expected[i][0]);
      expect(asset.crop_box_px).toEqual(expected[i][1]);
      expect(asset.dimensions_px).toEqual(expected[i][2]);
      expect(asset.source_page).toBe(6);
      expect(asset.render_page_size_px).toEqual([2448,3168]);
      expect(asset.license).toBe('CC BY 4.0');
      expect(body).toContain(asset.asset);
      for (const [p, sha] of [[asset.asset,asset.sha256],
        [asset.source_asset,asset.source_asset_sha256],
        [asset.license_evidence,asset.license_evidence_sha256]])
        expect(hash(readFileSync(resolve(root,p)))).toBe(sha);
      const [x,y,right,bottom] = asset.crop_box_px;
      const raster = execFileSync('pdftoppm', ['-f','6','-l','6','-r','288',
        '-x',String(x),'-y',String(y),'-W',String(right-x),'-H',String(bottom-y),
        '-singlefile','-png',resolve(root,asset.source_asset)], {maxBuffer:1024*1024});
      expect(raster).toEqual(readFileSync(resolve(root,asset.asset)));
    }
  });

  it('restores the exact prior chapter and indexes without rebaselining old receipts', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-retention-additions.json'));
    expect(journal.files).toHaveLength(4);
    for (const row of journal.files) {
      let text = beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-retention-additions.json');
      expect(hash(text)).toBe(row.after_sha256);
      for (const change of [...row.changes].reverse()) {
        expect(text.split(change.new)).toHaveLength(2);
        text = text.replace(change.new, () => change.old);
      }
      expect(hash(text)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-bert-objectives-additions.json')).toBe(text);
      expect(() => beforeTopicConsolidation(read(row.source)+'\nUnreviewed text.\n',
        row.source,root,'2026-10-29-bert-objectives-additions.json')).toThrow();
    }
  });
});
