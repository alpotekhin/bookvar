import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const chapter = '00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md';
const read = (p: string) => readFileSync(resolve(root,p));
const hash = (b: Buffer|string) => createHash('sha256').update(b).digest('hex');
const text = read(chapter).toString();
const assets = JSON.parse(read('00 Учебник/Assets/Figures/curated/gradient-clipping-2026-10-29/assets.json').toString());
const addition = text.split('<!-- BEGIN REVIEW: adaptive clipping 2026-10-29 -->\n')[1]
  .split('<!-- END REVIEW: adaptive clipping 2026-10-29 -->')[0];

describe('adaptive clipping in the canonical optimization explanation', () => {
  it('keeps source-language explanations, worked cases and both reference indexes', () => {
    expect(text.match(/^\x60{3}python$/gm)).toHaveLength(9);
    expect(addition.match(/^\x60{3}python$/gm)).toHaveLength(2);
    for (const anchor of ['agc-unitwise','autoclip']) {
      expect(text).toContain('id="'+anchor+'"');
      for (const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p).toString()).toContain('#'+anchor+'|');
    }
    const prose = addition.replace(/\s+/g,' ');
    for (const fact of ['first row by 1%','second by 100%',
      'relative SGD changes are now $0.01$ and $0.02$',
      'first finite gradient is therefore unchanged',
      'history and its configuration must be saved',
      'not evidence that the same threshold',
      'does not prove an identical training trajectory',
      'mathematical bound on Adam']) expect(prose).toContain(fact);
    expect(text).toContain('source_language: mixed');
  });

  it('preserves the exact original English passages and their permission evidence', () => {
    const normalize = (s:string) => s.replace(/^> ?/gm,'')
      .replace(/-\s*\n\s*/g,'').replace(/-/g,'').replace(/\s+/g,' ').trim();
    const quotes = addition.match(/(?:^>[^\n]*(?:\n|$))+/gm)!;
    expect(quotes).toHaveLength(2);
    const pdf = execFileSync('pdftotext',['-f','4','-l','4',
      resolve(root,assets[0].source_asset),'-'],{encoding:'utf8'});
    // Poppler inserts the preceding fraction's isolated F subscript here.
    const paragraph = pdf.replace('we clip\nF\ngradients','we clip\ngradients');
    expect(normalize(paragraph)).toContain(normalize(quotes[0]));
    const readme = read(assets[1].source_page_snapshot).toString();
    const explanation = readme.split('\n\n').find((p:string) =>
      p.startsWith('Training dynamics of a smaller mask inference network'))!;
    expect(normalize(quotes[1])).toBe(normalize(explanation));
    const license = execFileSync('pdftotext',
      [resolve(root,assets[0].license_evidence),'-'],{encoding:'utf8'});
    expect(license).toContain('Creative Commons Attribution 4.0 International');
    expect(read(assets[1].license_evidence).toString()).toContain('MIT License');
  });

  it('pins both complete source visuals, their original bytes and adjacent attribution', () => {
    expect(assets).toHaveLength(2);
    const registry = parse(read('05 Источники/asset-registry.yml').toString()).assets;
    for (const a of assets) {
      expect(registry.filter((r:any) => r.asset===a.asset)).toEqual([a]);
      expect(text).toContain(a.asset);
      for (const [p,h] of [[a.asset,a.sha256],[a.source_asset,a.source_asset_sha256],
        [a.license_evidence,a.license_evidence_sha256]]) expect(hash(read(p))).toBe(h);
    }
    expect(assets[0].dimensions_px).toEqual([970,510]);
    expect(assets[0].crop_box_px).toEqual([1215,265,2185,775]);
    expect(assets[0].source_page).toBe(4);
    expect(assets[0].sha256).toBe('083f5af777621f5e5d1e759360cffc650c12e5254d75ca56e1bdc7fbd17b94a9');
    expect(read(assets[1].asset)).toEqual(read(assets[1].source_asset));
    expect(assets[1].dimensions_px).toEqual([864,576]);
    expect(assets[1].frames).toBe(40);
    expect(read(assets[1].public_license_copy)).toEqual(read(assets[1].license_evidence));
  });

  it('restores the exact prior chapter, registry and indexes without editing old receipts', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-adaptive-clipping-additions.json').toString());
    expect(journal.files).toHaveLength(4);
    for (const row of journal.files) {
      let restored = beforeTopicConsolidation(read(row.source).toString(), row.source,
        root, '2026-10-29-adaptive-clipping-additions.json');
      expect(hash(restored)).toBe(row.after_sha256);
      for (const c of [...row.changes].reverse()) {
        expect(restored.split(c.new)).toHaveLength(2);
        restored = restored.replace(c.new,()=>c.old);
      }
      expect(hash(restored)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(read(row.source).toString(),row.source,root,
        '2026-10-29-retention-additions.json')).toBe(restored);
    }
  });
});
