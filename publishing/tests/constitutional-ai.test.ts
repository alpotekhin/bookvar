import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p:string) => readFileSync(resolve(root,p));
const hash = (x:Buffer|string) => createHash('sha256').update(x).digest('hex');
const chapter = '00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md';

describe('Constitutional AI data construction and preference targets', () => {
  it('executes every published numerical example without an LLM call', () => {
    const output = execFileSync('python3', ['publishing/tools/check_constitutional_ai_examples.py'],
      { cwd:root, encoding:'utf8' });
    const report = JSON.parse(output.slice(0,output.indexOf('\nPASS ')));
    expect(report.chapter_sha256).toBe(hash(read(chapter)));
    expect(report.published_python_fences).toBe(2);
    expect(report.gradient_checks).toBe(45);
    expect(report.language_model_calls).toBe(0);
    expect(report.q_a).toBeCloseTo(.2,14);
    expect(read(chapter).toString()).toContain('from all passes, not only the final version');
    expect(read(chapter).toString()).toContain('A pretrained student is');
    expect(read(chapter).toString()).toContain('uses a helpful RLHF model for feedback');
    expect(read(chapter).toString()).toContain('Human helpfulness labels remain');
  });

  it('keeps the complete original comparison figure and its actual license evidence', () => {
    const [a] = JSON.parse(read('00 Учебник/Assets/Figures/curated/constitutional-ai-2026-10-29/assets.json').toString());
    const rows = parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(rows.filter((x:any)=>x.asset===a.asset)).toEqual([a]);
    for(const [p,h] of [[a.asset,a.sha256],[a.source_asset,a.source_asset_sha256],
      [a.license_evidence,a.license_evidence_sha256]]) expect(hash(read(p))).toBe(h);
    expect(read(a.source_asset).subarray(0,5).toString()).toBe('%PDF-');
    expect(read(a.license_evidence).toString()).toContain('creativecommons.org/licenses/by/4.0');
    expect(a.license).toBe('CC BY 4.0');
    expect(a.source_page).toBe(9);
    expect(a.dimensions_px).toEqual([1780,750]);
    expect(a.crop_box_px).toEqual([340,280,2120,1030]);
    expect(read(chapter).toString()).toContain('full original caption');
  });

  it('uses the unmodified Lee et al. RLHF/RLAIF diagram with a pinned version and verified license', () => {
    const path = '00 Учебник/Assets/Figures/curated/rlaif-lee-2026-11-22/rlhf-vs-rlaif.png';
    const licensedPage = '05 Источники/Originals/rlaif-lee-2026-11-22/arxiv-abs-2309.00267v2.html';
    const digest = 'edb7992b19f76c3c5ac0f06e5390c9866c38c5bd5116eee52983b4630412e39e';
    const bytes = read(path);
    const rows = parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(rows.filter((entry: { asset: string }) => entry.asset === path)).toHaveLength(1);
    expect(rows.find((entry: { asset: string }) => entry.asset === path)).toMatchObject({
      author: 'Harrison Lee et al.',
      source_url: 'https://arxiv.org/html/2309.00267v2#S1.F2',
      source_image_url: 'https://arxiv.org/html/2309.00267v2/rlaif.png',
      source_asset: path, source_asset_sha256: digest, sha256: digest,
      license: 'CC BY 4.0', license_evidence: licensedPage,
      license_evidence_sha256: hash(read(licensedPage)),
      derivation: 'source-png-copy', dimensions_px: [959, 470], used_in: [chapter],
    });
    expect(read(licensedPage).toString()).toContain('creativecommons.org/licenses/by/4.0/');
    expect(read(licensedPage).toString()).toContain('2309.00267v2');
    expect(hash(bytes)).toBe(digest);
    expect([bytes.readUInt32BE(16), bytes.readUInt32BE(20)]).toEqual([959, 470]);
    expect(read(chapter).toString()).toContain(path);
  });

  it('restores only the reviewed additions and preserves the earlier chapter and figure', () => {
    const j = JSON.parse(read('docs/audit/2026-10-29-constitutional-ai-additions.json').toString());
    expect(j.files).toHaveLength(2);
    for(const row of j.files) {
      const live = read(row.source).toString();
      const text = beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-constitutional-ai-additions.json');
      expect(hash(text)).toBe(row.after_sha256);
      let restored = text;
      for(const c of [...row.changes].reverse()) {
        expect(restored.split(c.new)).toHaveLength(2);
        restored = restored.replace(c.new,()=>c.old);
      }
      expect(hash(restored)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-structured-dropout-additions.json')).toBe(restored);
      expect(()=>beforeTopicConsolidation(live+'\nUnreviewed edit.',row.source,root,
        '2026-10-29-structured-dropout-additions.json')).toThrow();
    }
    expect(read(chapter).toString()).toContain('constitutional-ai-overview.png');
    const raw = read('Concepts/Training/Constitutional AI.md');
    expect(hash(raw)).toBe('3b35e33f34fee062c071568abdebd21f157be268a83f7d1c169854d14b293b5c');
  });
});
