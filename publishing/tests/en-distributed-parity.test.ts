import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const report = JSON.parse(read('docs/audit/2026-10-29-en-distributed-parity.json'));
const atParityRevision = (path: string) => beforeTopicConsolidation(
  read(path), path, root, '2026-11-11-gpt-continuation-scoring-additions.json',
);

describe('English distributed-serving full substantive parity', () => {
  it('pins both completed EN/canonical revisions with full-body reconciliation', () => {
    expect(report.entries).toHaveLength(2);
    for (const entry of report.entries) {
      expect(hash(atParityRevision(entry.path))).toBe(entry.after_sha256);
      expect(hash(atParityRevision(entry.canonical))).toBe(entry.canonical_sha256);
      expect(atParityRevision(entry.path)).toContain('canonical_revision: "' + entry.canonical_sha256 + '"');
      expect(read(entry.path)).toContain('canonical_revision: "' + hash(read(entry.canonical)) + '"');
      expect(read(entry.path)).toContain('translation_status: current');
      expect(entry.en_full_body_read && entry.canonical_full_body_read).toBe(true);
      expect(entry.substantive_topics_reconciled.length).toBeGreaterThanOrEqual(12);
      expect(entry.outstanding_substantive_omissions).toEqual([]);
    }
  });

  it('records execution of all 18 canonical examples with equal ASTs, without benchmark claims', () => {
    expect(report.verification.published_python_executed).toBe(18);
    expect(report.entries.map((e: any) => e.python_blocks)).toEqual([8, 10]);
    for (const entry of report.entries) {
      expect(entry.canonical_code_ast_sequence_equal).toBe(true);
      expect(entry.executions).toHaveLength(entry.python_blocks);
      expect(entry.executions.every((e: any) => e.status === 'passed')).toBe(true);
      expect(entry.execution_mode).toContain('sharing a namespace');
    }
    expect(report.verification.gpu_or_model_inference).toContain('not run');
  });

  it('retains 17 canonical figures and the existing additional Dynamo diagram', () => {
    expect(report.totals.before_figures).toBe(9);
    expect(report.totals.after_figures).toBe(18);
    expect(report.entries.map((e: any) => e.canonical_figure_count)).toEqual([11, 6]);
    for (const entry of report.entries) {
      for (const figure of entry.figures) expect(existsSync(resolve(root, figure))).toBe(true);
    }
    expect(report.verification.build_and_visual_qa).toContain('not claimed');
  });

  it('preserves all 44 canonical heading fragments and the original licensed English excerpt', () => {
    expect(report.verification.canonical_heading_aliases).toBe(44);
    for (const entry of report.entries) {
      const text = read(entry.path);
      for (const heading of entry.heading_map) {
        expect(text).toContain('id="' + heading.canonical_fragment + '"');
      }
      expect(new Set(entry.retained_aliases).size).toBe(entry.retained_aliases.length);
      expect(text).not.toMatch(/^\s*\$\s*$/m);
      expect(text).not.toContain('<!-- EXAMPLE:');
    }
    expect(report.entries[1].original_english_excerpts).toEqual(['splitwise-layer-handoff']);
    expect(report.verification.original_english_excerpt_retained).toBe(true);
  });

  it('bounds the canonical reservation repair to the exact single-line arithmetic correction', () => {
    const fix = report.canonical_correction;
    const canonical = atParityRevision(fix.path);
    expect(hash(canonical)).toBe(fix.after_sha256);
    expect(hash(canonical.replace(fix.new, fix.old))).toBe(fix.before_sha256);
    expect(canonical).toContain('fits = used + size <= capacity');
    expect(read(report.entries[1].path)).toContain('a new 8192-position cache does not fit');
    expect(read(report.entries[1].path)).toContain('4096-position cache would fit exactly');
  });

  it('retains the important causal distinctions in both chapters', () => {
    const parallel = read(report.entries[0].path);
    const pd = read(report.entries[1].path);
    expect(parallel).toContain('No full $H=[H_0,H_1]$ was gathered');
    expect(parallel).toContain('simple average, 27.083, is wrong');
    expect(parallel).toContain('EP can combine with TP within each expert');
    expect(parallel).toContain('not a universal cluster-size');
    expect(pd).toContain('its K/V have not yet been');
    expect(pd).toContain('it does not stop a write already in flight');
    expect(pd).toContain('accurate per-request output-length prediction is left for future work');
    expect(pd).toContain('13.3 qualifying requests/s, but only 0.83125 per GPU');
  });

  it('publishes both complete licensed DistServe figures in each edition', () => {
    const paths = [report.entries[1].canonical, report.entries[1].path];
    const registry = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for (const [fig, page, name, width, height] of [
      [2, 4, 'interference', 1630, 742],
      [6, 8, 'runtime', 1504, 804],
    ] as const) {
      const asset = '00 Учебник/Assets/Figures/curated/distserve-v3-2026-11-11/'
        + 'distserve-v3-figure' + fig + '-' + name + '.png';
      const row = registry.find((entry: { asset: string }) => entry.asset === asset);
      const file = readFileSync(resolve(root, asset));
      const figure = 'https://arxiv.org/pdf/2401.09670v3#page=' + page;
      expect(row).toMatchObject({
        sha256: createHash('sha256').update(file).digest('hex'),
        dimensions_px: [width, height], source_page: page, license: 'CC BY-SA 4.0',
        figure_url: figure, used_in: paths,
      });
      expect([file.readUInt32BE(16), file.readUInt32BE(20)]).toEqual([width, height]);
      const replaced = registry.find((entry: { replacement?: string }) => entry.replacement === asset);
      expect(replaced).toMatchObject({ publication_status: 'retired', used_in: [] });
      for (const path of paths) {
        const chapter = read(path);
        expect(chapter).toContain(asset + '|');
        expect(chapter).toContain(figure);
        expect(chapter).not.toContain(replaced.asset);
      }
    }
  });

  it('binds the original Mooncake and Dynamo assets and notices to both editions', () => {
    const review = JSON.parse(read('docs/audit/2026-11-11-mooncake-dynamo-originals-additions.json'));
    const registry = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const paths = [report.entries[1].canonical, report.entries[1].path];
    expect(review.figures).toHaveLength(2);
    for (const figure of review.figures) {
      const row = registry.find((entry: { asset: string }) => entry.asset === figure.asset);
      const replaced = registry.find((entry: { replacement?: string }) => entry.replacement === figure.asset);
      const bytes = readFileSync(resolve(root, figure.asset));
      const gitBlob = createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');
      const license = readFileSync(resolve(root, figure.license_file));
      expect(gitBlob).toBe(figure.git_blob);
      expect(createHash('sha256').update(license).digest('hex')).toBe(figure.license_sha256);
      expect(license.toString()).toContain('TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION');
      expect(row).toMatchObject({
        commit: figure.git_commit, git_blob: figure.git_blob,
        sha256: createHash('sha256').update(bytes).digest('hex'),
        license: 'Apache-2.0', license_evidence: figure.license_file,
        license_evidence_sha256: figure.license_sha256, used_in: paths,
      });
      expect(replaced).toMatchObject({ publication_status: 'retired', used_in: [] });
      for (const path of paths) {
        const chapter = read(path);
        expect(chapter).toContain(figure.asset + '|');
        expect(chapter).toContain(figure.license_file.replace('site/public', ''));
        expect(chapter).not.toContain(replaced.asset);
      }
    }
  });
});

