import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const report = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-10-29-en-multimodal-parity.json'), 'utf8'));
const sha = (path: string) => createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex');

describe('English multimodal substantive parity evidence', () => {
  it('pins seven actual English/canonical revisions, not metadata-only translations', () => {
    expect(report.entries).toHaveLength(7);
    for (const entry of report.entries) {
      const text = readFileSync(resolve(root, entry.path), 'utf8');
      expect(sha(entry.path)).toBe(entry.after_sha256);
      expect(sha(entry.canonical)).toBe(entry.canonical_sha256);
      expect(text).toContain(`canonical_revision: "${entry.canonical_sha256}"`);
      expect(text).toContain('translation_status: current');
      expect(entry.en_full_body_read && entry.canonical_full_body_read).toBe(true);
      expect(entry.substantive_topics_reconciled.length).toBeGreaterThanOrEqual(7);
      expect(entry.outstanding_substantive_omissions).toEqual([]);
    }
  });

  it('records real execution of all 75 canonical examples with unchanged Python ASTs', () => {
    expect(report.verification.published_python_executed).toBe(75);
    expect(report.entries.map((e: any) => e.python_blocks)).toEqual([6, 6, 9, 8, 12, 17, 17]);
    for (const entry of report.entries) {
      expect(entry.canonical_code_ast_multiset_equal).toBe(true);
      expect(entry.executions).toHaveLength(entry.canonical_python_blocks);
      expect(entry.executions.every((e: any) => e.status === 'passed' && e.returncode === 0)).toBe(true);
    }
    expect(report.verification.gpu_or_trained_model_inference).toContain('not run');
  });

  it('retains all 50 source figures and explicitly leaves rendered acceptance to parent QA', () => {
    expect(report.totals.before_figures).toBe(29);
    expect(report.totals.after_figures).toBe(50);
    for (const entry of report.entries) {
      expect(entry.figures).toHaveLength(entry.figure_count);
      for (const figure of entry.figures) expect(existsSync(resolve(root, figure))).toBe(true);
    }
    expect(report.verification.build_and_visual_qa).toContain('not claimed');
  });

  it('preserves explicit source fragments and replaces obsolete ordinal link labels', () => {
    for (const entry of report.entries) {
      const text = readFileSync(resolve(root, entry.path), 'utf8');
      for (const id of entry.canonical_explicit_aliases) expect(text).toContain(`<span id="${id}"></span>`);
      expect(new Set(entry.retained_aliases).size).toBe(entry.retained_aliases.length);
      expect(text).not.toMatch(/\|64(?:\.\d+)?\]\]/);
      expect(text).not.toContain('<!-- PARITY EXAMPLE');
      const outsideCode = text.replace(/(```|~~~).*?\n\1/gs, '');
      expect(outsideCode).not.toMatch(/^\s*\$\s*$/m);
    }
  });

  it('keeps technical distinctions repaired during full-body reconciliation', () => {
    const read = (start: string) => readFileSync(resolve(root,
      report.entries.find((e: any) => e.path.split('/').at(-1).startsWith(start)).path), 'utf8');
    expect(read('64a ')).toContain('Both gates start at exactly zero');
    expect(read('64b ')).toContain('geometric coordinates do not replace cache length');
    expect(read('64c ')).toContain('names SAPO in this report version, not GRPO');
    expect(read('64d ')).toContain('not literal transcription or a CER measurement');
    expect(read('64e ')).toContain('Acoustic RVQ does not continue the semantic branch');
    expect(read('64f ')).toContain('client TTFT 500 ms');
    expect(read('64f ')).toContain('does not itself fix UUID collisions');
  });
});
