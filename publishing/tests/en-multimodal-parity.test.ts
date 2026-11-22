import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const report = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-10-29-en-multimodal-parity.json'), 'utf8'));
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const parityRevision = (path: string) => beforeTopicConsolidation(read(path), path,
  root, '2026-11-09-responsible-evaluation-additions.json');

describe('English multimodal substantive parity evidence', () => {
  it('pins seven actual English/canonical revisions, not metadata-only translations', () => {
    expect(report.entries).toHaveLength(7);
    for (const entry of report.entries) {
      const text = parityRevision(entry.path);
      expect(sha(text)).toBe(entry.after_sha256);
      expect(sha(parityRevision(entry.canonical))).toBe(entry.canonical_sha256);
      expect(text).toContain(`canonical_revision: "${entry.canonical_sha256}"`);
      expect(text).toContain('translation_status: current');
      expect(entry.en_full_body_read && entry.canonical_full_body_read).toBe(true);
      expect(entry.substantive_topics_reconciled.length).toBeGreaterThanOrEqual(7);
      expect(entry.outstanding_substantive_omissions).toEqual([]);
    }
  });

  it('keeps the current vision objective and original figure paths aligned in both editions', () => {
    const entry = report.entries.find((row: { path: string }) => row.path.endsWith('/64 Multimodal models.md'));
    const en = read(entry.path);
    const canonical = read(entry.canonical);
    expect(en).toContain(`canonical_revision: "${sha(canonical)}"`);
    const example = (text: string) => text.match(/<a id="vlm-multi-positive"><\/a>\n\n### [^\n]+\n([\s\S]*?)(?=\n### )/)![1];
    const math = (text: string) => example(text).match(/\$\$[\s\S]*?\$\$/)![0];
    expect(math(en)).toBe(math(canonical));
    const figures = (text: string) => [...text.matchAll(/!\[\[([^|\]]+\/vision-objectives-2026-11-09\/[^|\]]+)\|/g)]
      .map(row => row[1]);
    expect(figures(en)).toHaveLength(2);
    expect(figures(en)).toEqual(figures(canonical));
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

  it('shares the document table-score derivation and originals in both editions', () => {
    const en = read('en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding.md');
    const ru = read('00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md');
    expect(en).toContain(`canonical_revision: "${sha(ru)}"`);
    const segment = (text: string) => text.split('<a id="doc-grits"></a>')[1]
      .split('<a id="doc-interventions"></a>')[0];
    const derivation = (text: string) => segment(text).match(/\$\$[\s\S]*?\$\$/)?.[0];
    const figures = (text: string) => [...segment(text).matchAll(/grits-figure[^|]+\.png/g)]
      .map(match => match[0]);
    expect(derivation(ru)).toContain('=0.75');
    expect(derivation(en)).toBe(derivation(ru));
    expect(figures(ru)).toEqual(['grits-figure1-alignment.png', 'grits-figure2-representations.png']);
    expect(figures(en)).toEqual(figures(ru));
  });

  it('keeps the AuT and playback additions aligned in both editions', () => {
    const en = read('en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md');
    const ru = read('00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md');
    expect(en).toContain(`canonical_revision: "${sha(ru)}"`);
    for (const text of [ru, en]) {
      expect(text).toContain('qwen3omni-figure3-aut.png');
      expect(text).toContain('72+88+57+14+3=234');
      expect(text).toContain('160+160+210+14+3=547');
      expect(text).toContain('uninterrupted_start == 130');
    }
    const playback = (text: string) => text.match(/def playback\([\s\S]*?450\n\)/)?.[0];
    expect(playback(ru)).toBeDefined();
    expect(playback(en)).toBe(playback(ru));
  });

  it('pins the reviewed resolution edition and the document invoice regions', () => {
    const resolutionRu = read('00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md');
    const resolutionEn = read('en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions.md');
    expect(resolutionEn).toContain(`canonical_revision: "${sha(resolutionRu)}"`);
    const ru = read('00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md');
    const en = read('en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding.md');
    const segment = (text: string) => text.split('<a id="doc-paddleocr-reading-order"></a>')[1];
    for (const text of [en, ru]) {
      expect(segment(text)).toContain('pp-doclayout-v2-reading-order.png');
      expect(segment(text)).toContain('$(3,4,0,2,1)$');
      expect(segment(text)).toContain('$3,5,4,1,2$');
      expect(text).toContain('[700, 1140, 910, 1180]');
    }
    expect(en).not.toContain('[0.68, 0.82, 0.91, 0.87]');
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
    const documents = read('64d ');
    expect(documents).toContain('https://arxiv.org/pdf/2409.12191v1#page=29');
    expect(documents).toMatch(/returns an English\s+translation/);
    expect(documents).toMatch(/CER needs a transcription and a reference in the original\s+language/);
    expect(read('64e ')).toContain('Acoustic RVQ does not continue the semantic branch');
    expect(read('64f ')).toContain('client TTFT 500 ms');
    expect(read('64f ')).toContain('does not itself fix UUID collisions');
  });
});
