import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { inspectEnInferenceParity } from '../tools/en-inference-parity.mjs';

const root = resolve(import.meta.dirname, '../..');

describe('complete English inference chapter reconciliation', () => {
  it('keeps pinned receipt revisions and checks current section, source and fragment parity', () => {
    const { totals } = inspectEnInferenceParity(root);
    expect(totals.chapters).toBe(7);
    expect(totals.figures).toBe(44);
    expect(totals.displayMath).toBe(71);
    expect(totals.codeFences).toBe(43);
    expect(totals.originalExcerpts).toBe(20);
    expect(totals.canonicalSections).toBe(181);
  });

  it('reports the current decoding metadata separately from the unchanged historical receipt', () => {
    const { revisions } = inspectEnInferenceParity(root);
    const row = revisions.find((r: any) => r.source.endsWith('54 Декодирование и выбор следующего токена.md'));
    expect(row.receiptCanonicalSha256).toBe('cb53484debaf97e673ff30e804afc4d9941b94e6d87eebc8cae78bdda05685a0');
    expect(row.canonicalSha256).toBe(createHash('sha256').update(readFileSync(resolve(root, row.source))).digest('hex'));
    expect(row.englishSha256).toBe(createHash('sha256').update(readFileSync(resolve(root, row.path))).digest('hex'));
    expect(row.canonicalSha256).not.toBe(row.receiptCanonicalSha256);
    expect(row.englishSha256).not.toBe(row.receiptEnglishSha256);
  });

  it('separates executed CPU examples from runtime/GPU snippets and browser acceptance', () => {
    const { report } = inspectEnInferenceParity(root);
    const receipt = JSON.parse(readFileSync(resolve(root,
      'docs/audit/2026-10-29-en-inference-examples.json'), 'utf8'));
    expect(receipt.status).toBe('pass');
    expect(receipt.executed_python_blocks).toBe(33);
    expect(receipt.checker_sha256).toBe(createHash('sha256').update(readFileSync(resolve(root,
      'publishing/tools/check_en_inference_examples.py'))).digest('hex'));
    for (const chapter of report.chapters) {
      expect(receipt.chapters.find((entry: any) => entry.path === chapter.path)?.sha256)
        .toBe(chapter.after_sha256);
    }
    expect(report.verification.executed_python_blocks).toBe(33);
    expect(report.verification.independent_checks.attention_output_lse_gradient_cases).toBe(36);
    expect(report.verification.independent_checks.int4_pairs).toBe(256);
    expect(report.verification.independent_checks.residual_distributions).toBe(102);
    expect(report.verification.not_executed).toHaveLength(2);
    expect(report.verification.build_visual).toContain('Not claimed');
    for (const chapter of report.chapters) {
      expect(chapter.source_read_lines).toEqual([1, chapter.source_lines]);
      expect(chapter.substantive_parity).toBe(true);
      expect(chapter.outstanding_substantive_omissions).toEqual([]);
    }
  });
});
