import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeMathReflow } from '../tools/math-reflow-baseline.mjs';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const report = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-10-29-math-reflow.json'), 'utf8'));
const hash = (text: string) => createHash('sha256').update(text).digest('hex');

function mathTokens(text: string) {
  // Matrix row/column boundaries carry mathematical meaning and are retained.
  const matrices: string[] = [];
  text = text.replace(/\\begin\{pmatrix\}[\s\S]*?\\end\{pmatrix\}/g, matrix => {
    matrices.push(matrix.replace(/\s/g, ''));
    return `MATRIX${matrices.length - 1}`;
  });
  return { matrices, tokens: text
    .replace(/\\(?:begin|end)\{(?:aligned|gathered)\}/g, '')
    .replace(/\\(?:qquad|quad|left|right|bigl|bigr)/g, '')
    .replace(/\\\\/g, '').replace(/[&\s]/g, '').replace(/\{\}/g, '') };
}

describe('recorded textbook display-math reflow', () => {
  it('preserves every mathematical token and every matrix boundary', () => {
    for (const file of report.files) for (const change of file.changes) {
      expect(mathTokens(change.new)).toEqual(mathTokens(change.old));
    }
  });
  it('restores the exact preceding chapter, including prose, source text and figures', () => {
    for (const file of report.files) {
      const text = beforeTopicConsolidation(readFileSync(resolve(root, file.source), 'utf8'), file.source, root);
      expect(hash(text)).toBe(file.after);
      expect(hash(beforeMathReflow(text, file.source))).toBe(file.before);
    }
  });
  it('does not accept an unrecorded formula or erase an unrelated prose edit', () => {
    const file = report.files[0];
    const text = readFileSync(resolve(root, file.source), 'utf8');
    expect(() => beforeMathReflow(text.replace(file.changes[0].new, 'x=0'), file.source))
      .toThrow('Unrecognized math revision');
    expect(hash(beforeMathReflow(text + '\nExtra prose.\n', file.source)))
      .not.toBe(file.before);
  });
});
