import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-generalization-caption-additions.json'));
const row = journal.files[0];
const current = read(row.source);
const captionRevision = beforeTopicConsolidation(current, row.source, root,
  '2026-10-31-generalization-caption-additions.json');
const prior = beforeTopicConsolidation(current, row.source, root,
  '2026-10-30-loss-weighted-reduction-additions.json');

describe('generalization source-figure caption', () => {
  it('restores the exact preceding explanation and rejects unrelated drift', () => {
    expect(row.before_sha256).toBe('e1efdca476a16e47475fc763bc9a64c5524fdca74add685ba9d5dc8dd335e7c2');
    expect(sha(captionRevision)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(row.changes).toHaveLength(2);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
      '2026-10-30-loss-weighted-reduction-additions.json')).toThrow();
  });

  it('keeps the three programs and nine unchanged originals beside the new experiment', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(3);
    expect(fences(current)).toEqual(fences(prior));
    expect(images(captionRevision)).toEqual(images(prior));
    expect(images(current)).toHaveLength(10);
    expect(images(current).filter(image => !image.includes('sklearn-tree-bagging-bias-variance.png')))
      .toEqual(images(captionRevision).filter(image => !image.includes('cs229-bias-variance-decomposition.png')));
    expect(images(current).some(image => image.includes('cs229-bias-variance-decomposition.png'))).toBe(false);
    expect(current).toContain('https://cs229.stanford.edu/summer2023/BiasVarianceAnalysis.pdf#page=7');
    expect(current).toContain('В этой иллюстрации простая граница допускает меньше ошибок');
    expect(current).not.toContain('На одном и том же тестовом наборе сложная граница');
  });

  it('does not treat the original Google diagrams as an identical-coordinate dataset', () => {
    const points = (name: string) => {
      const svg = read('00 Учебник/Assets/Figures/curated/foundations-source-first/google-' + name + '-model.svg');
      const box = svg.match(/<path fill="#(?:0000ff|ff9900)" d="m([\d.-]+) ([\d.-]+)l576\.0 0l0 576\.0/);
      if (!box) throw new Error('Source forest rectangle changed');
      return new Set([...svg.matchAll(/<path fill="#00ffff" d="m([\d.-]+) ([\d.-]+)/g)]
        .map(m => [+(+m[1] - +box[1]).toFixed(3), +(+m[2] - +box[2]).toFixed(3)])
        .filter(([x, y]) => x >= 0 && y >= 0 && x <= 576 && y <= 576)
        .map(p => p.join(',')));
    };
    const complex = points('complex'), simple = points('simple');
    expect(complex.size).toBe(49);
    expect(simple.size).toBe(49);
    expect([...complex].filter(p => simple.has(p))).toHaveLength(48);
  });
});
