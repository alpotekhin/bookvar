import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-predictive-pipeline-additions.json'));
const row = journal.files[0];
const current = read(row.source);
const prior = beforeTopicConsolidation(current, row.source, root,
  '2026-10-31-generalization-caption-additions.json');

describe('source-first predictive pipeline additions', () => {
  it('restores the exact preceding revision and rejects unrelated drift', () => {
    expect(row.before_sha256).toBe('9251873332d9ad4e751c5df2b989794bde0b638f1225f481a2b0ad51fe866075');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
      '2026-10-31-generalization-caption-additions.json')).toThrow();
    const registryRow = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registryRow.source), registryRow.source, root,
      '2026-10-31-generalization-caption-additions.json'))).toBe(registryRow.before_sha256);
  });

  it('keeps both preceding programs and all seven figures, adding worked cases rather than stubs', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(5);
    expect(fences(current).slice(-2)).toEqual(fences(prior));
    expect(images(current)).toHaveLength(11);
    expect(images(current).filter(p => images(prior).includes(p))).toEqual(images(prior));
    for (const detail of ['unknown_value=-1', 'stump_scores', '[0.75, 1.0]',
      'no_rare_bucket', 'infrequent_categories_', 'first 5,000 rows',
      'correlated columns need not be interchangeable']) expect(current).toContain(detail);
  });

  it('publishes unchanged course figures with pinned or dated original provenance', () => {
    const folder = '00 Учебник/Assets/Figures/curated/classical-ml-pipeline/';
    for (const [name, digest] of [
      ['mooc-transformer-fit.svg', '68efcc9704fbd4e82c96727580a85a98a91178fd9362394470c8df342aa07e40'],
      ['mooc-transformer-fit-transform.svg', 'ac0959c33fed033f7c6c18e6a8377cd5e28efdf2304aa176bc69752f4df64f7c'],
      ['mooc-adult-histograms.png', '176e045cf23d449a1baa4b836ca4a01a487f1f696f2b045641d4dd0d63ac212a'],
      ['mooc-adult-pairplot.png', 'eb1a59dfdc6ff56ba9d9cbda4a64df32b0991a6bccaee122e4bee7d1892c618d'],
    ]) expect(sha(readFileSync(resolve(root, folder + name))), name).toBe(digest);
    expect(current).toContain('Original excerpt: Inria, scikit-learn MOOC');
    expect(current).toContain('not a Bookvar experiment');
    expect(journal.sourceComparison.full_original_lessons_read).toHaveLength(6);
  });
});
