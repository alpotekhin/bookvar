import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-linear-models-additions.json'));
const row = journal.files[0], current = read(row.source);
const prior = beforeTopicConsolidation(current, row.source, root,
  '2026-10-31-predictive-pipeline-additions.json');

describe('linear-model coordinate and feature-map additions', () => {
  it('restores the exact prior explanation and registry, rejecting unrelated edits', () => {
    expect(row.before_sha256).toBe('75f92f4704376d7989fcc9be6ac71d53566a4aa8032851798960430a63f4914a');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
      '2026-10-31-predictive-pipeline-additions.json')).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root,
      '2026-10-31-predictive-pipeline-additions.json'))).toBe(registry.before_sha256);
  });

  it('preserves all preceding programs and figures while adding worked mechanisms', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(5);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(17);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const detail of ['raw_intercept', 'standardized_point', 'bin_encoded', 'joint_model',
      'equal sums on the two', 'training fit', 'not fitted\npenguin coefficients'])
      expect(current).toContain(detail);
  });

  it('retains the complete original course visual bytes and licenses', () => {
    for (const [file, digest] of [
      ['mooc-logistic-boundary.svg', 'c30a7a2dee6fe82f57765c0a5ca421b17ebfce568df14ecf2a7b824878ea844b'],
      ['mooc-binning-classification.png', '2eec917c32e06325cdf094c18714b65de6df9be7405854752883b96710202f75'],
      ['mooc-splines-classification.png', 'cd2a5d8abf0a51973c9bbcbca4bf9f6e5b5e3a09a755122bec4a311b2a450921'],
      ['mooc-polynomial-classification.png', '153c5a32098739fd78e0d1ce8fd70a1b44c94e6644ae73ec6828c21a5d2359a7'],
    ]) expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/classical-ml-linear/' + file))), file).toBe(digest);
    expect(journal.sourceComparison.fullyRead).toHaveLength(6);
    expect(current).toContain('Original MOOC explanation');
    expect(current).toContain('CC BY 4.0');
  });
});
