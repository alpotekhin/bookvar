import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-ensembles-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-10-31-decision-trees-additions.json';
const prior = beforeTopicConsolidation(current, row.source, root, stop);

describe('ensemble pipeline, histogram and tree-growth additions', () => {
  it('restores the exact preceding chapter and registry, rejecting unrelated edits', () => {
    expect(row.before_sha256).toBe('4ed341f212c55bf58c407433d9f12459ae5bfdef3c3aba7a1d784b033fd87ebb');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root, stop)).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root, stop)))
      .toBe(registry.before_sha256);
  });

  it('keeps all preceding programs and figures and adds worked mechanisms', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(5);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(17);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const detail of ['poly_bag', 'poly_predictions', 'hist_G', 'hist_gains',
      'histogram subtraction', 'depth constraint', 'not a calibrated prediction interval'])
      expect(current).toContain(detail);
    expect(current).not.toContain('  - https://inria.github.io/scikit-learn-mooc/ensemble/\n');
    expect(current).toContain('Псевдо<wbr>остатки $-g_i$');
  });

  it('retains original source figures, source-language excerpts and complete licenses', () => {
    for (const [name, digest] of [
      ['mooc-bagging-polynomial.png', 'f4eb32e1b4596fb3340b3035bd3d163a0d5bd63f807de5c5e3e675294dd1f887'],
      ['xgboost-split-scan.png', 'c4651c603839288bf172b89d9b9a0bc02a1b6e19a05d04bda484a86cf430ef9b'],
      ['lightgbm-level-wise.png', '53f0cf05bdfad338037db47b3213a369e03a0aa2e8b3c2c7888a3caf6c832498'],
      ['lightgbm-leaf-wise.png', 'dbcc495e5e4b8dcc2f837e58fb72a339c09b0703fdfd19850a20ce981f921c59'],
    ]) expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/classical-ml-ensembles/' + name))), name).toBe(digest);
    for (const [name, digest] of [
      ['dmlc-web-data-934cb2a0-Apache-2.0.txt', 'c6596eb7be8581c18be736c846fb9173b69eccf6ef94c5135893ec56bd92ba08'],
      ['lightgbm-39ae188a-MIT.txt', 'ca70f15d841f7e76c8bbd284d25300cd4ae2ef1035e7e5252b13b73449464305'],
    ]) expect(sha(readFileSync(resolve(root, 'site/public/licenses/' + name))), name).toBe(digest);
    const original = read('05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_bagging.py.md');
    for (const excerpt of ['Scikit-learn implements the bagging procedure as a **meta-estimator**',
      'The ensemble itself is simply built by passing the resulting pipeline']) {
      expect(current).toContain(excerpt);
      expect(original).toContain(excerpt);
    }
    expect(journal.sourceComparison.fullyRead).toHaveLength(8);
  });
});
