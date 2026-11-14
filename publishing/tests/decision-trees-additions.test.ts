import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-decision-trees-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-10-31-linear-models-additions.json';
const prior = beforeTopicConsolidation(current, row.source, root, stop);

describe('decision-tree source explanations and worked additions', () => {
  it('restores the exact preceding chapter and registry, rejecting unreviewed changes', () => {
    expect(row.before_sha256).toBe('cd7ead3663bb14c0c95cf325d5c2ddb90adb703c4106557c7f62134f406c1d83');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root, stop)).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root, stop)))
      .toBe(registry.before_sha256);
  });

  it('retains all original programs and figures while adding complete worked mechanisms', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(5);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(17);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const detail of ['linear_small', 'asym_models', '600 points', '147, 159 and 3',
      'not to estimate test error', 'need not equal', 'max_depth']) expect(current).toContain(detail);
    expect(current).not.toContain('  - https://inria.github.io/scikit-learn-mooc/trees/\n');
    expect(current).toContain('| Порог $t$ | Джини после разбиения | Выигрыш |');
  });

  it('retains unchanged source bytes and the original English course explanation', () => {
    for (const [name, digest] of [
      ['mooc-asymmetric-depth2.png', 'fef3fdf4246c7ffa583c505c020ea7fc24129670cf5df33611b57ee7ea0e6341'],
      ['mooc-asymmetric-depth6.png', '453f478a06c49250cc473ac01db185793d5c3b9da6f6c6a687203fee800ead56'],
      ['mooc-asymmetric-tree6.png', 'e3944fb791940dcf053baefeea2457a113f7cab8c22af60627ae6c4299bf88c6'],
      ['mooc-tree-linear-extrapolation.png', '1cc48186f9f529ac55f97dc493083415311c0e11525765ef09ae59c9b263ed86'],
    ]) expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/classical-ml-trees/' + name))), name).toBe(digest);
    const source = read('05 Источники/Courses/Scikit-learn MOOC/python_scripts/trees_hyperparameters.py.md');
    for (const excerpt of ['We build a dataset where we illustrate this asymmetry.',
      'As expected, the left branch of the tree continue to grow']) {
      expect(current).toContain(excerpt);
      expect(source).toContain(excerpt);
    }
    expect(journal.sourceComparison.fullyRead).toHaveLength(6);
    expect(current).toContain('CC BY 4.0');
  });
});
