import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-evaluation-target-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-10-31-ensembles-additions.json';
const prior = beforeTopicConsolidation(current, row.source, root, stop);

describe('evaluation target-transform source addition', () => {
  it('restores the exact preceding chapter and registry without accepting arbitrary edits', () => {
    expect(row.before_sha256).toBe('c1c725e2074ccde8ecda865accdddf2c94232df190c66ed327e9148406c01a77');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root, stop)).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root, stop)))
      .toBe(registry.before_sha256);
  });

  it('keeps both earlier programs, all fifteen embeds and the detailed evaluation mechanisms', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(3);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(16);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const value of ['raw_constant', 'log_constant', 'geometric mean',
      'training fold', '20 at 3', '27', 'порог-выбирается-по-out-of-fold-предсказаниям'])
      expect(current).toContain(value);
  });

  it('uses the unchanged original course figure and attributed source-language passage', () => {
    expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/classical-ml-evaluation/regression-transformed-target.png'))))
      .toBe('42c821c57965432baa2d5f05d48516ebf94f928f0fabdc42497b7dc019373d9d');
    const original = read('05 Источники/Courses/Scikit-learn MOOC/python_scripts/metrics_regression.py.md');
    const quote = current.match(/^> On these plots,[\s\S]*?distribution\.\n/m)![0]
      .replace(/^> /gm, '');
    expect(original).toContain(quote);
    expect(current).toContain('CC BY 4.0. Original English excerpt; not translated.');
    expect(journal.sourceComparison.fullyRead).toHaveLength(6);
  });
});
