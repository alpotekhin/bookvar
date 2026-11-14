import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-31-clustering-course-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-10-31-evaluation-target-additions.json';
const prior = beforeTopicConsolidation(current, row.source, root, stop);

describe('clustering original-course examples and visual additions', () => {
  it('restores the preceding chapter and registry and rejects unrelated changes', () => {
    expect(row.before_sha256).toBe('4a09c7bd9dab975e556ee7eb44ae18a84adbf45061f274e6aedd6bac7ad77455');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root, stop)).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root, stop)))
      .toBe(registry.before_sha256);
  });

  it('retains all earlier programs/figures and the complete existing mechanisms', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(4);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(17);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const text of ['purchase_counts', 'count_positions', '0.375',
      'encoder inside each training partition', 'different transformed spaces',
      'Нормировка только наблюдений', 'Взаимная достижимость'])
      expect(current).toContain(text);
  });

  it('keeps original images and exact source-language excerpts', () => {
    for (const [name, digest] of [
      ['mooc-news-external-metrics.png', '712206c5a2610cac4dc2010e563719e91af6d217784fea59941e2a53a4f16f1b'],
      ['mooc-news-silhouette.png', '5604f75c9275f74e35b7679a4980af1f46372513ede60f273859d48cddcbca84'],
      ['mooc-rfm-quantile-layers.png', 'cfa0c2204e8cb75cd4039771c841ab49d814bc10bd48fc555ca77eeefad574ef'],
    ]) expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/classical-ml-clustering/' + name))), name).toBe(digest);
    const news = read('05 Источники/Courses/Scikit-learn MOOC/python_scripts/clustering_supervised_metrics.py.md');
    const rfm = read('05 Источники/Courses/Scikit-learn MOOC/python_scripts/clustering_sol_01.py.md')
      .replace(/^# ?/gm, '');
    for (const [start, end, source] of [
      ['However, categorizing', 'silhouette score.', news],
      ['We can observe that the quantile-transformed', 'overrepresented in the dataset.', rfm],
    ]) {
      const first = current.indexOf('> ' + start);
      const last = current.indexOf(end, first) + end.length;
      expect(first).toBeGreaterThan(0);
      expect(source).toContain(current.slice(first, last).replace(/^> /gm, ''));
    }
    expect(news).toContain('data_encoded = StringEncoder(random_state=0).fit_transform');
    expect(journal.sourceComparison.fullyRead).toHaveLength(6);
  });
});
