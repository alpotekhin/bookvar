import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const journalName = '2026-10-30-tensor-vector-operands-additions.json';
const journal = JSON.parse(readFileSync(resolve(root, 'docs/audit', journalName), 'utf8'));
const row = journal.files[0];
const current = readFileSync(resolve(root, row.source), 'utf8');
const text = beforeTopicConsolidation(current, row.source, root, journalName);
const sha = (value: string) => createHash('sha256').update(value).digest('hex');
const prior = beforeTopicConsolidation(current, row.source, root, '2026-10-30-clean-checkout-additions.json');

describe('one-dimensional NumPy matmul explanation', () => {
  it('restores the exact whole chapter read before this bounded addition', () => {
    expect(sha(text)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
      '2026-10-30-clean-checkout-additions.json')).toThrow();
  });

  it('retains all seven original illustrations and the existing two executable blocks', () => {
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(images(text)).toEqual(images(prior));
    expect(images(text)).toHaveLength(7);
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    expect(fences(text)).toHaveLength(2);
    expect(fences(text)[0]).toBe(fences(prior)[0]);
    expect(fences(text)[1]).toContain('assert (x @ v) == 1. and np.ndim(x @ v) == 0');
  });

  it('connects temporary axes, numerical results and the primary NumPy rule', () => {
    for (const detail of ['(2,3) @ (3,1) -> (2,1)', '(1,3) @ (3,2) -> (1,2)',
      '(1,3) @ (3,1) -> (1,1)', 'x @ W = [5,-2]',
      'https://numpy.org/doc/2.3/reference/generated/numpy.matmul.html#notes']) {
      expect(text).toContain(detail);
    }
  });

  it('explains norm scaling and the triangle inequality on the existing 3-4-5 example', () => {
    for (const detail of ['Norm properties', 'triangle inequality',
      '\\|\\alpha x\\|=|\\alpha|', 'A squared Euclidean length',
      'https://d2l.ai/chapter_preliminaries/linear-algebra.html#norms']) expect(text).toContain(detail);
  });

  it('does not treat free access to the older MML figure as a verified reproduction grant', () => {
    const registryPath = '05 Источники/asset-registry.yml';
    const registry = YAML.parse(beforeTopicConsolidation(readFileSync(resolve(root, registryPath), 'utf8'),
      registryPath, root, journalName));
    const figure = registry.assets.find((item: { id: string }) => item.id === 'curated-mml-vector-geometry-00000000009e');
    expect(figure.metadata_status).toBe('needs-confirmation');
    expect(figure.rights_status).toBe('publisher-figure-reproduction-unverified');
    expect(figure.license).toContain('no reproduction grant confirmed');
    expect(text).toContain(figure.asset);
  });
});
