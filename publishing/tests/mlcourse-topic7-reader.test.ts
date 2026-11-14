import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { mlcourseTopic7ReaderBody } from '../adapter/mlcourse-topic7-reader.js';

const root = new URL('../../', import.meta.url);
const read = (path: string) => readFileSync(new URL(path, root));
const route = 'sources/courses/advanced-unsupervised-classical/mlcourse-topic7';
const path = '05 Источники/Courses/Advanced Unsupervised and Classical ML/mlcourse-topic7.html.md';
const raw = read(path).toString();
const transformed = mlcourseTopic7ReaderBody(route, path, raw);
const evidence = JSON.parse(read('publishing/adapter/mlcourse-topic7-evidence.json').toString()) as {
  plots: Record<string, { asset: string; alt: string }>;
  codeCells: Array<{ block: number; source: string }>;
  thirdParty: Record<string, { sha256: string }>;
};
const blocks = (text: string) => [...text.matchAll(/^\x60{3}(python)?\n([\s\S]*?)^\x60{3}$/gm)];
const normalized = (text: string) => text.replace(/\s+/g, ' ').trim();
const sha256 = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

describe('mlcourse.ai Topic 7 reader', () => {
  it('restores the pinned Python indentation without altering original outputs', () => {
    const original = blocks(raw);
    const rendered = blocks(transformed);
    expect(original).toHaveLength(23);
    expect(rendered).toHaveLength(original.length);
    expect(rendered.filter(match => match[1] === 'python')).toHaveLength(18);
    for (const [index, match] of rendered.entries()) {
      const cell = evidence.codeCells.find(entry => entry.block === index);
      if (cell) {
        expect(match[2]).toBe(cell.source.replace(/^\n+/, '').trimEnd() + '\n');
        expect(normalized(match[2])).toBe(normalized(original[index][2]));
      } else {
        expect(match[0]).toBe(original[index][0]);
      }
    }
    expect(transformed).toMatch(/for i in range\(10\):\n {4}plt\.subplot/);
    expect(transformed).toMatch(/for algo in algorithms:\n {4}algo\.fit\(X\)/);
    expect(() => mlcourseTopic7ReaderBody(route, path, raw.replace('import numpy as np', 'import math')))
      .toThrow(/no longer matches the reviewed code block/);
    expect(mlcourseTopic7ReaderBody('textbook/example', path, raw)).toBe(raw);
    expect(mlcourseTopic7ReaderBody(route, 'other.md', raw)).toBe(raw);
  });

  it('keeps the ten original outputs locally with attribution and separate third-party provenance', () => {
    const folder = 'Assets/Sources/mlcourse.ai/topic-7/';
    expect(Object.keys(evidence.plots)).toHaveLength(10);
    for (const [hash, row] of Object.entries(evidence.plots)) {
      expect(sha256(read(folder + row.asset))).toBe(hash);
      expect(transformed).toContain('![[' + folder + row.asset + '|' + row.alt + ']]');
      expect(transformed).toContain('[original PNG](https://mlcourse.ai/_images/' + hash + '.png)');
    }
    expect(sha256(read(folder + 'GaussianScatterPCA.svg'))).toBe(evidence.thirdParty.pca.sha256);
    expect(transformed).toContain('Nicoguaro, [GaussianScatterPCA.svg]');
    expect(transformed).toContain('This gallery version is not the historical screenshot');
    expect(transformed).not.toMatch(/!\[[^\]\n]*\]\(https?:/);
  });

  it('reconstructs the original metrics table and identifies limitations of the examples', () => {
    expect(transformed).toContain('| Algorithm | ARI | AMI | Homogeneity | Completeness | V-measure | Silhouette |');
    expect(transformed).toContain('| Spectral | 0.756461 | 0.852040 | 0.831691 | 0.876614 | 0.853562 | 0.182729 |');
    expect(transformed).toContain('| Agglomerative | 0.794003 | 0.866832 | 0.857513 | 0.879096 | 0.868170 | 0.178497 |');
    for (const detail of ['test rows influence', '41/45 (91.1%)', '89.43%', '90.32%', 'both ARI and AMI can be negative',
      'square roots of inertia', 'median, not a mean']) {
      expect(transformed.toLowerCase()).toContain(detail.toLowerCase());
    }
  });

  it('converts the lesson’s TeX delimiters without changing equations or Python strings', () => {
    expect(transformed).toContain('from $n$ to $k$ with $k \\leq n$');
    expect(transformed).toContain('$$\n\\text{RI} = \\frac{2(a + b)}{n(n-1)}.\n$$');
    expect(transformed).toContain('plt.ylabel("$J(C_k)$");');
    expect(transformed.match(/^\$\$$/gm)).toHaveLength(24);
  });
});
