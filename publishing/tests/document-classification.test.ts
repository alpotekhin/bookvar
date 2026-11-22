import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import matter from 'gray-matter';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const sha = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const ruPath = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов.md';
const enPath = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03b Document classification.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const receipt = JSON.parse(read('docs/audit/2026-11-22-document-classification-editorial-pass.json').toString());

describe('Document classification', () => {
  it('retains the same worked example, equations and original figures in both editions', () => {
    const review = '2026-11-22-document-classification-editorial-pass.json';
    expect(sha(beforeTopicConsolidation(ru, ruPath, root, review))).toBe(receipt.files[0].after_sha256);
    expect(sha(beforeTopicConsolidation(en, enPath, root, review))).toBe(receipt.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({locale: 'en', translation_of: ruPath,
      canonical_revision: sha(ru), translation_status: 'current'});
    const code = (text: string) =>
      [...text.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const formulae = (text: string) =>
      [...text.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    const images = (text: string) =>
      [...text.matchAll(/^!\[\[02 Areas\/ML & DL\/([^\]|]+)\|([^\]]+)\]\]/gm)].map(m => [m[1], m[2]]);
    expect(code(ru)).toHaveLength(1);
    expect(code(en)).toEqual(code(ru));
    expect(formulae(ru)).toHaveLength(1);
    expect(formulae(en)).toEqual(formulae(ru));
    expect(execFileSync('python3', ['-c', code(ru)[0]], {encoding: 'utf8'}).trim())
      .toBe('Bigrams: p(positive) = 0.688, 0.356');
    const r = images(ru), e = images(en);
    expect(r).toHaveLength(2);
    expect(e.map(([file]) => file)).toEqual(r.map(([file]) => file));
    const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());
    for (const [i, [file, alt]] of r.entries()) {
      const figure = receipt.figures[i];
      expect(registry.assets.find((row: {asset: string}) => row.asset === file)).toMatchObject({
        sha256: sha(read(file)), source_url: figure.source, license: figure.license,
        used_in: [ruPath, enPath],
      });
      expect(sha(read(file))).toBe(figure.sha256);
      expect(ru).toContain(figure.source);
      expect(en).toContain(figure.source);
      expect(e[i][1]).not.toBe(alt);
    }
  });
  it('recomputes the pooled vectors, classification head, and threshold tradeoff', () => {
    const vectors = [[1, 0], [0, 0.5], [0.5, -0.5]];
    const mean = vectors[0].map((_, j) => vectors.reduce((s, row) => s + row[j], 0) / vectors.length);
    expect(mean).toEqual([0.5, 0]);
    expect(mean[0] - mean[1]).toBe(0.5);
    const context = [[0.6, -0.2], [-0.2, 0.6]];
    const probs = context.map(([a, b]) => 1 / (1 + Math.exp(-(a - b + 0.2))));
    expect(probs.map(p => p.toFixed(3))).toEqual(['0.731', '0.354']);
    const result = (tp: number, fp: number, fn: number) => [tp + fp, tp/(tp+fp), tp/(tp+fn)];
    expect(result(40, 60, 10)).toEqual([100, 0.40, 0.80]);
    expect(result(30, 10, 20)).toEqual([40, 0.75, 0.60]);
    expect(ru).toContain('0,75 | 0,60');
    expect(en).toContain('0.75 | 0.60');
  });
});
