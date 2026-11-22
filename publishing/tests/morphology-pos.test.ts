import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import matter from 'gray-matter';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
const ruPath = '00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей.md';
const enPath = 'en/00 Textbook/02 Text representation and tokenization/03 Morphology and sequence labeling.md';
const ru = read(ruPath), en = read(enPath);
const registry = YAML.parse(read('05 Источники/asset-registry.yml'));
const audit = JSON.parse(read('docs/audit/2026-11-22-morphology-pos-editorial-pass.json'));

describe('Morphology and sequence labeling', () => {
  it('pins both read editions and preserves the examples and figures', () => {
    expect(sha(ru)).toBe(audit.files.find((row: {source: string}) => row.source === ruPath).after_sha256);
    expect(sha(en)).toBe(audit.added_sources.find((row: {source: string}) => row.source === enPath).sha256);
    expect(matter(en).data).toMatchObject({
      locale: 'en',
      translation_status: 'current',
      translation_of: ruPath,
      canonical_revision: sha(ru),
    });
    const examples = (text: string) => [...text.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const figures = (text: string) => [...text.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)].map(m => m.slice(1));
    const formulae = (text: string) => Array.from(text.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm),
      m => m[1].replace(/(\d)\{,\}(?=\d)/g, '$1.'));
    expect(examples(ru)).toHaveLength(1);
    expect(examples(en)).toEqual(examples(ru));
    expect(formulae(ru)).toHaveLength(2);
    expect(formulae(en)).toEqual(formulae(ru));
    const images = figures(ru);
    expect(images).toHaveLength(2);
    expect(figures(en).map(([path]) => path)).toEqual(images.map(([path]) => path));
    for (const [i, [path, alt]] of images.entries()) {
      expect(alt).not.toBe(figures(en)[i][1]);
      const row = registry.assets.find((item: {asset: string}) => item.asset === path);
      expect(row).toMatchObject({
        source_page: [5, 14][i],
        author: 'Herman Kamper',
        license: 'CC BY-SA 4.0',
        sha256: sha(readFileSync(resolve(root, path))),
        used_in: [ruPath, enPath],
      });
      expect(ru).toContain(row.source_url);
      expect(en).toContain(row.source_url);
    }
    expect(execFileSync('python3', ['-c', examples(ru)[0]], {encoding: 'utf8'}).trim())
      .toBe("('PRON', 'VERB', 'VERB') 0.0018");
  });

  it('recomputes the source trellis, the CRF gradient, and the feature scores', () => {
    const a = [[0.2, 0.4, 0.4], [0.7, 0.1, 0.2]];
    const b = [[0.6, 0.2, 0.2], [0.1, 0.6, 0.3]];
    const scores = [0, 1].flatMap(second => [0, 1].map(third =>
      [b[0][0] * a[0][second] * b[second][1] * a[second][third] * b[third][2] * a[third][2],
        [0, second, third]] as const));
    const [best, path] = scores.sort(([p], [q]) => q - p)[0];
    expect(path).toEqual([0, 1, 0]);
    expect(best).toBeCloseTo(0.008064, 10);

    const noun = 1.1 - 0.8, verb = 0.8 + 0.3;
    const p = Math.exp(verb) / (Math.exp(noun) + Math.exp(verb));
    expect(p.toFixed(3)).toBe('0.690');
    expect((-Math.log(p)).toFixed(3)).toBe('0.371');
    expect((p - 1).toFixed(3)).toBe('-0.310');

    const gold = [['Plur', '3'], ['Plur', 'Pres'], ['Plur', 'Acc'], []];
    const predicted = [['Sing', '3'], ['Plur', 'Pres'], ['Sing', 'Acc'], []];
    const matches = gold.flatMap((row, i) => row.filter((value, j) => value === predicted[i][j]));
    expect(gold.filter((row, i) => row.every((value, j) => value === predicted[i][j])).length).toBe(2);
    expect(2 * matches.length / (gold.flat().length + predicted.flat().length)).toBe(2 / 3);
    expect(ru).toContain('micro-F1 тоже равна $2/3$');
    expect(en).toContain('micro-F1 is $2/3$');
  });
});
