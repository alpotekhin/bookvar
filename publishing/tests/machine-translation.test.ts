import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import matter from 'gray-matter';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const ruPath = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод.md';
const enPath = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/08 Machine translation.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-machine-translation-editorial-pass.json';
const review = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());

describe('Machine translation', () => {
  it('pins real editions with the same loss and executable example', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(review.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(review.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      locale: 'en', translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    const blocks = (s: string) => [...s.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const math = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(blocks(ru)).toHaveLength(1);
    expect(blocks(en)).toEqual(blocks(ru));
    expect(math(ru)).toHaveLength(1);
    expect(math(en)).toEqual(math(ru));
    expect(() => execFileSync('python3', ['-c', blocks(ru)[0]])).not.toThrow();
  });

  it('keeps both alternatives in the printed table, with the same meaning contrast', () => {
    const inputs = (s: string) => [...s.matchAll(/^\| [^|]+\| `(do (?:not )?close the valve before noon|before noon do not shut the valve)` \|$/gm)]
      .map(match => match[1]);
    const pair = [
      'do not close the valve before noon',
      'do close the valve before noon',
      'before noon do not shut the valve',
    ];
    expect(inputs(ru)).toEqual(pair);
    expect(inputs(en)).toEqual(pair);
    expect(ru).toContain('$67{,}3$');
    expect(en).toContain('$67.3$');
  });

  it('keeps all three licensed source figures with localized alt text', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const a = figures(ru), b = figures(en);
    expect(a).toHaveLength(3);
    expect(b.map(([path]) => path)).toEqual(a.map(([path]) => path));
    for (const [i, [path, alt]] of a.entries()) {
      expect(b[i][1]).not.toBe(alt);
      const row = registry.assets.find((item: {asset: string}) => item.asset === path);
      expect(row).toMatchObject({
        license: 'CC BY 4.0', rights_status: 'source-license-verified',
        sha256: hash(read(path)), used_in: [ruPath, enPath],
      });
      expect(ru).toContain(row.source_url);
      expect(en).toContain(row.source_url);
    }
  });
});
