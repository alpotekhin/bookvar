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
const ruPath = '00 Учебник/02 Представление текста и токенизация/04 Синтаксический разбор.md';
const enPath = 'en/00 Textbook/02 Text representation and tokenization/04 Syntactic parsing.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-syntactic-parsing-editorial-pass.json';
const audit = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());

describe('Syntactic parsing', () => {
  it('pins the two editions, the runnable exercise, and the scored recurrence', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(audit.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(audit.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      locale: 'en', translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    const code = (s: string) => [...s.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const equations = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(equations(ru)).toHaveLength(1);
    expect(equations(en)).toEqual(equations(ru));
    expect(code(ru)).toHaveLength(1);
    expect(code(en)).toEqual(code(ru));
    expect(() => execFileSync('python3', ['-c', code(ru)[0]])).not.toThrow();
  });

  it('checks the attachment example directly from both printed tables', () => {
    for (const text of [ru, en]) {
      // The dependent ID is the first table column. Compare HEAD and DEPREL separately.
      const rows = [...text.matchAll(/^\| (\d+) \| (\w+) \| `(\d+), (\w+)` \| `(\d+), (\w+)` \|$/gm)];
      expect(rows).toHaveLength(7);
      expect(rows.filter(row => row[3] === row[5])).toHaveLength(6);
      expect(rows.filter(row => row[3] === row[5] && row[4] === row[6])).toHaveLength(6);
      expect(rows[6].slice(1)).toEqual(['7', 'telescope', '2', 'obl', '4', 'nmod']);
      expect(text).toContain('$F_1=5/6$');
    }
  });

  it('keeps the two licensed originals and their localized descriptions', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const images = figures(ru);
    const alternate = figures(en);
    expect(images).toHaveLength(2);
    expect(alternate.map(([path]) => path)).toEqual(images.map(([path]) => path));
    for (const [i, [path, alt]] of images.entries()) {
      expect(alternate[i][1]).not.toBe(alt);
      const row = registry.assets.find((item: { asset: string }) => item.asset === path);
      expect(row).toMatchObject({
        license: 'CC BY 4.0', sha256: hash(read(path)),
        rights_status: 'source-license-verified', used_in: [ruPath, enPath],
      });
      expect(ru).toContain(row.source_url);
      expect(en).toContain(row.source_url);
    }
  });
});
