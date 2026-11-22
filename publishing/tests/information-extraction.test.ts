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
const ruPath = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03c Извлечение информации из документов.md';
const enPath = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03c Information extraction from documents.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-information-extraction-editorial-pass.json';
const review = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());
const blocks = (text: string, language: string) =>
  [...text.matchAll(new RegExp('^~~~' + language + '\\n([\\s\\S]*?)^~~~$', 'gm'))].map(m => m[1]);

describe('Information extraction from documents', () => {
  it('pins two real editions with the same executable example and equation', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(review.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(review.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      locale: 'en', translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    expect(blocks(ru, 'python')).toHaveLength(1);
    expect(blocks(en, 'python')).toEqual(blocks(ru, 'python'));
    expect(() => execFileSync('python3', ['-c', blocks(ru, 'python')[0]])).not.toThrow();
    const math = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(math(ru)).toHaveLength(1);
    expect(math(en)).toEqual(math(ru));
  });

  it('checks the printed event and stage-level metrics in both editions', () => {
    for (const text of [ru, en]) {
      expect(blocks(text, 'yaml')).toHaveLength(1);
      expect(YAML.parse(blocks(text, 'yaml')[0])).toEqual({
        event: 'assume_directorship', person: 'P17', organization: 'O01',
        year: 2023, evidence_sentences: [1, 2],
      });
      expect(text).toContain('$F_1=6/7$');
      for (const denominator of ['$8/10$', '$6/8$', '$6/10$']) expect(text).toContain(denominator);
    }
  });

  it('preserves source attribution and the same three images with localized descriptions', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const r = figures(ru), e = figures(en);
    expect(r).toHaveLength(3);
    expect(e.map(([name]) => name)).toEqual(r.map(([name]) => name));
    for (const [k, [path, alt]] of r.entries()) {
      expect(e[k][1]).not.toBe(alt);
      const row = registry.assets.find((entry: {asset: string}) => entry.asset === path);
      expect(row).toMatchObject({
        license: 'CC BY 4.0', rights_status: 'source-license-verified', sha256: hash(read(path)),
        used_in: [ruPath, enPath],
      });
      expect(ru).toContain(row.source_url);
      expect(en).toContain(row.source_url);
    }
  });
});
