import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import matter from 'gray-matter';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const hash = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');
const ruPath = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/09 Суммаризация текста.md';
const enPath = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/09 Text summarization.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-summarization-editorial-pass.json';
const audit = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());

describe('Text summarization', () => {
  it('keeps current editions and runs their common numerical example', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(audit.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(audit.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    const blocks = (s: string) => [...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}$/gm)].map(m => m[1]);
    const math = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(blocks(ru)).toHaveLength(1);
    expect(blocks(en)).toEqual(blocks(ru));
    expect(math(ru)).toHaveLength(1);
    expect(math(en)).toEqual(math(ru));
    expect(execFileSync('python3', ['-c', blocks(ru)[0]], { encoding: 'utf8' }).trim())
      .toBe('[2, 3] 0.17 0.632 0.5');
    for (const [s, wrong, faithful] of [[ru, '0,900', '0,632'], [en, '0.900', '0.632']]) {
      expect(s).toContain('| 9 | 9/10 | 9/10 | ' + wrong + ' |');
      expect(s).toContain('| 6 | 6/10 | 6/9 | ' + faithful + ' |');
    }
  });

  it('uses the same three original figures with localized descriptions', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const a = figures(ru), b = figures(en);
    expect(a).toHaveLength(3);
    expect(b.map(([path]) => path)).toEqual(a.map(([path]) => path));
    for (const [i, [path, ruAlt]] of a.entries()) {
      expect(b[i][1]).not.toBe(ruAlt);
      const row = registry.assets.find((asset: {asset:string}) => asset.asset === path);
      expect(row).toMatchObject({ license: 'CC BY 4.0',
        rights_status: 'source-license-verified', sha256: hash(read(path)), used_in: [ruPath, enPath] });
      expect(ru).toContain(row.source_url);
      expect(en).toContain(row.source_url);
    }
  });
});
