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
const ruPath = '00 Учебник/16 Multimodal Models/64ea Распознавание речи.md';
const enPath = 'en/00 Textbook/16 Multimodal Models/64ea Automatic speech recognition.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-speech-recognition-editorial-pass.json';
const audit = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());

describe('Automatic speech recognition', () => {
  it('pins current and historical editions and checks their numerical example', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(audit.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(audit.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    const blocks = (s: string) => [...s.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const math = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(blocks(ru)).toHaveLength(1);
    expect(blocks(en)).toEqual(blocks(ru));
    expect(math(ru)).toHaveLength(2);
    expect(math(en)).toEqual(math(ru));
    expect(execFileSync('python3', ['-c', blocks(ru)[0]], { encoding: 'utf8' }).trim())
      .toBe('CTC LL 0.3384; CTC L 0.5832; WER 0.60; first partial 315 ms');
  });

  it('keeps four original licensed figures, cited and described in both languages', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const a = figures(ru), b = figures(en);
    expect(a).toHaveLength(4);
    expect(b.map(([path]) => path)).toEqual(a.map(([path]) => path));
    for (const [i, [path, ruAlt]] of a.entries()) {
      expect(b[i][1]).not.toBe(ruAlt);
      const row = registry.assets.find((asset: {asset:string}) => asset.asset === path);
      expect(row).toMatchObject({ license: 'CC BY 4.0',
        rights_status: 'source-license-verified', sha256: hash(read(path)), used_in: [ruPath, enPath] });
      for (const chapter of [ru, en]) {
        expect(chapter).toContain(row.source_url);
        expect(chapter).toContain(row.source_image_url);
      }
    }
  });
});
