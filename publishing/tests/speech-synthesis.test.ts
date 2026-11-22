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
const ruPath = '00 Учебник/16 Multimodal Models/64eb Синтез речи.md';
const enPath = 'en/00 Textbook/16 Multimodal Models/64eb Text-to-speech synthesis.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const journal = '2026-11-22-speech-synthesis-editorial-pass.json';
const audit = JSON.parse(read('docs/audit/' + journal).toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());

describe('Text-to-speech synthesis', () => {
  it('pins both editions and verifies their common worked example', () => {
    expect(hash(beforeTopicConsolidation(ru, ruPath, root, journal))).toBe(audit.files[0].after_sha256);
    expect(hash(beforeTopicConsolidation(en, enPath, root, journal))).toBe(audit.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      translation_of: ruPath, canonical_revision: hash(ru), translation_status: 'current',
    });
    const blocks = (s: string) => [...s.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    expect(blocks(ru)).toHaveLength(1);
    expect(blocks(en)).toEqual(blocks(ru));
    expect(execFileSync('python3', ['-c', blocks(ru)[0]], { encoding: 'utf8' }).trim())
      .toBe('alignment 13; mel 20 frames; PCM 4800 samples; RVQ 40 indices');
    for (const text of [ru, en]) {
      expect(text).toContain('$(1,2,2,1,3)$');
      expect(text).toContain('$(1,2,2,2,3)$');
      expect(text).toContain('$20\\cdot240=4800$');
    }
  });

  it('has three licensed original diagrams and localized captions', () => {
    const figures = (s: string) => [...s.matchAll(/^!\[\[02 Areas\/ML & DL\/([^|]+)\|([^\]]+)\]\]/gm)]
      .map(m => m.slice(1));
    const a = figures(ru), b = figures(en);
    expect(a).toHaveLength(3);
    expect(b.map(([path]) => path)).toEqual(a.map(([path]) => path));
    for (const [i, [path, ruAlt]] of a.entries()) {
      expect(b[i][1]).not.toBe(ruAlt);
      const row = registry.assets.find((asset: {asset:string}) => asset.asset === path);
      expect(row).toMatchObject({ license: 'CC BY 4.0', rights_status: 'source-license-verified',
        sha256: hash(read(path)), used_in: [ruPath, enPath] });
      for (const chapter of [ru, en]) {
        expect(chapter).toContain(row.source_url);
        expect(chapter).toContain(row.source_image_url);
      }
    }
  });
});
