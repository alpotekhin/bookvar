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
const ruPath = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей.md';
const enPath = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03a Named entity recognition.md';
const ru = read(ruPath).toString(), en = read(enPath).toString();
const audit = JSON.parse(read('docs/audit/2026-11-22-ner-editorial-pass.json').toString());

describe('Named entity recognition', () => {
  it('pins the full editions, example, and two unmodified source figures', () => {
    const review = '2026-11-22-ner-editorial-pass.json';
    expect(sha(beforeTopicConsolidation(ru, ruPath, root, review))).toBe(audit.files[0].after_sha256);
    expect(sha(beforeTopicConsolidation(en, enPath, root, review))).toBe(audit.added_sources[0].sha256);
    expect(matter(en).data).toMatchObject({
      locale: 'en', translation_of: ruPath, canonical_revision: sha(ru), translation_status: 'current',
    });
    const code = (text: string) => [...text.matchAll(/^~~~python\n([\s\S]*?)^~~~$/gm)].map(m => m[1]);
    const equations = (text: string) => [...text.matchAll(/^\$\$\n([\s\S]*?)^\$\$$/gm)].map(m => m[1]);
    expect(code(ru)).toHaveLength(1);
    expect(code(en)).toEqual(code(ru));
    expect(equations(ru)).toHaveLength(1);
    expect(equations(en)).toEqual(equations(ru));
    expect(execFileSync('python3', ['-c', code(ru)[0]], { encoding: 'utf8' }).trim())
      .toBe('tp=0 fp=3 fn=3; F1=0.0');
    const images = (text: string) =>
      [...text.matchAll(/^!\[\[02 Areas\/ML & DL\/([^\]|]+)\|([^\]]+)\]\]/gm)].map(m => [m[1], m[2]]);
    const ruImages = images(ru), enImages = images(en);
    expect(ruImages).toHaveLength(2);
    expect(ruImages.map(([file]) => file)).toEqual(enImages.map(([file]) => file));
    const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString());
    for (const [i, [file, alt]] of ruImages.entries()) {
      const witness = audit.figures[i];
      const record = registry.assets.find((row: {asset: string}) => row.asset === file);
      expect(record).toMatchObject({
        sha256: sha(read(file)), license: 'CC BY 4.0', source_url: witness.source,
        used_in: [ruPath, enPath],
      });
      expect(record.sha256).toBe(witness.sha256);
      expect(ru).toContain(witness.source);
      expect(en).toContain(witness.source);
      expect(enImages[i][1]).not.toBe(alt);
    }
  });
  it('recomputes exact span and token scores independently', () => {
    const text = 'Maya Chen joined Hugging Face in New York last spring.';
    const spans = [['Maya Chen', 'PER'], ['Hugging Face', 'ORG'], ['New York', 'LOC']];
    const gold = spans.map(([name, type]) => [text.indexOf(name), text.indexOf(name) + name.length, type]);
    expect(gold).toEqual([[0, 9, 'PER'], [17, 29, 'ORG'], [33, 41, 'LOC']]);
    const short = spans.map(([name, type]) => [text.indexOf(name), text.indexOf(name) + name.indexOf(' '), type]);
    expect(short).toEqual([[0, 4, 'PER'], [17, 24, 'ORG'], [33, 36, 'LOC']]);
    const len = [...text.matchAll(/[a-z]+|\./gi)].length;
    const exact = (guess: (number|string)[][]) => {
      const tp = guess.filter(span => gold.some(g => JSON.stringify(g) === JSON.stringify(span))).length;
      return 2 * tp / (gold.length + guess.length);
    };
    expect(len).toBe(11);
    expect((len - 3 + 89) / (len + 89)).toBe(0.97);
    expect(exact(short)).toBe(0);
    expect(exact([gold[0], ...short.slice(1)])).toBe(1/3);
    expect(ru).toContain('$97/100$');
    expect(en).toContain('$97/100$');
  });
});
