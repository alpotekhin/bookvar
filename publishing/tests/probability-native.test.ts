import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-30-probability-native-additions.json'));
const chapter = journal.files[0].source;
const current = read(chapter);
const nativeRevision = beforeTopicConsolidation(current, chapter, root,
  '2026-10-30-probability-native-additions.json');
const prior = beforeTopicConsolidation(current, chapter, root,
  '2026-10-30-derivative-native-additions.json');
const figures = JSON.parse(read('00 Учебник/Assets/Figures/curated/probability-native-2026-10-30/assets.json'));

describe('source-language probability explanations and native figures', () => {
  it('restores exactly the reviewed chapter and preceding registry without rebaselining them', () => {
    expect(journal.files[0].before_sha256)
      .toBe('f658322dcd6ebd32a9d888aa3f612bfd9821e8adc04bdeffc813b1ebc10c6505');
    expect(journal.files[1].before_sha256)
      .toBe('9f9bbde15ca3b38e241547c28edf9125be30f088590c46cfaf6494bc7f88d772');
    for (const row of journal.files) {
      const live = read(row.source);
      const text = beforeTopicConsolidation(live, row.source, root,
        '2026-10-30-probability-native-additions.json');
      expect(sha(text)).toBe(row.after_sha256);
      expect(sha(beforeTopicConsolidation(live, row.source, root,
        '2026-10-30-derivative-native-additions.json'))).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(live + '\nUNREVIEWED\n', row.source, root,
        '2026-10-30-derivative-native-additions.json')).toThrow();
    }
  });

  it('retains the earlier program and numerical examples while extending the explanation', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    expect(fences(current)).toHaveLength(1);
    expect(fences(nativeRevision)).toEqual(fences(prior));
    expect(fences(current)[0]).toContain('answer_nll, 0.9485599924429406');
    for (const example of ['24 | 16 | 40', '0{,}6365', '2{,}5903',
      '\\frac{10}{7}', '0{,}375', '\\binom{13}{9}=715'])
      expect(current, example).toContain(example);
    for (const mechanism of ['#### From interval probabilities to a density',
      '#### Cumulative distribution function', '#### Marginalization for a joint density',
      '### Maximum likelihood for continuous variables',
      '#### One observation from an exponential distribution',
      '\\hat\\alpha_\\epsilon', '0.28768', 'not be exactly the same'])
      expect(current, mechanism).toContain(mechanism);
    expect(current).toContain('https://mml-book.github.io/book/mml-book.pdf#page=188');
    expect(current).not.toContain('mml-discrete-continuous-density.png');
    expect([...current.matchAll(/!\[\[([^\]]+)\]\]/g)]).toHaveLength(4);
    for (const file of ['d2l-likelihood-coin.svg', 'd2l-coin-frequency-convergence.svg'])
      expect(current).toContain(file);
    expect(journal.binary_moves).toHaveLength(1);
    expect(existsSync(resolve(root, journal.binary_moves[0].from))).toBe(false);
  });

  it('binds both complete SVGs to their exact original bytes and public source license', () => {
    expect(figures).toHaveLength(2);
    const registry = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const hashes = ['35d85344b8bd9faf7cda96b89bfd235754990ec381f3fdb0ca9d30a812bac27f',
      'cda896c59fd8dfb81f4d5d03fd392142da25b331378e1da668a0088cbd44dd51'];
    for (const [i, row] of figures.entries()) {
      expect(sha(readFileSync(resolve(root, row.asset)))).toBe(hashes[i]);
      expect(row.sha256).toBe(hashes[i]);
      expect(row.source_asset).toBe(row.asset);
      expect(row.source_code_commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
      expect(row.source_code_sha256).toBe('506e48d177d6753b65a671702f4e85422d0501496a085728960799ec4bd1e43f');
      expect(row.modifications).toContain('Complete original SVG copied byte-for-byte');
      expect(row.license).toBe('CC BY-SA 4.0');
      expect(sha(read(row.public_license_copy))).toBe(row.license_evidence_sha256);
      expect(row.used_in).toEqual([chapter]);
      expect(registry.find((item: { id: string }) => item.id === row.id)).toEqual(row);
    }
    expect(figures[0].source_version).toContain('not an immutable image URL');
    expect(figures[1].source_image_url).toContain('/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/marginal.svg');
  });
});
