import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string | Buffer) => createHash('sha256').update(text).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-30-derivative-native-additions.json'));
const chapter = journal.files[0].source;
const current = read(chapter);
const prior = beforeTopicConsolidation(current, chapter, root,
  '2026-10-30-tensor-native-geometry-additions.json');
const figures = JSON.parse(read('00 Учебник/Assets/Figures/curated/derivative-native-2026-10-30/assets.json'));

describe('source-native derivative explanation and figures', () => {
  it('pins and exactly restores the previous chapter, registry and legacy list', () => {
    for (const row of journal.files) {
      const live = read(row.source);
      const text = beforeTopicConsolidation(live, row.source, root,
        '2026-10-30-derivative-native-additions.json');
      expect(sha(text)).toBe(row.after_sha256);
      expect(sha(beforeTopicConsolidation(live, row.source, root,
        '2026-10-30-tensor-native-geometry-additions.json'))).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(live + '\nUNREVIEWED\n', row.source, root,
        '2026-10-30-tensor-native-geometry-additions.json')).toThrow();
    }
  });

  it('retains the original numerical programs and the complete matrix-output convention', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    expect(fences(current)).toHaveLength(3);
    expect(fences(current)).toEqual(fences(prior));
    for (const mechanism of ['Если функция дифференцируема в рассматриваемой точке',
      '### Differentiation rules', '### Several paths through a graph',
      'J_{\\,i+4(j-1),\\,k}=T_{ijk}', '24 производных',
      'f=22x+36y', '0&1&1', 'Неодинаковые множители и векторные состояния'])
      expect(current, mechanism).toContain(mechanism);
    const paths = current.split('### Several paths through a graph')[1]
      .split('## Производная векторной функции: якобиан')[0];
    expect(paths).toContain('v&=-y');
    expect(paths).not.toMatch(/\bz\b/);
    for (const page of ['147', '161', '166']) expect(current).toContain('mml-book.pdf#page=' + page);
    expect([...current.matchAll(/!\[\[([^\]]+)\]\]/g)]).toHaveLength(4);
    expect(current).toContain('google-gradient-descent.png');
    expect(current).toContain('cs231n-chain.svg');
    for (const move of journal.binary_moves) {
      expect(current).not.toContain(move.from);
      expect(existsSync(resolve(root, move.from))).toBe(false);
    }
  });

  it('publishes the two unchanged D2L SVGs with their original language and public license', () => {
    expect(figures).toHaveLength(2);
    const registry = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    const digests = ['fcb32c3fa05846549df55720dc958478f9260507c4d1377a4b31d3940cf9bce2',
      '897746e112a76eb09af174d5f79fbb49a936142567029d92d29a84d602e5cda6'];
    for (const [index, row] of figures.entries()) {
      expect(sha(readFileSync(resolve(root, row.asset)))).toBe(digests[index]);
      expect(row.sha256).toBe(digests[index]);
      expect(row.source_asset).toBe(row.asset);
      expect(row.source_code_commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
      expect(row.modifications).toContain('Complete original SVG copied byte-for-byte');
      expect(row.license).toBe('CC BY-SA 4.0');
      expect(sha(read(row.public_license_copy))).toBe(row.license_evidence_sha256);
      expect(row.used_in).toEqual([chapter]);
      expect(registry.find((r: { id: string }) => r.id === row.id)).toEqual(row);
    }
    const chain = registry.find((r: { asset: string }) => r.asset.endsWith('/backpropagation-2026-10-21/cs231n-chain.svg'));
    expect(chain.used_in).toContain(chapter);
    expect(sha(read(chain.public_license_copy))).toBe(chain.license_evidence_sha256);
  });
});
