import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-30-tensor-native-geometry-additions.json'));
const chapter = journal.files[0].source;
const current = read(chapter);
const text = beforeTopicConsolidation(current, chapter, root,
  '2026-10-30-tensor-native-geometry-additions.json');
const prior = beforeTopicConsolidation(current, chapter, root,
  '2026-10-30-tensor-vector-operands-additions.json');
const figures = JSON.parse(read('00 Учебник/Assets/Figures/curated/tensor-native-geometry-2026-10-30/assets.json'));

describe('native D2L geometry in tensor foundations', () => {
  it('restores exactly the preceding chapter, registry and shrinking legacy list', () => {
    for (const row of journal.files) {
      const current = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-30-tensor-native-geometry-additions.json');
      expect(sha(current)).toBe(row.after_sha256);
      expect(sha(beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-30-tensor-vector-operands-additions.json'))).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(read(row.source) + '\nUNREVIEWED\n', row.source, root,
        '2026-10-30-tensor-vector-operands-additions.json')).toThrow();
    }
  });

  it('pins the two complete source SVGs and their public CC BY-SA notice', () => {
    expect(figures).toHaveLength(2);
    const expected = [
      ['vec-add.svg', 'c4cafd62fbe2eaff376fb6142f4bbd78d2cb5cfc90448042db61a91a1267f384'],
      ['vec-angle.svg', '46cda699171505338dbd063fd66b0cbacecd95eb8db16b5bc89a7456380e9c3f']
    ];
    const registry = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    for (const [index, row] of figures.entries()) {
      const [name, digest] = expected[index];
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/' + name);
      expect(sha(readFileSync(resolve(root, row.asset)))).toBe(digest);
      expect(row.source_asset).toBe(row.asset);
      expect(row.sha256).toBe(digest);
      expect(row.license).toBe('CC BY-SA 4.0');
      expect(row.derivation).toBe('source-svg-copy');
      expect(row.modifications).toContain('Complete original SVG copied byte-for-byte');
      expect(sha(read(row.public_license_copy))).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
      expect(registry.find((item: { id: string }) => item.id === row.id)).toEqual(row);
      expect(text).toContain(row.asset);
    }
  });

  it('keeps all numerical explanations and six other images while linking the MML original', () => {
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    const oldMml = '02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/foundations-source-first/mml-vector-geometry.png';
    expect(images(text)).toHaveLength(8);
    for (const image of images(prior).filter(item => item !== oldMml)) expect(images(text)).toContain(image);
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    expect(fences(text)).toEqual(fences(prior));
    expect(text).toContain('f(t)=1+2t');
    expect(text).toContain('https://mml-book.github.io/book/mml-book.pdf#page=23');
    expect(text).not.toContain('mml-vector-geometry.png');
    expect(existsSync(resolve(root, '00 Учебник/Assets/Figures/curated/foundations-source-first/mml-vector-geometry.png'))).toBe(false);
    expect(YAML.parse(read('05 Источники/asset-registry.yml')).assets.some((row: { id: string }) =>
      row.id === 'curated-mml-vector-geometry-00000000009e')).toBe(false);
  });
});
