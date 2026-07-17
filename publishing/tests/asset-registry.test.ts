import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');

describe('publication asset registry', () => {
  it('records every tracked textbook figure with reviewable rights metadata', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: Array<Record<string, unknown>>;
    };
    const paths = registry.assets.map((asset) => asset.asset);
    const figuresRoot = resolve(root, '00 Учебник/Assets/Figures');
    const files = readdirSync(figuresRoot, { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile())
      .map((entry) => `00 Учебник/Assets/Figures/${entry.parentPath.slice(figuresRoot.length + 1)}${entry.parentPath === figuresRoot ? '' : '/'}${entry.name}`)
      .sort();

    expect(paths).toHaveLength(new Set(paths).size);
    expect(paths.toSorted()).toEqual(files);
    expect(paths).toContain('00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png');
    for (const asset of registry.assets) {
      expect(asset).toHaveProperty('author');
      expect(asset).toHaveProperty('source_url');
      expect(asset).toHaveProperty('license');
      expect(asset).toHaveProperty('license_url');
      expect(asset).toHaveProperty('modifications');
      expect(asset).toHaveProperty('used_in');
      expect(asset).toHaveProperty('metadata_status');
    }
  });
});
