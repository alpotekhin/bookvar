import { existsSync, readFileSync, readdirSync } from 'node:fs';
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
      expect(typeof asset.asset).toBe('string');
      expect(asset.author === null || typeof asset.author === 'string').toBe(true);
      expect(asset.source_url === null || typeof asset.source_url === 'string').toBe(true);
      expect(asset.license === null || typeof asset.license === 'string').toBe(true);
      expect(asset.license_url === null || typeof asset.license_url === 'string').toBe(true);
      expect(typeof asset.modifications).toBe('string');
      expect(['verified', 'needs-confirmation']).toContain(asset.metadata_status);
      expect(Array.isArray(asset.used_in)).toBe(true);
      for (const usage of asset.used_in as unknown[]) {
        expect(typeof usage).toBe('string');
        expect(existsSync(resolve(root, usage as string)), `Missing used_in note: ${String(usage)}`).toBe(true);
      }

      if (asset.metadata_status === 'verified') {
        expect(typeof asset.author).toBe('string');
        expect(typeof asset.source_url).toBe('string');
        expect(typeof asset.license).toBe('string');
        expect(typeof asset.license_url).toBe('string');
      }
    }

    const attention = registry.assets.find((asset) =>
      asset.asset === '00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png'
    );
    expect(attention?.used_in).toEqual([
      '00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md'
    ]);

    const deepseekR1 = registry.assets.find((asset) =>
      asset.asset === '00 Учебник/Assets/Figures/deepseek-r1-figure1-hq.png'
    );
    expect(deepseekR1?.used_in).toEqual([
      '00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md',
      '00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока.md'
    ]);
  });
});
