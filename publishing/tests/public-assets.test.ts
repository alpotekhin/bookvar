import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { verifyPublicAssets } from '../adapter/public-assets.js';
import { loadManifest } from '../adapter/manifest.js';
import { mapMarkdownProse } from '../adapter/markdown-prose.js';

const root = resolve(import.meta.dirname, '../..');
const registry = YAML.parse(readFileSync(join(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
const legacy = JSON.parse(readFileSync(join(import.meta.dirname, 'fixtures/legacy-unhashed-assets.json'), 'utf8'));

describe('distributed image bytes and public notices', () => {
  it('checks the full public registry without requiring private source snapshots', () => {
    const report = verifyPublicAssets(root, registry, new Set(legacy.assets));
    expect(report.assets).toBe(registry.length);
    expect(report.hashed).toBeGreaterThan(1000);
    expect(report.publicNotices).toBeGreaterThan(30);
    expect(report.hashed + report.legacyUnhashed).toBe(report.assets);
    expect(legacy.assets).toEqual(registry.filter((row: { sha256?: string }) => !row.sha256)
      .map((row: { asset: string }) => row.asset).sort());
  });

  it('does not let a previously hashed record fall back into the legacy exception', () => {
    const row = registry.find((item: { sha256?: string }) => item.sha256);
    expect(() => verifyPublicAssets(root, [{ ...row, sha256: undefined }], new Set(legacy.assets)))
      .toThrow('needs SHA-256');
  });

  it('requires every published curated embed to have a registry entry', () => {
    const manifest = loadManifest(join(root, 'publishing/navigation.yml'));
    const assets = new Set<string>(registry.map((row: { asset: string }) => row.asset));
    const missing = new Set<string>();
    for (const page of manifest.sections.flatMap(section => section.pages)) {
      for (const source of [page.source, page.sourceEn].filter(Boolean) as string[]) {
        mapMarkdownProse(readFileSync(join(root, source), 'utf8'), prose => {
          for (const match of prose.matchAll(/!\[\[([^\]\n]+)\]\]/g)) {
            const raw = match[1].split('|', 1)[0].replaceAll('\\', '/');
            const marker = 'ML & DL/';
            const path = raw.includes(marker) ? raw.slice(raw.indexOf(marker) + marker.length) : raw;
            if (path.startsWith('00 Учебник/Assets/Figures/curated/') && !assets.has(path)) missing.add(path);
          }
          return prose;
        });
      }
    }
    expect([...missing]).toEqual([]);
  });

  it('rejects changed bytes, new unhashed assets, missing notices and path escapes', () => {
    const dir = mkdtempSync(join(tmpdir(), 'bookvar-public-assets-'));
    writeFileSync(join(dir, 'figure.svg'), '<svg/>');
    const sha256 = createHash('sha256').update('<svg/>').digest('hex');
    const row = { asset: 'figure.svg', sha256 };
    expect(verifyPublicAssets(dir, [row], new Set()).hashed).toBe(1);
    expect(() => verifyPublicAssets(dir, [{ ...row, sha256: '0'.repeat(64) }], new Set())).toThrow('SHA-256');
    expect(() => verifyPublicAssets(dir, [{ asset: 'figure.svg' }], new Set())).toThrow('needs SHA-256');
    expect(() => verifyPublicAssets(dir, [{ ...row, public_license_copy: 'site/public/licenses/absent.txt' }], new Set())).toThrow();
    expect(() => verifyPublicAssets(dir, [{ asset: '../outside.svg', sha256 }], new Set())).toThrow('escapes');
  });
});
