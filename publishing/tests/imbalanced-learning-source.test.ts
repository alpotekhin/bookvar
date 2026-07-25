import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const sourceRoot = resolve(root, '05 Источники/imbalanced-learn/0.14.2');

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

describe('imbalanced-learn source-native corpus', () => {
  it('pins every imported original to the 0.14.2 upstream commit and SHA-256', () => {
    const manifest = parse(readFileSync(resolve(sourceRoot, 'manifest.yml'), 'utf8')) as {
      source: {
        version: string;
        commit: string;
        license: string;
      };
      files: Array<{
        path: string;
        upstream_path: string;
        sha256: string;
      }>;
    };

    expect(manifest.source).toMatchObject({
      version: '0.14.2',
      commit: '8504e95f0160f61d1b617ca66f779646d2ee609e',
      license: 'MIT'
    });
    expect(manifest.files).toHaveLength(20);
    expect(new Set(manifest.files.map(({ path }) => path)).size).toBe(20);

    for (const file of manifest.files) {
      const localPath = resolve(sourceRoot, file.path);
      expect(existsSync(localPath), `Missing imported source: ${file.path}`).toBe(true);
      expect(file.upstream_path).not.toMatch(/^\.|\/\.\.\//);
      expect(sha256(localPath), `Source drift: ${file.path}`).toBe(file.sha256);
    }
  });

  it('registers all four official gallery figures with pinned provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as {
      assets: Array<{
        asset: string;
        upstream_commit?: string;
        sha256?: string;
      }>;
    };
    const entries = registry.assets.filter(({ asset }) =>
      asset.startsWith('00 Учебник/Assets/Figures/imbalanced-learning/')
    );

    expect(entries).toHaveLength(4);
    for (const entry of entries) {
      expect(entry.upstream_commit).toBe('8504e95f0160f61d1b617ca66f779646d2ee609e');
      expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
    }
  });

  it('publishes every pinned original as a readable page on exactly 21 routes', () => {
    const publication = JSON.parse(
      readFileSync(resolve(sourceRoot, 'published/import-manifest.json'), 'utf8')
    ) as {
      page_count: number;
      route_count: number;
      index_route: string;
      pages: Array<{
        kind: string;
        source_sha256: string;
        output: string;
        route: string;
      }>;
    };

    expect(publication).toMatchObject({
      page_count: 20,
      route_count: 21,
      index_route: 'sources/imbalanced-learn/0-14-2'
    });
    expect(publication.pages).toHaveLength(20);
    expect(new Set(publication.pages.map(({ route }) => route)).size).toBe(20);

    for (const page of publication.pages) {
      const output = resolve(root, page.output);
      expect(existsSync(output), `Missing readable source page: ${page.output}`).toBe(true);
      const markdown = readFileSync(output, 'utf8');
      expect(markdown).toContain(`source_sha256: ${page.source_sha256}`);
      expect(markdown).toContain('Pinned original source');
      expect(markdown).toMatch(/^#\s+\S/m);
      expect(markdown).not.toMatch(/^\.\. (?:currentmodule|image|math|topic|warning|note)::/m);
      expect(markdown).not.toMatch(/:(?:class|func|mod|math|cite|ref):`/);
    }

    const navigation = readFileSync(resolve(root, 'publishing/navigation.yml'), 'utf8');
    const registeredRoutes = navigation.match(
      /^\s+route: sources\/imbalanced-learn\/0-14-2(?:\/[^\s]+)?$/gm
    );
    expect(registeredRoutes).toHaveLength(21);
  });

  it('interleaves gallery narrative and code instead of dumping raw Python', () => {
    const publication = JSON.parse(
      readFileSync(resolve(sourceRoot, 'published/import-manifest.json'), 'utf8')
    ) as {
      pages: Array<{ kind: string; output: string }>;
    };
    const gallery = publication.pages.filter(({ kind }) => kind === 'gallery-example');
    expect(gallery).toHaveLength(8);

    for (const page of gallery) {
      const markdown = readFileSync(resolve(root, page.output), 'utf8');
      expect(markdown).toContain('```python');
      expect(markdown).not.toContain('# %% [markdown]');
      const narrative = markdown
        .replace(/^---[\s\S]*?---/m, '')
        .replace(/```python[\s\S]*?```/g, '')
        .replace(/^>.*$/gm, '');
      expect(narrative.match(/[A-Za-z]{3,}/g)?.length ?? 0).toBeGreaterThan(20);
    }
  });

  it('renders all 27 bibliography entries as individually addressable records', () => {
    const bibliography = readFileSync(
      resolve(sourceRoot, 'published/user-guide/bibliography.md'),
      'utf8'
    );
    expect(bibliography).toContain('**27 complete BibTeX entries**');
    expect(bibliography.match(/^##\s+\S+$/gm)).toHaveLength(27);
    expect(bibliography.match(/<summary>Original BibTeX record<\/summary>/g)).toHaveLength(27);
  });
});
