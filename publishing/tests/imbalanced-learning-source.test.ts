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
  it('pins the complete local gallery set and retains attribution in the generated guide', () => {
    const { version, commit, docs_base, notice, notice_sha256, figures, galleries } = JSON.parse(
      readFileSync(resolve(sourceRoot, 'figures.json'), 'utf8')
    ) as {
      version: string; commit: string; docs_base: string; notice: string; notice_sha256: string;
      figures: Record<string, { gallery: string; asset: string; alt: string;
        sha256: string; bytes: number; width: number; height: number }>;
      galleries: Record<string, { title: string; author: string; code_path: string;
        code_download_url: string; code_sha256: string; page_sha256: string }>;
    };
    expect(version).toBe('0.14.2');
    expect(commit).toBe('8504e95f0160f61d1b617ca66f779646d2ee609e');
    expect(docs_base).toBe('https://imbalanced-learn.org/0.14/');
    expect(sha256(resolve(root, notice))).toBe(notice_sha256);
    expect(notice_sha256).toBe(sha256(resolve(sourceRoot, 'original/LICENSE')));
    expect(Object.keys(figures)).toHaveLength(23);
    expect(Object.keys(galleries)).toHaveLength(7);
    expect(Object.values(figures).filter(figure => figure.asset.startsWith('Assets/'))).toHaveLength(20);

    const publication = JSON.parse(readFileSync(resolve(sourceRoot, 'published/import-manifest.json'), 'utf8'));
    const pages = publication.pages.map(({ output }: { output: string }) => readFileSync(resolve(root, output), 'utf8')) as string[];
    for (const [filename, figure] of Object.entries(figures)) {
      expect(filename).toMatch(/^sphx_glr_plot_\w+_\d{3}\.png$/);
      const asset = resolve(root, figure.asset);
      const bytes = readFileSync(asset);
      expect(bytes.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
      expect([bytes.length, bytes.readUInt32BE(16), bytes.readUInt32BE(20)])
        .toEqual([figure.bytes, figure.width, figure.height]);
      expect(sha256(asset)).toBe(figure.sha256);
      const embed = `![[${figure.asset}|${figure.alt}]]`;
      const owners = pages.filter(page => page.includes(embed));
      expect(owners).toHaveLength(1);
      const [, caption] = owners[0].split(embed);
      const gallery = galleries[figure.gallery];
      expect(caption.split('\n').filter(Boolean)[0])
        .toContain(`${gallery.author}, [${gallery.title}](${docs_base}${figure.gallery})`);
      expect(caption.split('\n').filter(Boolean)[0]).toContain(`${docs_base}_images/${filename}`);
      expect(gallery.code_path).toBe(figure.gallery.replace('auto_examples/', 'examples/').replace('.html', '.py'));
      expect(gallery.code_download_url).toMatch(/^https:\/\/imbalanced-learn\.org\/0\.14\/_downloads\/[a-f0-9]{32}\/plot_\w+\.py$/);
      expect(gallery.code_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(gallery.page_sha256).toMatch(/^[a-f0-9]{64}$/);
    }
    expect(pages.join('\n')).not.toContain('https://imbalanced-learn.org/stable/_images/');
  });

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

  it('renders standalone RST prompts as code and explains labels clipped in the original PNG', () => {
    const source = readFileSync(resolve(sourceRoot, 'original/user-guide/model_selection.rst'), 'utf8');
    const markdown = readFileSync(resolve(sourceRoot, 'published/user-guide/model-selection.md'), 'utf8');
    const prompts = (text: string) => text.match(/^\s*(?:>>>|\.\.\.) .+$/gm)?.map(line => line.trim());
    expect(prompts(markdown)).toEqual(prompts(source));
    expect(prompts(markdown.replace(/^```python\n[\s\S]*?^```$/gm, ''))).toBeUndefined();
    expect(markdown).toContain('Top: InstanceHardnessCV. Bottom: StratifiedKFold.');
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
