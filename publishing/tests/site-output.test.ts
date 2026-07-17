import { access, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import YAML from 'yaml';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const distDir = join(rootDir, 'site', 'dist');
const pagefindDir = join(distDir, 'pagefind');

interface PagefindResult {
  data(): Promise<{ meta: { title: string }; url: string }>;
}

interface PagefindApi {
  destroy(): Promise<void>;
  options(options: Record<string, unknown>): Promise<void>;
  search(term: string): Promise<{ results: PagefindResult[] }>;
}

let pagefind: PagefindApi;
let originalFetch: typeof globalThis.fetch;

beforeAll(async () => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
    if (!url.startsWith('file:')) return originalFetch(input, init);
    const contents = await readFile(fileURLToPath(url));
    return new Response(contents as unknown as BodyInit, {
      status: 200,
      headers: { 'content-type': url.endsWith('.pagefind') ? 'application/wasm' : 'application/octet-stream' }
    });
  };

  const basePath = pathToFileURL(`${pagefindDir}/`).href;
  pagefind = await import(`${basePath}pagefind.js`) as PagefindApi;
  await pagefind.options({ basePath, language: 'ru' });
});

afterAll(async () => {
  await pagefind?.destroy();
  globalThis.fetch = originalFetch;
});

describe('static handbook output', () => {
  it('does not publish local file edit links or workstation paths', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );
    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).not.toMatch(/file:\/\/|\/Users\//);
    }
  });

  it('renders formulas and Mermaid without losing the source diagram', async () => {
    const formula = await readFile(
      join(distDir, 'textbook', 'foundations', 'backpropagation', 'index.html'),
      'utf8'
    );
    const diagram = await readFile(
      join(distDir, 'textbook', 'rag', 'module-map', 'index.html'),
      'utf8'
    );

    expect(formula).toContain('class="katex');
    expect(diagram).toContain('class="mermaid"');
    expect(diagram).toContain('flowchart LR');
  });

  it('emits every manifest route as Russian HTML', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );

    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).toMatch(/^<!DOCTYPE html><html lang="ru"/);
    }
  });

  it('includes previous and next links on an interior handbook page', async () => {
    const html = await readFile(
      join(distDir, 'textbook', 'transformer', 'self-attention', 'index.html'),
      'utf8'
    );
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
  });

  it('writes a Russian Pagefind index', async () => {
    await expect(access(join(pagefindDir, 'pagefind.js'))).resolves.toBeUndefined();
    const entry = await readFile(join(pagefindDir, 'pagefind-entry.json'), 'utf8');
    expect(entry).toContain('ru');
  });

  it('finds the attention chapter using its indexed Transformer terminology', async () => {
    const search = await pagefind.search('Transformer');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/transformer/self-attention/')))
      .toBe(true);
  });

  it('stems Russian inflections in the generated index', async () => {
    const search = await pagefind.search('нейронов');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/foundations/backpropagation/')))
      .toBe(true);
  });
});
