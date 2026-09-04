import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { createRouteRegistry } from '../adapter/routes.js';

const root = resolve(import.meta.dirname, '../..');

describe('route registry', () => {
  it('resolves a full Obsidian path and an alias', () => {
    const registry = createRouteRegistry([{
      sourcePath: '00 Учебник/05 Attention/01 Attention.md',
      route: '/textbook/attention/',
      title: 'Attention'
    }]);
    expect(registry.routeForWikiTarget('00 Учебник/05 Attention/01 Attention')).toBe('/textbook/attention/');
    expect(registry.routeForWikiTarget('01 Attention')).toBe('/textbook/attention/');
    expect(registry.routeForWikiTarget(
      '02 Areas/ML & DL/00 Учебник/05 Attention/01 Attention'
    )).toBe('/textbook/attention/');
  });

  it('throws on an ambiguous basename', () => {
    expect(() => createRouteRegistry([
      { sourcePath: 'a/Index.md', route: '/a/', title: 'A' },
      { sourcePath: 'b/Index.md', route: '/b/', title: 'B' }
    ])).toThrow(/ambiguous/i);
  });

  it('can preserve full-path routes while omitting ambiguous basename aliases', () => {
    const registry = createRouteRegistry([
      { sourcePath: 'a/Index.md', route: '/a/', title: 'A' },
      { sourcePath: 'b/Index.md', route: '/b/', title: 'B' }
    ], { allowAmbiguousBasenames: true });

    expect(registry.routeForWikiTarget('a/Index')).toBe('/a/');
    expect(registry.routeForWikiTarget('b/Index')).toBe('/b/');
    expect(registry.routeForWikiTarget('Index')).toBeUndefined();
  });

  it('keeps aliases when localized source paths point to the same route', () => {
    const registry = createRouteRegistry([
      { sourcePath: '00 Учебник/14 Inference/55 Cache.md', route: '/en/textbook/cache/', title: 'Cache' },
      { sourcePath: 'en/00 Textbook/14 Inference/55 Cache.md', route: '/en/textbook/cache/', title: 'Cache' }
    ]);

    expect(registry.routeForWikiTarget('00 Учебник/14 Inference/55 Cache')).toBe('/en/textbook/cache/');
    expect(registry.routeForWikiTarget('en/00 Textbook/14 Inference/55 Cache')).toBe('/en/textbook/cache/');
    expect(registry.routeForWikiTarget('55 Cache')).toBe('/en/textbook/cache/');
  });

  it('publishes the pinned course hubs and every source-native Berkeley reading below source routes', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const entries = manifest.sections.flatMap((section) => section.pages);
    const registry = createRouteRegistry(entries.map((entry) => ({
      sourcePath: entry.source,
      route: entry.route.endsWith('/index')
        ? `/${entry.route.slice(0, -'/index'.length)}/`
        : `/${entry.route}/`,
      title: entry.source
    })), { allowAmbiguousBasenames: true });

    expect(registry.routeForWikiTarget(
      '05 Источники/Courses/Stanford CS336 Spring 2026/_index'
    )).toBe('/sources/courses/stanford-cs336-spring-2026/');
    expect(registry.routeForWikiTarget(
      '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index'
    )).toBe('/sources/courses/berkeley-advanced-llm-agents-spring-2025/');

    const readingsDir = resolve(
      root,
      '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Readings'
    );
    const readingSources = readdirSync(readingsDir)
      .filter((name) => name.endsWith('.md'))
      .map((name) => `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Readings/${name}`);
    expect(readingSources).toHaveLength(37);
    for (const source of readingSources) {
      expect(registry.routeForWikiTarget(source.slice(0, -'.md'.length)), source)
        .toMatch(/^\/sources\/courses\/berkeley-advanced-llm-agents-spring-2025\/readings\//);
    }
  });
});
