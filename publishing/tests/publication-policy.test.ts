import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';

import { loadManifest, type PublicationSidebarItem } from '../adapter/manifest.js';

const root = resolve(import.meta.dirname, '../..');

describe('publication workflow policy', () => {
  it('separates unit tests from built-output tests and runs them around the build', () => {
    const publishing = JSON.parse(readFileSync(resolve(root, 'publishing/package.json'), 'utf8'));
    const workflow = readFileSync(resolve(root, '.github/workflows/verify.yml'), 'utf8');

    expect(publishing.scripts['test:unit']).toBe('vitest run --exclude tests/site-output.test.ts');
    expect(publishing.scripts['test:output']).toBe('vitest run tests/site-output.test.ts');
    expect(workflow.indexOf('pnpm --dir publishing test:unit'))
      .toBeLessThan(workflow.indexOf('pnpm --dir site build'));
    expect(workflow.indexOf('pnpm --dir site build'))
      .toBeLessThan(workflow.indexOf('pnpm --dir publishing test:output'));
  });

  it('does not track a second navigation source and generates the ignored sidebar before consumers', () => {
    const ignore = readFileSync(resolve(root, '.gitignore'), 'utf8');
    const site = JSON.parse(readFileSync(resolve(root, 'site/package.json'), 'utf8'));

    expect(ignore).toContain('site/generated-sidebar.mjs');
    expect(site.scripts.precheck).toBe('pnpm content:build');
    expect(site.scripts.prebuild).toBe('pnpm content:build');
  });

  it('pins every direct dependency to an exact version', () => {
    for (const manifest of ['publishing/package.json', 'site/package.json']) {
      const pkg = JSON.parse(readFileSync(resolve(root, manifest), 'utf8'));
      for (const [name, spec] of Object.entries({ ...pkg.dependencies, ...pkg.devDependencies })) {
        expect(spec, `${manifest}: ${name}`).toMatch(/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
      }
    }
  });

  it('keeps rich article media and attribution inside the reading column', () => {
    const css = readFileSync(resolve(root, 'site/src/styles/custom.css'), 'utf8');

    expect(css).toMatch(/\.sl-markdown-content img\s*\{[^}]*max-width:\s*100%[^}]*height:\s*auto/s);
    expect(css).toMatch(/\.sl-markdown-content \.katex-display[\s\S]*?overflow-x:\s*auto/);
    expect(css).toMatch(/\.sl-markdown-content table[\s\S]*?overflow-x:\s*auto/);
    expect(css).toMatch(/\.sl-markdown-content pre[\s\S]*?overflow-x:\s*auto/);
    expect(css).toMatch(/\.sl-markdown-content figure\s*\{[^}]*max-width:\s*100%/s);
    expect(css).toMatch(/\.sl-markdown-content figcaption\s*\{[^}]*overflow-wrap:\s*anywhere/s);
    expect(css).toMatch(/\.source-attribution\s*\{[^}]*max-width:\s*100%[^}]*overflow-wrap:\s*anywhere/s);
  });

  it('places pinned course hubs only under Источники → Курсы and preserves textbook concept navigation', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const sources = manifest.sections.find((section) => section.id === 'sources');
    const textbook = manifest.sections.find((section) => section.id === 'textbook');
    const courses = sources?.sidebar?.find((item) => item.label === 'Курсы');
    const courseItems = courses && 'items' in courses
      ? courses.items.filter((item): item is Extract<PublicationSidebarItem, { route: string }> => 'route' in item)
      : [];
    const hubRoutes = [
      'sources/courses/stanford-cs336-spring-2026/index',
      'sources/courses/berkeley-advanced-llm-agents-spring-2025/index'
    ];

    expect(courseItems.map((item) => item.route)).toEqual([
      'sources/kursy',
      ...hubRoutes
    ]);
    expect(textbook?.pages.some((page) => hubRoutes.includes(page.route))).toBe(false);
    expect(textbook?.sidebar?.some((item) => 'route' in item && hubRoutes.includes(item.route))).toBe(false);
    expect(textbook?.pages.some((page) => page.route === 'textbook/transformer/self-attention')).toBe(true);
  });

  it('keeps the four legacy Stanford notes on disk but out of publication navigation', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const publishedSources = new Set(manifest.sections.flatMap((section) => section.pages.map((page) => page.source)));
    const legacySources = [
      'Courses/Stanford CS336/CS336 — Inference.md',
      'Courses/Stanford CS336/CS336 — Scaling Laws.md',
      'Courses/Stanford CS336/CS336 — Tokenization.md',
      'Courses/Stanford CS336/CS336 — Mixture of Experts.md'
    ];

    for (const source of legacySources) {
      const content = readFileSync(resolve(root, source), 'utf8');
      expect(content, source).toContain('status: legacy');
      expect(content, source).toContain('source_only: true');
      expect(content, source).toContain('robots: noindex');
      expect(content, source).toContain('search_exclude: true');
      expect(content, source).toContain('canonical_target:');
      expect(publishedSources.has(source), source).toBe(false);
    }
  });

  it('links the source library index to both pinned course hubs', () => {
    const sourceIndex = readFileSync(resolve(root, '05 Источники/_index.md'), 'utf8');
    const navigation = YAML.parse(readFileSync(resolve(root, 'publishing/navigation.yml'), 'utf8')) as {
      sections: Array<{ pages: Array<{ source: string; route: string }> }>;
    };
    const routes = new Map(navigation.sections.flatMap((section) =>
      section.pages.map((page) => [page.source, page.route] as const)
    ));
    const hubs = [
      '05 Источники/Courses/Stanford CS336 Spring 2026/_index.md',
      '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index.md'
    ];

    for (const hub of hubs) {
      expect(routes.get(hub), hub).toBeDefined();
      expect(sourceIndex, hub).toContain(`[[02 Areas/ML & DL/${hub.slice(0, -'.md'.length)}`);
    }
  });
});
