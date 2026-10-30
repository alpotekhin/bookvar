import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';

import { loadManifest, type PublicationSidebarItem } from '../adapter/manifest.js';
import { wikiHeadingSlug } from '../adapter/links.js';

const root = resolve(import.meta.dirname, '../..');

describe('publication workflow policy', () => {
  it('provides the 404 content entry requested by Starlight without indexing it as a chapter', () => {
    const entry = resolve(root, 'site/src/content/docs/404.md');
    expect(existsSync(entry), 'Starlight requests the docs/404 entry during every build').toBe(true);
    const frontmatter = YAML.parse(readFileSync(entry, 'utf8').split('---')[1]);
    expect(frontmatter).toMatchObject({ template: 'splash', editUrl: false, pagefind: false });
  });

  it('separates unit tests from built-output tests and runs them around the build', () => {
    const publishing = JSON.parse(readFileSync(resolve(root, 'publishing/package.json'), 'utf8'));
    const workflow = readFileSync(resolve(root, '.github/workflows/verify.yml'), 'utf8');

    expect(publishing.scripts['test:unit']).toBe('vitest run --exclude tests/site-output.test.ts');
    expect(publishing.scripts['test:publication']).toBe('vitest run --config vitest.publication.config.ts');
    expect(publishing.scripts['test:output']).toBe('vitest run tests/site-output.test.ts');
    expect(workflow.indexOf('pnpm --dir publishing test:publication')).toBeGreaterThan(0);
    expect(workflow.indexOf('pnpm --dir publishing test:publication'))
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

  it('gives original seq2seq SVGs responsive reading sizes without cropping them', () => {
    const css = readFileSync(resolve(root, 'site/src/styles/custom.css'), 'utf8');
    expect(css).toMatch(/img\[src\*='\/curated\/seq2seq-2026-10-21\/'\][^}]*width:\s*min\(100%, 38rem\)/s);
    expect(css).toMatch(/img\[src\$='\/seq2seq-2026-10-21\/s2s-prob2.svg'\][^}]*width:\s*min\(100%, 24rem\)/s);
    expect(css).toMatch(/img\[src\$='\/d2l-recurrent\/seq2seq-attention.svg'\][^}]*width:\s*min\(100%, 38rem\)/s);
    expect(css).toMatch(/\.sl-markdown-content:has\(img\[src\*='\/curated\/seq2seq-2026-10-21\/'\]\) table code\s*\{\s*white-space:\s*nowrap;/);
  });

  it('gives self-attention source diagrams a responsive reading size', () => {
    const css = readFileSync(resolve(root, 'site/src/styles/custom.css'), 'utf8');
    expect(css).toMatch(/img\[src\*='\/curated\/self-attention-2026-10-21\/'\]\s*\{\s*width:\s*min\(100%, 38rem\);/);
  });

  it('gives recurrent-attention source diagrams a responsive reading size', () => {
    const css = readFileSync(resolve(root, 'site/src/styles/custom.css'), 'utf8');
    expect(css).toMatch(/img\[src\*='\/curated\/attention-translation-2026-10-21\/'\]\s*\{\s*width:\s*min\(100%, 38rem\);/);
  });

  it('keeps original courses in the library rather than parallel textbook navigation', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const sources = manifest.sections.find((section) => section.id === 'sources');
    const textbook = manifest.sections.find((section) => section.id === 'textbook');
    const flatten = (items: PublicationSidebarItem[]): string[] => items.flatMap(item =>
      'route' in item ? [item.route] : flatten(item.items));
    const libraryRoutes = flatten(sources?.sidebar ?? []);
    const hubRoutes = [
      'sources/courses/stanford-cs336-spring-2026/index',
      'sources/courses/berkeley-advanced-llm-agents-spring-2025/index'
    ];

    expect(libraryRoutes).toContain('sources/kursy');
    expect(libraryRoutes.filter(route => hubRoutes.includes(route))).toEqual([]);
    expect(sources?.pages.filter(page => hubRoutes.includes(page.route))).toHaveLength(2);
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

  it('links the library to the complete course catalogue and retains both pinned course hubs there', () => {
    const sourceIndex = readFileSync(resolve(root, '05 Источники/_index.md'), 'utf8');
    const catalogue = readFileSync(resolve(root, '05 Источники/Курсы.md'), 'utf8');
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
      expect(catalogue, hub).toContain(hub.slice(0, -'.md'.length));
    }
    expect(sourceIndex).toContain('[[05 Источники/Курсы|');
  });

  it('offers one textbook, a topic reference and a library; course hubs put reading before audit details', () => {
    const homepage = readFileSync(resolve(root, 'site/src/content/docs/index.md'), 'utf8');
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const publishedSources = new Set(manifest.sections.flatMap((section) => section.pages.map((page) => page.source)));
    const courses = [
      ['Stanford CS336', 'stanford-cs336-spring-2026', 'Stanford CS336 Spring 2026'],
      ['Berkeley Advanced LLM Agents', 'berkeley-advanced-llm-agents-spring-2025', 'Berkeley Advanced LLM Agents Spring 2025']
    ];
    expect(homepage).toContain('href="./textbook/"');
    expect(homepage).toContain('href="./reference/"');
    expect(homepage).toContain('href="./sources/"');
    expect(homepage).not.toContain('id="university-courses"');
    for (const [name, route, folder] of courses) {
      expect(homepage).not.toContain(`href="./sources/courses/${route}/"`);
      expect(homepage).not.toContain(`href="./sources/courses/${route}/index/"`);
      const hub = readFileSync(resolve(root, `05 Источники/Courses/${folder}/_index.md`), 'utf8');
      expect(hub.indexOf('## Где читать тему')).toBeGreaterThan(0);
      expect(hub.indexOf('<details>')).toBeGreaterThan(hub.indexOf('## Где читать тему'));
      const readerText = hub.split('<details>')[0];
      expect(readerText).not.toMatch(/dependency closure|semantic extraction|Visual ledger|pinned revisions|source layer/i);
      for (const match of readerText.matchAll(/\[\[([^\]|#]+)(?:[^\]]*)\]\]/g)) {
        expect(publishedSources.has(`${match[1]}.md`), `${name}: ${match[1]}`).toBe(true);
      }
    }
    const berkeley = readFileSync(resolve(root, `05 Источники/Courses/${courses[1][2]}/_index.md`), 'utf8');
    expect(berkeley).toContain('NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.');
    expect(berkeley).toContain('не официальные задания Berkeley');
    expect(berkeley).not.toContain('](Lectures/meeting-06-slides.pdf)');
    expect([...berkeley.matchAll(/\]\(Readings\/meeting-\d+-reading-\d+\.md\)/g)]).toHaveLength(37);
    expect([...berkeley.matchAll(/\]\(https:\/\/www.youtube.com\/watch\?v=/g)]).toHaveLength(12);
    for (let number = 1; number <= 12; number += 1) {
      expect(berkeley).toContain(`<a id="meeting-${String(number).padStart(2, '0')}"></a>`);
    }
  });

  it('keeps reader-facing Stanford coverage counts consistent with both current ledgers', () => {
    const courseRoot = resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026');
    const countLedger = (name: string) => {
      const rows = JSON.parse(readFileSync(resolve(courseRoot, name), 'utf8')).rows as
        Array<{ disposition: string; source_object?: string }>;
      const count = (status: string) => rows.filter(row => row.disposition === status).length;
      return { total: rows.length, integrated: count('integrated'),
        covered: count('covered-existing'), contract: count('contract-only'),
        sourceOnly: count('source-only'), excluded: count('excluded'),
        assignments: rows.filter(row => row.disposition === 'integrated' &&
          row.source_object?.startsWith('assignment')).length };
    };
    const coverage = countLedger('coverage.yml');
    const visuals = countLedger('visuals.yml');
    const ru = readFileSync(resolve(root, '05 Источники/Курсы.md'), 'utf8');
    const en = readFileSync(resolve(root, 'en/05 Sources/Courses.md'), 'utf8');
    const numbers = (text: string, pattern: RegExp) => {
      const match = text.match(pattern);
      expect(match, pattern.source).not.toBeNull();
      return match!.slice(1).map(Number);
    };
    const semanticCounts = [coverage.total, coverage.integrated, coverage.covered,
      coverage.contract, coverage.sourceOnly, coverage.excluded];
    const visualCounts = [visuals.total, visuals.integrated, visuals.covered,
      visuals.sourceOnly, visuals.excluded];
    expect(numbers(ru, /Из (\d+) записей покрытия (\d+) имеют статус `integrated`, (\d+) — `covered-existing`,\s*(\d+) — `contract-only`, (\d+) — `source-only`, (\d+) — `excluded`/))
      .toEqual(semanticCounts);
    expect(numbers(en, /coverage ledger contains (\d+) semantic units: (\d+) integrated, (\d+) already\s*covered, (\d+) assignment-contract-only, (\d+) source-only, and (\d+) excluded/))
      .toEqual(semanticCounts);
    expect(numbers(ru, /реестр содержит (\d+) визуальных записей: (\d+) `integrated`,\s*(\d+) `covered-existing`, (\d+) `source-only`, (\d+) `excluded`/))
      .toEqual(visualCounts);
    expect(numbers(en, /visual ledger has (\d+) records: (\d+) integrated,\s*(\d+) already covered, (\d+) source-only, and (\d+) excluded/))
      .toEqual(visualCounts);
    expect(numbers(en, /Of the (\d+) integrated units, (\d+) come from assignment material and (\d+) from/))
      .toEqual([coverage.integrated, coverage.assignments,
        coverage.integrated - coverage.assignments]);
  });

  it('uses stable Stanford lecture anchors in both course overviews and preserves old heading links', () => {
    const courseRoot = resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026');
    const hub = readFileSync(resolve(courseRoot, '_index.md'), 'utf8');
    for (const source of ['05 Источники/Курсы.md', 'en/05 Sources/Courses.md']) {
      const overview = readFileSync(resolve(root, source), 'utf8');
      const links = [...overview.matchAll(/\[\[05 Источники\/Courses\/Stanford CS336 Spring 2026\/_index#([^|\]]+)/g)];
      expect(links, source).toHaveLength(9);
      for (const [, anchor] of links) {
        expect(anchor, source).toMatch(/^lecture-\d{2}$/);
        expect(hub, `${source}#${anchor}`).toContain(`<a id="${anchor}"></a>`);
      }
    }
    const manifest = JSON.parse(readFileSync(resolve(courseRoot, 'source-manifest.yml'), 'utf8')) as {
      objects: Array<{ id: string; title: string }>;
    };
    for (const object of manifest.objects) {
      const previousHeading = `${object.id}: ${object.title}`;
      expect(hub, previousHeading).toContain(`<a id="${wikiHeadingSlug(previousHeading)}"></a>`);
    }
  });
});
