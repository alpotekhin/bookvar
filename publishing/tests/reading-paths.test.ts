import { describe, expect, it } from 'vitest';
import { readerTitle, readingNeighbours, readingPaths } from '../adapter/reading-paths.js';
import type { PublicationSection } from '../adapter/manifest.js';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import matter from 'gray-matter';
import { loadManifest } from '../adapter/manifest.js';
import { beforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';
import { beforeConfidentLearningAddition } from '../tools/confident-learning-baseline.mjs';
import { beforeSelectionGapsAddition } from '../tools/selection-gaps-baseline.mjs';
import { beforeServingOperations } from '../tools/serving-operations-baseline.mjs';
import { beforeServingNode } from '../tools/serving-node-baseline.mjs';
import { beforeCs336Residuals } from '../tools/cs336-residuals-baseline.mjs';
import { beforeHardwareSourceCompletion } from '../tools/hardware-source-baseline.mjs';
import { beforeSystemsCompletion } from '../tools/systems-completion-baseline.mjs';
import { beforeTensorFactorization } from '../tools/tensor-factorization-baseline.mjs';

const section: PublicationSection = {
  id: 'textbook', title: 'Book',
  pages: ['textbook/index', 'textbook/how-to-use', 'intro', 'loss', 'active', 'coreset']
    .map(route => ({ source: `${route}.md`, route })),
  sidebar: [
    { label: 'Core', readingPath: 'main', items: [
      { label: 'Basics', items: [{ label: 'Intro', route: 'intro' }, { label: 'Loss', route: 'loss' }] },
    ] },
    { label: 'Modules', items: [{ label: 'Data', readingPath: 'data', items: [
      { label: 'Active', route: 'active' }, { label: 'Coreset', route: 'coreset' },
    ] }] },
  ],
};

describe('independent reader paths', () => {
  it('removes obsolete reader ordinals without rewriting canonical or source titles', () => {
    expect(readerTitle('00 Учебник/chapter.md', '64.7. Diffusion and flow')).toBe('Diffusion and flow');
    expect(readerTitle('00 Учебник/chapter.md', '70. Формальные доказательства')).toBe('Формальные доказательства');
    for (const title of ['GPT-3.5', '3D representations', '3.5 billion parameters', '100 вопросов'])
      expect(readerTitle('00 Учебник/chapter.md', title)).toBe(title);
    expect(readerTitle('en/00 Textbook/chapter.md', '64.7. Diffusion and flow')).toBe('Diffusion and flow');
    for (const source of ['05 Источники/lecture.md', 'en/05 Sources/lecture.md'])
      expect(readerTitle(source, '1. Introduction')).toBe('1. Introduction');
  });
  it('keeps explicit in-text navigation on the same path as the site buttons', () => {
    const root = resolve(import.meta.dirname, '../..');
    const book = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.find(s => s.id === 'textbook')!;
    const target = (link: string) => link.replace(/^\[\[/, '').split(/[|#]/)[0]
      .replace(/\]\]$/, '').replace(/^02 Areas\/ML & DL\//, '').replace(/\.md$/, '');
    const source = (route: string) => book.pages.find(p => p.route === route)!.source.replace(/\.md$/, '');
    let checked = 0;
    for (const path of readingPaths(book)) for (const [i, route] of path.routes.entries()) {
      const { content } = matter(readFileSync(resolve(root, source(route) + '.md'), 'utf8'));
      const plain = content.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '');
      const links = [...plain.matchAll(/(?:^| · )(?:\*\*)?(Назад|Дальше|Далее|Предыдущая(?: глава| страница)?|Следующая(?: глава)?|К модулям):(?:\*\*)?\s*(\[\[[^\]]+\]\])/gm)]
        .map(m => ({ label: m[1], link: m[2] }));
      for (const m of plain.matchAll(/^←\s*(\[\[[^\]]+\]\])/gm)) links.push({ label: 'Назад', link: m[1] });
      for (const { label, link } of links) {
        const back = /Назад|Предыдущая/.test(label);
        const neighbour = label === 'К модулям' ? 'textbook/index' : back
          ? path.routes[i - 1] ?? (path.id === 'main' ? 'textbook/how-to-use' : 'textbook/index')
          : path.routes[i + 1] ?? 'textbook/index';
        const expected = source(neighbour);
        const actual = target(link);
        expect.soft(actual.includes('/') ? actual : expected.split('/').at(-1) === actual ? expected : actual,
          `${route}: ${label}`).toBe(expected);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(200);
  });
  it('matches canonical previous/next metadata to the current reader paths', () => {
    const root = resolve(import.meta.dirname, '../..');
    const book = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.find(s => s.id === 'textbook')!;
    const source = (route: string) => book.pages.find(p => p.route === route)!.source.replace(/\.md$/, '');
    const target = (link: string) => link.replace(/^\[\[/, '').split(/[|#]/)[0]
      .replace(/\]\]$/, '').replace(/^02 Areas\/ML & DL\//, '').replace(/\.md$/, '');
    const paths = readingPaths(book);
    expect(paths.find(p => p.id === 'main')!.routes).toHaveLength(64);
    expect(paths).toHaveLength(14);
    for (const path of paths) for (const [i, route] of path.routes.entries()) {
      const page = book.pages.find(p => p.route === route)!;
      const { data } = matter(readFileSync(resolve(root, page.source), 'utf8'));
      const prev = i ? source(path.routes[i - 1]) : path.id === 'main'
        ? source('textbook/how-to-use') : source('textbook/index');
      const next = i + 1 < path.routes.length ? source(path.routes[i + 1]) : source('textbook/index');
      for (const [key, expected] of [['previous', prev], ['prev', prev], ['next', next]]) {
        if (data[key] == null) continue;
        for (const link of Array.isArray(data[key]) ? data[key] : [data[key]]) {
          expect(target(link), `${route}: ${key}`).toBe(expected);
        }
      }
    }
  });
  it('does not hide explanation changes behind historical navigation restoration', () => {
    const root = resolve(import.meta.dirname, '../..');
    const report = JSON.parse(readFileSync(resolve(root, 'docs/audit/2026-10-29-reader-route-navigation.json'), 'utf8'));
    const file = report.files[0];
    const current = readFileSync(resolve(root, file.source), 'utf8');
    const restored = beforeReaderRouteNavigation(current, file.source, root);
    const changed = beforeReaderRouteNavigation(current + '\nUnreviewed explanation.\n', file.source, root);
    expect(createHash('sha256').update(restored).digest('hex')).toBe(file.before);
    expect(createHash('sha256').update(changed).digest('hex')).not.toBe(file.before);
    expect(changed).toContain('Unreviewed explanation.');
  });
  it('keeps the main sequence and each specialist module separate', () => {
    const paths = readingPaths(section);
    expect(paths.map(path => path.routes)).toEqual([['intro', 'loss'], ['active', 'coreset']]);
    expect(readingNeighbours(paths, 'loss')?.next).toBeUndefined();
    expect(readingNeighbours(paths, 'active')?.prev).toBeUndefined();
    expect(readingNeighbours(paths, 'active')?.next).toBe('coreset');
    expect(readingNeighbours(paths, 'textbook/index')).toBeUndefined();
  });
  it('rejects losing chapters when reorganizing the contents', () => {
    expect(() => readingPaths({ ...section, pages: [...section.pages, { source: 'forgot.md', route: 'forgot' }] }))
      .toThrow('Chapters without a reading path: forgot');
  });
  it('rejects overlapping and nested paths', () => {
    expect(() => readingPaths({ ...section, sidebar: [...section.sidebar!, {
      label: 'Duplicate', readingPath: 'dup', items: [{ label: 'Intro', route: 'intro' }],
    }] })).toThrow('multiple reading paths');
    expect(() => readingPaths({ ...section, sidebar: [{
      label: 'Parent', readingPath: 'parent', items: section.sidebar!,
    }] })).toThrow('Nested reading path');
  });
  it('changes only recorded navigation fragments in canonical chapters', () => {
    const root = resolve(import.meta.dirname, '../..');
    const report = JSON.parse(readFileSync(resolve(root, 'docs/audit/2026-10-29-reader-route-navigation.json'), 'utf8'));
    const sha = (text: string) => createHash('sha256').update(text).digest('hex');
    for (const file of report.files) {
      const preceding = beforeTopicConsolidation(readFileSync(resolve(root, file.source), 'utf8'), file.source, root);
      let text = beforeHardwareSourceCompletion(preceding, file.source, root);
      text = beforeCs336Residuals(text, file.source, root);
      text = beforeServingNode(text, file.source, root);
      text = beforeServingOperations(text, file.source, root);
      text = beforeSystemsCompletion(text, file.source, root);
      text = beforeTensorFactorization(text, file.source, root);
      text = beforeConfidentLearningAddition(text, file.source, root);
      text = beforeSelectionGapsAddition(text, file.source, root);
      expect(sha(text), file.source).toBe(file.after);
      for (const change of [...file.changes].reverse()) {
        expect(text, file.source).toContain(change.new);
        text = text.replace(change.new, change.old);
      }
      expect(sha(text), `Unrecorded body change: ${file.source}`).toBe(file.before);
    }
  });
});
