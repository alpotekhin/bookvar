import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { normalizeObsidianHeading, parseMarkdownHeadings } from '../adapter/build.js';
import { readingPaths } from '../adapter/reading-paths.js';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
const book = manifest.sections.find(s => s.id === 'textbook')!;
const sources = manifest.sections.find(s => s.id === 'sources')!;
const hubs = [
  ['05 Источники/Courses/Stanford CS336 Spring 2026/_index.md', 9],
  ['05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index.md', 8],
] as const;

describe('university courses as source libraries', () => {
  it('keeps both hubs in sources, outside every textbook reading path', () => {
    const routes = readingPaths(book).flatMap(p => p.routes);
    for (const [source] of hubs) {
      expect(parse(read(source).split('---')[1]).type).toBe('source-note');
      const page = sources.pages.find(p => p.source === source)!;
      expect(page, source).toBeDefined();
      expect(page.route).toMatch(/^sources\/courses\//);
      expect(routes).not.toContain(page.route);
      expect(book.pages.some(p => p.source === source)).toBe(false);
    }
  });

  it('resolves each thematic table to a canonical chapter and an existing heading', () => {
    for (const [hub, rowCount] of hubs) {
      const table = read(hub).split('## Где читать тему\n')[1].split(/^## |^<a id=/m)[0];
      const rows = table.split('\n').filter(line => line.startsWith('|')).slice(2);
      expect(rows, hub).toHaveLength(rowCount);
      for (const row of rows) {
        const targets = [...row.matchAll(/\[\[(00 Учебник\/[^\]|]+)(?:\|[^\]]+)?\]\]/g)];
        expect(targets.length, row).toBeGreaterThan(0);
        for (const match of targets) {
          const [path, anchor] = match[1].split('#');
          const source = path + '.md';
          expect(book.pages.some(p => p.source === source), source).toBe(true);
          const body = read(source);
          expect(parse(body.split('---')[1]).status).not.toBe('redirect');
          if (anchor) expect(parseMarkdownHeadings(body, true).some(h =>
            h.explicitAnchor === anchor || h.normalized === normalizeObsidianHeading(anchor)), match[1]).toBe(true);
        }
      }
    }
  });

  it('preserves complete lecture catalogs without claiming the whole courses are integrated', () => {
    const stanford = read(hubs[0][0]), berkeley = read(hubs[1][0]);
    for (let i = 1; i <= 19; i++) expect(stanford).toContain('id="lecture-' + String(i).padStart(2, '0') + '"');
    for (let i = 1; i <= 12; i++) expect(berkeley).toContain('id="meeting-' + String(i).padStart(2, '0') + '"');
    expect(stanford).toContain('inventory complete; editorial integration active');
    expect(berkeley).toContain('Само наличие связанного раздела ещё не подтверждает');
    expect(berkeley).toContain('DO NOT DISTRIBUTE.');
    expect(berkeley).not.toContain('](Lectures/meeting-06-slides.pdf)');
  });
});
