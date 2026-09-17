import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadManifest, type PublicationSidebarItem } from '../adapter/manifest.js';

const root = resolve(import.meta.dirname, '../..');
const textbook = loadManifest(resolve(root, 'publishing/navigation.yml')).sections
  .find((section) => section.id === 'textbook')!;
const ru = readFileSync(resolve(root, '00 Учебник/_index.md'), 'utf8');
const en = readFileSync(resolve(root, 'en/00 Textbook/_index.md'), 'utf8');
const normalize = (path: string) => path.replace(/^02 Areas\/ML & DL\//, '').replace(/\.md$/, '');
const routes = new Map(textbook.pages.flatMap((page) => [page.source, page.sourceEn]
  .filter((source): source is string => Boolean(source))
  .map((source) => [normalize(source), page.route] as const)));

function chapterLinks(text: string): string[] {
  return [...text.matchAll(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g)]
    .map((match) => normalize(match[1]))
    .filter((source) => source.startsWith('00 Учебник/') || source.startsWith('en/00 Textbook/'))
    .map((source) => {
      expect(routes.has(source), `Unpublished textbook index target: ${source}`).toBe(true);
      return routes.get(source)!;
    });
}

function flatten(items: PublicationSidebarItem[]): Array<Extract<PublicationSidebarItem, { route: string }>> {
  return items.flatMap((item) => 'route' in item ? [item] : flatten(item.items));
}

describe('reader-facing textbook order', () => {
  it('does not leak source-file ordinals into chapter link labels', () => {
    for (const match of ru.matchAll(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g)) {
      const label = match[2] ?? match[1].split('/').at(-1)!;
      expect(label, match[1]).not.toMatch(/^\d+[a-z]?\s/);
    }
  });

  it('renders the systems sequence as separate Markdown list items', () => {
    const systems = ru.split('### VIII.')[1].split('### IX.')[0];
    expect(systems.match(/^- S\d+\. \[\[/gm)).toHaveLength(8);
    expect(systems).not.toMatch(/^S\d+\./m);
  });

  it('includes every canonical chapter in the Russian full route in sidebar order', () => {
    const fullRoute = ru.split('## Полный маршрут')[1].split('## Другие режимы')[0];
    expect(chapterLinks(fullRoute)).toEqual(textbook.pages.slice(1).map((page) => page.route));
  });

  it('assigns the same ordinal to each numbered Russian chapter and its sidebar link', () => {
    const ordinal = (text: string) => text.match(/^(S\d+|\d+(?:\.\d+)*)\.?\s/)?.[1];
    const sidebarNumbers = new Map(flatten(textbook.sidebar ?? [])
      .map((item) => [item.route, ordinal(item.label)]));
    for (const line of ru.split('\n')) {
      const clean = line.replace(/^\s*(?:-\s*)?/, '');
      const number = ordinal(clean);
      if (!number || !clean.includes('[[')) continue;
      const [route] = chapterLinks(clean);
      expect(sidebarNumbers.get(route), route).toBe(number);
    }
  });

  it('opens the first canonical chapter of each module from the English curriculum', () => {
    const curriculum = en.split('## Curriculum')[1].split('\n## ')[0];
    const modules = (textbook.sidebar ?? []).filter((item) => 'items' in item);
    expect(chapterLinks(curriculum)).toEqual(modules.map((module) => flatten([module])[0].route));
  });

  it('lists every authored English chapter once and does not label a fallback as available', () => {
    const available = en.split(/^## /m).filter((section) => /^[^\n]*available in English\n/.test(section));
    expect(chapterLinks(available.join('\n'))).toEqual(
      textbook.pages.slice(1).filter((page) => page.sourceEn).map((page) => page.route)
    );
  });
});
