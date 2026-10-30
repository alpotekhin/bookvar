import type { PublicationSection, PublicationSidebarItem } from './manifest.js';

export interface ReadingPath {
  id: string;
  label: string;
  labelEn?: string;
  routes: string[];
}

// Source filenames and titles may retain old flat chapter ordinals. The site
// has independent reading paths, so those ordinals are not reader positions.
// Limit this presentation rule to textbook pages in either locale, not source titles
// such as course lecture numbers or versioned model names.
export function readerTitle(source: string, title: string): string {
  return (source.startsWith('00 Учебник/') || source.startsWith('en/00 Textbook/'))
    ? title.replace(/^\d+(?:\.\d+)*\.\s+/, '') : title;
}

export function readingPaths(section: PublicationSection): ReadingPath[] {
  const paths: ReadingPath[] = [];
  const flatten = (items: PublicationSidebarItem[]): string[] => items.flatMap(item => {
    if ('route' in item) return [item.route];
    if (item.readingPath) throw new Error(`Nested reading path: ${item.readingPath}`);
    return flatten(item.items);
  });
  function visit(items: PublicationSidebarItem[]) {
    for (const item of items) {
      if ('route' in item) continue;
      if (item.readingPath) paths.push({
        id: item.readingPath, label: item.label, labelEn: item.labelEn, routes: flatten(item.items),
      });
      else visit(item.items);
    }
  }
  visit(section.sidebar ?? []);
  if (!paths.length) return [];
  if (new Set(paths.map(path => path.id)).size !== paths.length) throw new Error('Duplicate reading path ID');
  const routes = paths.flatMap(path => path.routes);
  if (new Set(routes).size !== routes.length) throw new Error('A chapter belongs to multiple reading paths');
  const assigned = new Set(routes);
  const missing = section.pages.filter(page => !['textbook/index', 'textbook/how-to-use'].includes(page.route)
    && !assigned.has(page.route));
  if (missing.length) throw new Error(`Chapters without a reading path: ${missing.map(page => page.route).join(', ')}`);
  return paths;
}

export function readingNeighbours(paths: ReadingPath[], route: string) {
  const path = paths.find(path => path.routes.includes(route));
  if (!path) return undefined;
  const i = path.routes.indexOf(route);
  return { path, prev: path.routes[i - 1], next: path.routes[i + 1] };
}
