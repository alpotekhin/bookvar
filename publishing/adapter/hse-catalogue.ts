const archiveRoot = '05 Источники/Courses/HSE ML course/';
const routeRoot = 'sources/courses/hse-ml-course/';
const upstream = 'https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/';

/** A public catalogue entry uses the Obsidian path as a link identity, not as an input file. */
export function hseCatalogueEntry(source: string, route: string): { title: string; href: string } | undefined {
  const path = source.replaceAll('\\', '/');
  const archive = path.startsWith(archiveRoot);
  const catalogue = route.startsWith(routeRoot) && route !== routeRoot + 'index';
  if (!archive && !catalogue) return;
  if (!archive || !catalogue) throw new Error('HSE archive entries must use the HSE catalogue route: ' + source);

  const original = path.slice(archiveRoot.length).replace(/\.md$/, '');
  const parts = original.split('/');
  const file = parts.at(-1)!;
  if (!path.endsWith('.md') || !/^ml[12]-2026-spring$/.test(parts[0]) ||
    parts.length < 2 || parts.some(part => !part || part === '.' || part === '..' || /[\x00-\x1f]/.test(part)) ||
    !/\.(?:pdf|ipynb|py)$/.test(file)) {
    throw new Error('Invalid HSE catalogue source path: ' + source);
  }
  const title = file.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ').trim().toLowerCase()
    .replace(/(^|[^\p{L}])(\p{L})/gu, (_, delimiter: string, char: string) => delimiter + char.toUpperCase());
  return { title, href: upstream + parts.map(encodeURIComponent).join('/') };
}
