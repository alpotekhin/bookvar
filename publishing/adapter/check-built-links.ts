import { readdir, readFile } from 'node:fs/promises';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

interface BuiltPage {
  file: string;
  route: string;
  html: string;
  ids: Set<string>;
}

async function builtFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) return builtFiles(path);
    return entry.isFile() ? [path] : [];
  }));
  return nested.flat();
}

function routeForFile(distDir: string, file: string): string {
  const offset = relative(distDir, dirname(file)).split(sep).join('/');
  return offset === '' ? '/' : `/${offset}/`;
}

function attributeValues(html: string, name: string): string[] {
  const values: string[] = [];
  const pattern = new RegExp(`\\b${name}=["']([^"']*)["']`, 'gi');
  for (const match of html.matchAll(pattern)) values.push(match[1].replaceAll('&amp;', '&'));
  return values;
}

function normalizedRoute(pathname: string): string {
  if (pathname.endsWith('/')) return pathname;
  return `${pathname}/`;
}

export async function findBrokenBuiltLinks(distDir: string): Promise<string[]> {
  const files = await builtFiles(distDir);
  const filePaths = new Set(files.map((file) => `/${relative(distDir, file).split(sep).join('/')}`));
  const pages = await Promise.all(files.filter((file) => file.endsWith(`${sep}index.html`)).map(async (file): Promise<BuiltPage> => {
    const html = await readFile(file, 'utf8');
    return { file, route: routeForFile(distDir, file), html, ids: new Set(attributeValues(html, 'id')) };
  }));
  const byRoute = new Map(pages.map((page) => [page.route, page]));
  const errors: string[] = [];

  for (const page of pages) {
    for (const href of [...attributeValues(page.html, 'href'), ...attributeValues(page.html, 'src')]) {
      if (/^(?:https?:|mailto:|tel:|javascript:)/i.test(href)) continue;
      const url = new URL(href, `https://handbook.invalid${page.route}`);
      if (/^\/(?:_astro|assets|pagefind)(?:\/|$)/.test(url.pathname) || /\.[a-z0-9]+$/i.test(url.pathname)) {
        const filePath = decodeURIComponent(url.pathname);
        if (!filePaths.has(filePath)) errors.push(`${page.route} -> missing file ${filePath}`);
        continue;
      }
      const route = normalizedRoute(decodeURIComponent(url.pathname));
      const target = byRoute.get(route);
      if (!target) {
        errors.push(`${page.route} -> missing route ${route}`);
        continue;
      }
      if (url.hash) {
        const fragment = decodeURIComponent(url.hash.slice(1));
        if (!target.ids.has(fragment)) errors.push(`${page.route} -> missing fragment ${route}#${fragment}`);
      }
    }
  }
  return [...new Set(errors)].sort();
}

const modulePath = fileURLToPath(import.meta.url);
if (process.argv[1] && resolve(process.argv[1]) === modulePath) {
  const distDir = resolve(dirname(modulePath), '../../site/dist');
  const errors = await findBrokenBuiltLinks(distDir);
  if (errors.length > 0) {
    console.error(['Broken built links:', ...errors.map((error) => `- ${error}`)].join('\n'));
    process.exitCode = 1;
  } else {
    console.log('Built link check passed: 0 broken internal routes, fragments, or files.');
  }
}
