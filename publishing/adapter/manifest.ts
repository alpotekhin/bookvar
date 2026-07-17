import { existsSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { parse } from 'yaml';

const SECTION_IDS = ['textbook', 'models', 'sources', 'questions', 'practice'] as const;
const SECTION_ID_SET = new Set<string>(SECTION_IDS);
const ASCII_ROUTE = /^[a-z0-9]+(?:[/-][a-z0-9]+)*$/;

export type PublicationSectionId = typeof SECTION_IDS[number];

export interface PublicationPage {
  source: string;
  route: string;
}

export interface PublicationSection {
  id: PublicationSectionId;
  title: string;
  pages: PublicationPage[];
}

export interface PublicationManifest {
  siteTitle: string;
  sections: PublicationSection[];
}

function resolveContained(root: string, path: string): string {
  if (isAbsolute(path)) throw new Error(`Source path escapes publication root: ${path}`);
  const absolute = resolve(root, path);
  const offset = relative(root, absolute);
  if (offset === '..' || offset.startsWith(`..${sep}`) || isAbsolute(offset)) {
    throw new Error(`Source path escapes publication root: ${path}`);
  }
  return absolute;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be a mapping`);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function exactKeys(value: Record<string, unknown>, keys: string[], label: string): void {
  const expected = new Set(keys);
  const unknown = Object.keys(value).filter((key) => !expected.has(key));
  const missing = keys.filter((key) => !(key in value));
  if (unknown.length > 0 || missing.length > 0) {
    throw new Error(`${label} schema error (unknown: ${unknown.join(', ') || 'none'}; missing: ${missing.join(', ') || 'none'})`);
  }
}

export function loadManifest(path: string): PublicationManifest {
  const raw = record(parse(readFileSync(path, 'utf8')), 'Manifest');
  exactKeys(raw, ['site_title', 'sections'], 'Manifest');
  const siteTitle = text(raw.site_title, 'site_title');
  if (!Array.isArray(raw.sections) || raw.sections.length === 0) {
    throw new Error('sections must be a non-empty array');
  }

  const root = resolve(dirname(path), '..');
  const sources = new Set<string>();
  const routes = new Set<string>();
  const sectionIds = new Set<string>();

  const sections = raw.sections.map((sectionValue, sectionIndex): PublicationSection => {
    const section = record(sectionValue, `sections[${sectionIndex}]`);
    exactKeys(section, ['id', 'title', 'pages'], `sections[${sectionIndex}]`);
    const id = text(section.id, `sections[${sectionIndex}].id`);
    if (!SECTION_ID_SET.has(id)) throw new Error(`Unknown section ID: ${id}`);
    if (sectionIds.has(id)) throw new Error(`Duplicate section ID: ${id}`);
    sectionIds.add(id);
    const title = text(section.title, `sections[${sectionIndex}].title`);
    if (!Array.isArray(section.pages) || section.pages.length === 0) {
      throw new Error(`sections[${sectionIndex}].pages must be a non-empty array`);
    }

    const pages = section.pages.map((pageValue, pageIndex): PublicationPage => {
      const label = `sections[${sectionIndex}].pages[${pageIndex}]`;
      const page = record(pageValue, label);
      exactKeys(page, ['source', 'route'], label);
      const source = text(page.source, `${label}.source`);
      const route = text(page.route, `${label}.route`);

      if (route.startsWith('/')) throw new Error(`Route must not begin with /: ${route}`);
      if (!ASCII_ROUTE.test(route)) throw new Error(`Route must be stable lowercase ASCII: ${route}`);
      if (routes.has(route)) throw new Error(`Duplicate route: ${route}`);
      if (sources.has(source)) throw new Error(`Duplicate source: ${source}`);
      const sourcePath = resolveContained(root, source);
      if (!existsSync(sourcePath)) throw new Error(`Missing source: ${source}`);

      routes.add(route);
      sources.add(source);
      return { source, route };
    });

    return { id: id as PublicationSectionId, title, pages };
  });

  return { siteTitle, sections };
}
