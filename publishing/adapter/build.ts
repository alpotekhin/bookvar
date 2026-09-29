import { access, copyFile, mkdir, readFile, realpath, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import { convertCallouts } from './callouts.js';
import { readPage } from './frontmatter.js';
import { convertWikiSyntax, wikiHeadingSlug } from './links.js';
import { loadLinkAllowlist } from './link-policy.js';
import { loadManifest } from './manifest.js';
import type { PublicationSidebarItem } from './manifest.js';
import { createRouteRegistry } from './routes.js';
import { mapMarkdownProse } from './markdown-prose.js';

export interface BuildOptions {
  rootDir: string;
  manifestPath: string;
  outputDir: string;
  reportPath: string;
}

function publicationBasePath(): string {
  const configured = process.env.PUBLICATION_BASE_PATH?.trim() ?? '';
  if (configured === '' || configured === '/') return '';
  return `/${configured.replace(/^\/+|\/+$/g, '')}`;
}

interface Asset {
  source: string;
  publicPath: string;
}

interface SidebarEntry {
  label: string;
  translations?: { en: string };
  slug?: string;
  items?: SidebarEntry[];
  collapsed?: boolean;
}

export function publicationHref(route: string, locale?: 'en'): string {
  const localizedRoute = locale ? `${locale}/${route}` : route;
  const path = localizedRoute.endsWith('/index')
    ? `/${localizedRoute.slice(0, -'/index'.length)}/`
    : `/${localizedRoute}/`;
  return `${publicationBasePath()}${path}`;
}

const REFERENCE_SIDEBAR_GROUPS = [
  { label: 'Механизмы', labelEn: 'Mechanisms', includes: (route: string) => route.startsWith('reference/') },
  {
    label: 'Модели и семейства',
    labelEn: 'Models and families',
    includes: (route: string) => route.startsWith('models/') && route !== 'models/timeline'
  },
  { label: 'Обзоры направлений', labelEn: 'Research overviews', includes: (route: string) => route.startsWith('research/') },
  { label: 'Хронология', labelEn: 'Timeline', includes: (route: string) => route === 'models/timeline' }
] as const;

const SIDEBAR_SECTION_ORDER = ['textbook', 'models', 'sources', 'practice', 'questions'] as const;

function visibleInSidebar(route: string): boolean {
  return !route.includes('/legacy/')
    && !route.startsWith('sources/papers/')
    && !route.startsWith('sources/courses/')
    && !route.startsWith('sources/imbalanced-learn/');
}

function searchableInPublication(route: string): boolean {
  // Deep source archives are supporting evidence, not chapters. Keep them
  // available through the curated source indexes and direct links, but do not
  // let mechanically extracted slides, papers, and notebooks dominate the
  // textbook search results.
  return !route.startsWith('sources/courses/')
    && !route.startsWith('sources/papers/')
    && !route.startsWith('sources/imbalanced-learn/');
}

function normalizeProtocolRelativeHtmlUrls(markdown: string): string {
  // Imported notebooks often preserve HTML copied from Wikimedia and other
  // sites with `//host/path` URLs. Astro otherwise treats these as local
  // publication paths during static rendering.
  return markdown.replace(/\b(href|src)=(['"])\/\//gi, '$1=$2https://');
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderSourceAttribution(page: Awaited<ReturnType<typeof readPage>>): string {
  if (page.sourceFragments.length === 0) return '';
  return page.sourceFragments.map((fragment) => [
    '<aside class="source-attribution" aria-label="Source attribution">',
    '  <p class="source-attribution__label">Original course material</p>',
    `  <p><strong>${escapeHtml(fragment.course)}</strong> · ${escapeHtml(fragment.author)}</p>`,
    `  <p>Language: ${fragment.language} · License: ${escapeHtml(fragment.license)} · Modification: ${fragment.transformation}</p>`,
    `  <p><a href="${escapeHtml(fragment.sourceUrl)}">Open the pinned source</a></p>`,
    '</aside>'
  ].join('\n')).join('\n\n') + '\n\n';
}

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;
const EMBED = /!\[\[([^\]\n]+)\]\]/g;

function removeLeadingSourceHeading(markdown: string): string {
  let offset = 0;
  for (const line of markdown.split(/(?<=\n)/)) {
    const content = line.replace(/\r?\n$/, '');
    if (/^[ \t]*$/.test(content)
      || /^ {0,3}<(a|span)\s+id=["']([^"'<>\s]+)["'](?:\s+aria-hidden=["']true["'])?\s*>\s*<\/\1>\s*$/.test(content)) {
      offset += line.length;
      continue;
    }
    if (!/^#[ \t]+.+$/.test(content)) return markdown;
    const anchors = markdown.slice(0, offset).replace(/^(?:[ \t]*\r?\n)*/, '');
    return anchors + markdown.slice(offset + line.length);
  }
  return markdown;
}

export function normalizeObsidianHeading(heading: string): string {
  return heading.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ');
}

interface MarkdownHeading {
  text: string;
  normalized: string;
  offset: number;
  explicitAnchor?: string;
}

export function parseMarkdownHeadings(markdown: string, includeExplicitAnchors = false): MarkdownHeading[] {
  const headings: MarkdownHeading[] = [];
  let fence: { character: '`' | '~'; length: number } | undefined;
  let offset = 0;
  for (const line of markdown.split(/(?<=\n)/)) {
    const content = line.replace(/\r?\n$/, '');
    const delimiterMatch = content.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    const delimiter = delimiterMatch
      && !(delimiterMatch[1][0] === '`' && delimiterMatch[2].includes('`'))
      ? delimiterMatch
      : null;
    if (fence) {
      const closing = content.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
      if (
        closing
        && closing[1][0] === fence.character
        && closing[1].length >= fence.length
      ) {
        fence = undefined;
      }
    } else if (delimiter) {
      fence = {
        character: delimiter[1][0] as '`' | '~',
        length: delimiter[1].length
      };
    } else {
      const match = content.match(/^#{1,6}[ \t]+(.+?)[ \t]*#?[ \t]*$/);
      if (match) {
        const text = match[1].replace(/[ \t]+#$/, '').trim();
        headings.push({ text, normalized: normalizeObsidianHeading(text), offset });
      }
      if (includeExplicitAnchors) {
        // Only empty, standalone anchors are editorial fragment aliases.
        // The same fence scanner excludes literal HTML in code examples.
        const anchor = content.match(/^ {0,3}<(a|span)\s+id=["']([^"'<>\s]+)["'](?:\s+aria-hidden=["']true["'])?\s*>\s*<\/\1>\s*$/);
        if (anchor) headings.push({
          text: anchor[2], normalized: normalizeObsidianHeading(anchor[2]),
          offset, explicitAnchor: anchor[2]
        });
      }
    }
    offset += line.length;
  }
  return headings;
}

export function insertFragmentAliases(
  markdown: string,
  aliases: ReadonlyMap<string, string>
): { markdown: string; missing: string[] } {
  const headings = parseMarkdownHeadings(markdown);
  const insertions: Array<{ offset: number; anchor: string }> = [];
  const missing: string[] = [];
  for (const [targetHeading, anchor] of aliases) {
    const matches = headings.filter((heading) => heading.normalized === normalizeObsidianHeading(targetHeading));
    if (matches.length !== 1) missing.push(targetHeading);
    else insertions.push({ offset: matches[0].offset, anchor });
  }
  let converted = markdown;
  for (const insertion of insertions.sort((a, b) => b.offset - a.offset)) {
    converted = `${converted.slice(0, insertion.offset)}<span id="${insertion.anchor}" aria-hidden="true"></span>\n${converted.slice(insertion.offset)}`;
  }
  return { markdown: converted, missing };
}

function contained(root: string, candidate: string, label: string): string {
  if (isAbsolute(candidate)) throw new Error(`${label} path escapes rootDir: ${candidate}`);
  const absolute = resolve(root, candidate);
  const offset = relative(resolve(root), absolute);
  if (offset === '..' || offset.startsWith(`..${sep}`) || isAbsolute(offset)) {
    throw new Error(`${label} path escapes rootDir: ${candidate}`);
  }
  return absolute;
}

const COURSE_PUBLICATIONS = [
  {
    sourceRoot: '05 Источники/Courses/Stanford CS336 Spring 2026',
    routeRoot: 'sources/courses/stanford-cs336-spring-2026'
  },
  {
    sourceRoot: '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025',
    routeRoot: 'sources/courses/berkeley-advanced-llm-agents-spring-2025'
  }
] as const;
type CoursePublication = (typeof COURSE_PUBLICATIONS)[number];
interface LinkedCourseArtifact extends Asset {
  course: CoursePublication;
}

const LINKED_SOURCE_ARTIFACT = /\.(?:pdf|py|json|ya?ml|txt|md)$/i;
const MARKDOWN_LINK = /(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)/g;
const BERKELEY_READING_ROOT =
  '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Readings';
const AUDITED_BERKELEY_READING_SOURCES = new Set([
  'meeting-01-reading-01.md',
  'meeting-01-reading-02.md',
  'meeting-01-reading-03.md',
  'meeting-02-reading-01.md',
  'meeting-02-reading-02.md',
  'meeting-02-reading-03.md',
  'meeting-03-reading-01.md',
  'meeting-03-reading-02.md',
  'meeting-03-reading-03.md',
  'meeting-04-reading-01.md',
  'meeting-04-reading-02.md',
  'meeting-04-reading-03.md',
  'meeting-05-reading-01.md',
  'meeting-05-reading-02.md',
  'meeting-06-reading-01.md',
  'meeting-06-reading-02.md',
  'meeting-06-reading-03.md',
  'meeting-06-reading-04.md',
  'meeting-07-reading-01.md',
  'meeting-07-reading-02.md',
  'meeting-08-reading-01.md',
  'meeting-08-reading-02.md',
  'meeting-08-reading-03.md',
  'meeting-08-reading-04.md',
  'meeting-09-reading-01.md',
  'meeting-09-reading-02.md',
  'meeting-09-reading-03.md',
  'meeting-10-reading-01.md',
  'meeting-10-reading-02.md',
  'meeting-10-reading-03.md',
  'meeting-10-reading-04.md',
  'meeting-11-reading-01.md',
  'meeting-11-reading-02.md',
  'meeting-12-reading-01.md',
  'meeting-12-reading-02.md',
  'meeting-12-reading-03.md',
  'meeting-12-reading-04.md'
].map((filename) => `${BERKELEY_READING_ROOT}/${filename}`));
const BERKELEY_READING_PATTERN =
  /^05 Источники\/Courses\/Berkeley Advanced LLM Agents Spring 2025\/Readings\/meeting-\d{2}-reading-\d{2}\.md$/;
const APPROVED_RAW_MARKDOWN_ARTIFACTS = new Set([
  '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/README.md',
  '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment2-systems/README.md',
  '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment3-scaling/README.md',
  '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment4-data/README.md',
  '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment5-alignment/README.md'
]);

function prepareLinkedCourseArtifacts(
  rootDir: string,
  source: string,
  route: string,
  markdown: string,
  routeBySource: ReadonlyMap<string, string>
): { markdown: string; artifacts: LinkedCourseArtifact[] } {
  const course = COURSE_PUBLICATIONS.find(({ routeRoot }) =>
    route === `${routeRoot}/index` || route.startsWith(`${routeRoot}/`)
  );
  const artifacts: LinkedCourseArtifact[] = [];
  const converted = mapMarkdownProse(markdown, (prose) => prose.replace(MARKDOWN_LINK, (match, label: string, rawTarget: string) => {
    const target = rawTarget.trim();
    if (
      target === ''
      || target.startsWith('#')
      || target.startsWith('/')
      || /^[a-z][a-z0-9+.-]*:/i.test(target)
    ) {
      return match;
    }

    const suffixAt = target.search(/[?#]/);
    const encodedPath = suffixAt === -1 ? target : target.slice(0, suffixAt);
    const suffix = suffixAt === -1 ? '' : target.slice(suffixAt);
    let decodedPath: string;
    try {
      decodedPath = decodeURIComponent(encodedPath);
    } catch {
      return match;
    }
    if (!course && !LINKED_SOURCE_ARTIFACT.test(decodedPath)) return match;
    const targetSource = join(dirname(source), decodedPath).split(sep).join('/');
    const targetPath = contained(rootDir, targetSource, 'Linked source artifact');
    const normalizedSource = relative(rootDir, targetPath).split(sep).join('/');
    const targetRoute = routeBySource.get(normalizedSource);
    if (targetRoute) {
      return `[${label}](${publicationHref(targetRoute)}${suffix})`;
    }
    const artifactCourse = course ?? COURSE_PUBLICATIONS.find(
      ({ sourceRoot }) => normalizedSource.startsWith(`${sourceRoot}/`)
    );
    if (!artifactCourse || !LINKED_SOURCE_ARTIFACT.test(decodedPath)) return match;
    if (/\.md$/i.test(decodedPath) && !APPROVED_RAW_MARKDOWN_ARTIFACTS.has(normalizedSource)) {
      throw new Error(
        `Linked source artifact is not an approved raw Markdown course artifact: ${normalizedSource}`
      );
    }

    const courseRoot = resolve(rootDir, artifactCourse.sourceRoot);
    const sourceOffset = relative(courseRoot, targetPath);
    if (
      sourceOffset === ''
      || sourceOffset === '..'
      || sourceOffset.startsWith(`..${sep}`)
      || isAbsolute(sourceOffset)
    ) {
      throw new Error(`Linked source artifact path escapes course root: ${target}`);
    }
    const publicPath = join(artifactCourse.routeRoot, sourceOffset).split(sep).join('/');
    contained(rootDir, publicPath, 'Linked source artifact destination');
    artifacts.push({ source: targetPath, publicPath, course: artifactCourse });
    const href = publicPath.split('/').map(encodeURIComponent).join('/');
    return `[${label}](${publicationBasePath()}/${href}${suffix})`;
  }));
  return { markdown: converted, artifacts };
}

function normalizeAsset(rootDir: string, expression: string): Asset {
  const target = expression.split('|', 1)[0].replaceAll('\\', '/');
  if (isAbsolute(target) || target.split('/').includes('..')) {
    throw new Error(`Asset path escapes rootDir: ${target}`);
  }

  const marker = 'ML & DL/';
  const markerAt = target.indexOf(marker);
  const belowRoot = markerAt === -1 ? target : target.slice(markerAt + marker.length);
  const source = contained(rootDir, belowRoot, 'Asset');
  const publicPath = belowRoot.startsWith('Assets/')
    ? belowRoot.slice('Assets/'.length)
    : belowRoot;
  contained(rootDir, publicPath, 'Asset');
  return { source, publicPath };
}

function prepareAssets(rootDir: string, markdown: string): { markdown: string; assets: Asset[] } {
  const assets: Asset[] = [];
  const converted = markdown.replace(EMBED, (match, expression: string) => {
    const target = expression.split('|', 1)[0];
    if (!IMAGE_EXTENSION.test(target)) return match;
    const asset = normalizeAsset(rootDir, expression);
    assets.push(asset);
    const filename = asset.publicPath.slice(asset.publicPath.lastIndexOf('/') + 1);
    const alt = filename.replace(IMAGE_EXTENSION, '');
    const href = asset.publicPath.split('/').map(encodeURIComponent).join('/');
    const assetHref = `${publicationBasePath()}/assets/${href}`;
    return `[![${alt}](${assetHref})](${assetHref})`;
  });
  return { markdown: converted, assets };
}

function readPublicationPage(sourcePath: string, raw: string): ReturnType<typeof readPage> {
  if (!BERKELEY_READING_PATTERN.test(sourcePath)) return readPage(sourcePath, raw);
  if (!AUDITED_BERKELEY_READING_SOURCES.has(sourcePath)) {
    throw new Error(`Source-native reading is not in the audited Berkeley reading set: ${sourcePath}`);
  }
  const title = raw.match(/^#\s+(.+?)\s*$/m)?.[1];
  if (!title) {
    throw new Error(`Source-native Berkeley reading is missing its title heading: ${sourcePath}`);
  }
  return readPage(sourcePath, matter.stringify(raw, {
    title,
    type: 'source-note',
    status: 'verified'
  }));
}

export async function buildPublication(options: BuildOptions): Promise<void> {
  const rootDir = resolve(options.rootDir);
  const outputOffset = relative(rootDir, resolve(options.outputDir));
  if (outputOffset === '') throw new Error('Output directory must be below rootDir');
  const outputDir = contained(rootDir, outputOffset, 'Output');
  const manifest = loadManifest(options.manifestPath);
  const allowlist = loadLinkAllowlist(join(dirname(options.manifestPath), 'link-allowlist.json'));
  const entries = manifest.sections.flatMap((section) => section.pages);
  const routeBySource = new Map(entries.map((entry) => [
    entry.source.replaceAll('\\', '/'), entry.route
  ]));
  const parsed = await Promise.all(entries.map(async (entry) => {
    const sourcePath = contained(rootDir, entry.source, 'Source');
    const raw = await readFile(sourcePath, 'utf8');
    const page = readPublicationPage(entry.source, raw);
    if (!entry.sourceEn) return { entry, page };
    const sourceEnPath = contained(rootDir, entry.sourceEn, 'Source');
    const rawEn = await readFile(sourceEnPath, 'utf8');
    return { entry, page, pageEn: readPublicationPage(entry.sourceEn, rawEn) };
  }));
  const titles = new Map(parsed.map(({ entry, page }) => [entry.route, page.title]));
  const titlesEn = new Map(parsed
    .filter(({ pageEn }) => pageEn)
    .map(({ entry, pageEn }) => [entry.route, pageEn!.title]));
  const statuses = new Map(parsed.map(({ entry, page }) => [entry.route, page.status]));
  const translatedLabel = (labelEn?: string): Pick<SidebarEntry, 'translations'> =>
    labelEn ? { translations: { en: labelEn } } : {};
  const translatedPageLabel = (label: string, labelEn: string | undefined): string | undefined => {
    if (!labelEn) return undefined;
    const prefix = label.match(/^((?:S\d+|\d+(?:\.\d+)?)\.\s+)/)?.[1] ?? '';
    return prefix && !labelEn.startsWith(prefix) ? `${prefix}${labelEn}` : labelEn;
  };
  const sidebarItems = (items: PublicationSidebarItem[]): SidebarEntry[] => items.map((item) => 'route' in item
    ? {
        label: item.label,
        ...translatedLabel(item.labelEn ?? translatedPageLabel(item.label, titlesEn.get(item.route))),
        slug: item.route
      }
    : {
        label: item.label,
        ...translatedLabel(item.labelEn),
        collapsed: true,
        items: sidebarItems(item.items)
      });
  const sidebar = [...manifest.sections]
    .sort((left, right) =>
      SIDEBAR_SECTION_ORDER.indexOf(left.id) - SIDEBAR_SECTION_ORDER.indexOf(right.id)
    )
    .map((section) => {
    if (section.sidebar) {
      return {
        label: section.title,
        ...translatedLabel(section.titleEn),
        items: sidebarItems(section.sidebar)
      };
    }
    if (section.id === 'models') {
      const referencePages = manifest.sections
        .filter((candidate) => candidate.id === 'models' || candidate.id === 'sources')
        .flatMap((candidate) => candidate.pages);
      return {
        label: section.title,
        ...translatedLabel(section.titleEn),
        items: REFERENCE_SIDEBAR_GROUPS.map((group) => ({
          label: group.label,
          ...translatedLabel(group.labelEn),
          items: referencePages
            .filter((entry) => group.includes(entry.route)
              && visibleInSidebar(entry.route)
              && !['redirect', 'legacy'].includes(statuses.get(entry.route) ?? ''))
            .map((entry) => ({
              label: titles.get(entry.route)!,
              ...translatedLabel(titlesEn.get(entry.route)),
              slug: entry.route
            }))
        }))
      };
    }

    if (section.id !== 'sources') {
      return {
        label: section.title,
        ...translatedLabel(section.titleEn),
        items: section.pages.filter((entry) => visibleInSidebar(entry.route) && statuses.get(entry.route) !== 'redirect').map((entry) => ({
          label: titles.get(entry.route)!,
          ...translatedLabel(titlesEn.get(entry.route)),
          slug: entry.route
        }))
      };
    }

    return {
      label: section.title,
      ...translatedLabel(section.titleEn),
      items: section.pages
        .filter((entry) => entry.route.startsWith('sources/')
          && visibleInSidebar(entry.route)
          && !['redirect', 'legacy'].includes(statuses.get(entry.route) ?? ''))
        .map((entry) => ({
          label: titles.get(entry.route)!,
          ...translatedLabel(titlesEn.get(entry.route)),
          slug: entry.route
        }))
    };
  });

  const assetsByPublicPath = new Map<string, Asset>();
  const linkedArtifactsByPublicPath = new Map<string, LinkedCourseArtifact>();
  const localizedPages = parsed.flatMap(({ entry, page, pageEn }) => [
    { entry, page, source: entry.source, locale: undefined },
    ...(pageEn && entry.sourceEn
      ? [{ entry, page: pageEn, source: entry.sourceEn, locale: 'en' as const }]
      : [])
  ]);
  const preparedPages = localizedPages.map(({ entry, page, source, locale }) => {
    const linked = prepareLinkedCourseArtifacts(rootDir, source, entry.route, page.body, routeBySource);
    for (const artifact of linked.artifacts) {
      const existing = linkedArtifactsByPublicPath.get(artifact.publicPath);
      if (existing && existing.source !== artifact.source) {
        throw new Error([
          `Linked source artifact destination collision: ${artifact.publicPath}`,
          `- ${relative(rootDir, existing.source)}`,
          `- ${relative(rootDir, artifact.source)}`
        ].join('\n'));
      }
      linkedArtifactsByPublicPath.set(artifact.publicPath, artifact);
    }
    const prepared = prepareAssets(rootDir, linked.markdown);
    for (const asset of prepared.assets) {
      const existing = assetsByPublicPath.get(asset.publicPath);
      if (existing && existing.source !== asset.source) {
        throw new Error([
          `Asset destination collision: ${asset.publicPath}`,
          `- ${relative(rootDir, existing.source)}`,
          `- ${relative(rootDir, asset.source)}`
        ].join('\n'));
      }
      assetsByPublicPath.set(asset.publicPath, asset);
    }
    return { entry, page, source, locale, prepared };
  });
  for (const asset of assetsByPublicPath.values()) {
    try {
      await access(asset.source);
    } catch (error) {
      throw new Error(`Missing asset: ${relative(rootDir, asset.source)}`, { cause: error });
    }
  }

  for (const artifact of linkedArtifactsByPublicPath.values()) {
    try {
      await access(artifact.source);
      const [realCourseRoot, realSource] = await Promise.all([
        realpath(contained(rootDir, artifact.course.sourceRoot, 'Course source root')),
        realpath(artifact.source)
      ]);
      const sourceOffset = relative(realCourseRoot, realSource);
      if (sourceOffset === '..' || sourceOffset.startsWith(`..${sep}`) || isAbsolute(sourceOffset)) {
        throw new Error(`Linked source artifact path escapes course root: ${relative(rootDir, artifact.source)}`);
      }
    } catch (error) {
      if (error instanceof Error && error.message.startsWith('Linked source artifact path escapes')) throw error;
      throw new Error(`Missing linked source artifact: ${relative(rootDir, artifact.source)}`, { cause: error });
    }
  }

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });
  const sidebarPath = contained(rootDir, 'site/generated-sidebar.mjs', 'Sidebar');
  await mkdir(dirname(sidebarPath), { recursive: true });
  await writeFile(sidebarPath, [
    '// Generated from publishing/navigation.yml. Do not edit.',
    `export default ${JSON.stringify(sidebar, null, 2)};`,
    ''
  ].join('\n'), 'utf8');
  const assetDir = join(rootDir, 'site', 'public', 'assets');
  await rm(assetDir, { recursive: true, force: true });
  await mkdir(assetDir, { recursive: true });
  const publicDir = contained(rootDir, 'site/public', 'Public');
  for (const { routeRoot } of COURSE_PUBLICATIONS) {
    await rm(contained(publicDir, routeRoot, 'Linked source artifact root'), { recursive: true, force: true });
  }

  const report = { pages: [] as Array<{
    source: string;
    route: string;
    unresolved: string[];
    allowlisted: Array<{ target: string; reason: string }>;
  }> };
  const registries = new Map<'root' | 'en', ReturnType<typeof createRouteRegistry>>([
    ['root', createRouteRegistry(parsed.map(({ entry, page }) => ({
      sourcePath: entry.source,
      route: publicationHref(entry.route),
      title: page.title
    })), { allowAmbiguousBasenames: true })],
    ['en', createRouteRegistry(parsed.flatMap(({ entry, page, pageEn }) => {
      const localized = {
        route: publicationHref(entry.route, 'en'),
        title: pageEn?.title ?? page.title
      };
      return [
        { sourcePath: entry.source, ...localized },
        ...(entry.sourceEn ? [{ sourcePath: entry.sourceEn, ...localized }] : [])
      ];
    }), { allowAmbiguousBasenames: true })]
  ]);
  const headingsByRoute = new Map(preparedPages.map(({ entry, locale, prepared }) => [
    publicationHref(entry.route, locale),
    parseMarkdownHeadings(prepared.markdown, true)
  ]));
  for (const { entry, pageEn } of parsed) {
    if (pageEn) continue;
    headingsByRoute.set(
      publicationHref(entry.route, 'en'),
      headingsByRoute.get(publicationHref(entry.route)) ?? []
    );
  }
  const fragmentAnchors = new Map<string, Map<string, string>>();
  for (const { entry, locale, prepared } of preparedPages) {
    const registry = registries.get(locale ?? 'root')!;
    for (const match of prepared.markdown.matchAll(/!?\[\[([^\]\n]+)\]\]/g)) {
      const expression = match[1].split('|', 1)[0];
      const headingAt = expression.indexOf('#');
      if (headingAt === -1) continue;
      const target = expression.slice(0, headingAt);
      const heading = expression.slice(headingAt + 1);
      if (!heading) continue;
      const route = target === ''
        ? publicationHref(entry.route, locale)
        : registry.routeForWikiTarget(target);
      if (!route) continue;
      const exact = headingsByRoute.get(route)?.filter(
        (candidate) => candidate.explicitAnchor
          ? candidate.explicitAnchor === heading || candidate.explicitAnchor === wikiHeadingSlug(heading)
          : candidate.normalized === normalizeObsidianHeading(heading)
      ) ?? [];
      const resolvedFragments = new Set(exact.map(
        (candidate) => candidate.explicitAnchor ?? wikiHeadingSlug(candidate.text)
      ));
      // One retained alias may coincide with one native heading. Multiple
      // headings or repeated explicit IDs are still ambiguous destinations.
      if (exact.filter((candidate) => !candidate.explicitAnchor).length > 1
        || exact.filter((candidate) => candidate.explicitAnchor).length > 1) continue;
      if (resolvedFragments.size !== 1) continue;
      const anchors = fragmentAnchors.get(route) ?? new Map<string, string>();
      // Use an existing explicit alias or the native heading slug; never inject
      // another HTML element into the chapter during link conversion.
      anchors.set(heading, [...resolvedFragments][0]);
      fragmentAnchors.set(route, anchors);
    }
  }

  for (const { entry, page, source, locale, prepared } of preparedPages) {
    const registry = registries.get(locale ?? 'root')!;
    const currentRoute = publicationHref(entry.route, locale);
    const pageRegistry = {
      routeForWikiTarget: registry.routeForWikiTarget,
      fragmentForWikiTarget(target: string, heading: string): string | undefined {
        const route = target === '' ? currentRoute : registry.routeForWikiTarget(target);
        if (!route) return undefined;
        return fragmentAnchors.get(route)?.get(heading);
      }
    };
    const converted = convertWikiSyntax(prepared.markdown, pageRegistry, allowlist);
    const legacyUnresolved = page.status === 'legacy'
      ? [...new Set(converted.unresolved)].map((target) => ({
          target,
          reason: 'legacy page preserved for old links; not part of the canonical publication'
        }))
      : [];
    const unresolved = page.status === 'legacy' ? [] : converted.unresolved;
    const allowlisted = [...converted.allowlisted, ...legacyUnresolved];
    const markdown = renderSourceAttribution(page) + normalizeProtocolRelativeHtmlUrls(
      removeLeadingSourceHeading(convertCallouts(converted.markdown))
    );
    const outputRoute = locale ? `${locale}/${entry.route}` : entry.route;
    const target = contained(outputDir, `${outputRoute}.md`, 'Output');
    await mkdir(dirname(target), { recursive: true });
    const metadata: Record<string, string | boolean | Date> = {
      title: page.title,
      description: page.title,
      slug: outputRoute
    };
    if (!searchableInPublication(entry.route) || ['redirect', 'legacy'].includes(page.status)) {
      metadata.pagefind = false;
    }
    if (page.lastUpdated) metadata.lastUpdated = new Date(page.lastUpdated);
    await writeFile(target, matter.stringify(markdown, metadata), 'utf8');

    if (unresolved.length > 0 || allowlisted.length > 0) {
      report.pages.push({
        source,
        route: outputRoute,
        unresolved: [...new Set(unresolved)],
        allowlisted: [...new Map(allowlisted.map((item) => [item.target, item])).values()]
      });
    }
  }

  const unexplained = report.pages.flatMap((page) => page.unresolved.map((target) => `${page.source}: ${target}`));
  if (unexplained.length > 0) {
    throw new Error(['Unexplained unresolved wiki links:', ...unexplained.map((item) => `- ${item}`)].join('\n'));
  }

  for (const asset of assetsByPublicPath.values()) {
    const destination = contained(assetDir, asset.publicPath, 'Asset');
    await mkdir(dirname(destination), { recursive: true });
    try {
      await copyFile(asset.source, destination);
    } catch (error) {
      throw new Error(`Missing asset: ${relative(rootDir, asset.source)}`, { cause: error });
    }
  }
  for (const artifact of linkedArtifactsByPublicPath.values()) {
    const destination = contained(publicDir, artifact.publicPath, 'Linked source artifact');
    await mkdir(dirname(destination), { recursive: true });
    try {
      await copyFile(artifact.source, destination);
    } catch (error) {
      throw new Error(
        `Missing linked source artifact: ${relative(rootDir, artifact.source)}`,
        { cause: error }
      );
    }
  }

  await mkdir(dirname(options.reportPath), { recursive: true });
  await writeFile(options.reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
}

const modulePath = fileURLToPath(import.meta.url);
if (process.argv[1] && resolve(process.argv[1]) === modulePath) {
  const publishingDir = resolve(dirname(modulePath), '..');
  const rootDir = resolve(publishingDir, '..');
  await buildPublication({
    rootDir,
    manifestPath: join(publishingDir, 'navigation.yml'),
    outputDir: join(rootDir, 'site', 'src', 'content', 'docs', 'generated'),
    reportPath: join(rootDir, 'publishing-report.json')
  });
}
