import { access, copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import { convertCallouts } from './callouts.js';
import { readPage } from './frontmatter.js';
import { convertWikiSyntax, wikiHeadingSlug } from './links.js';
import { loadLinkAllowlist } from './link-policy.js';
import { loadManifest } from './manifest.js';
import { createRouteRegistry } from './routes.js';

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

export function publicationHref(route: string): string {
  const path = route.endsWith('/index')
    ? `/${route.slice(0, -'/index'.length)}/`
    : `/${route}/`;
  return `${publicationBasePath()}${path}`;
}

const REFERENCE_SIDEBAR_GROUPS = [
  { label: 'Механизмы', includes: (route: string) => route.startsWith('reference/') },
  {
    label: 'Модели и семейства',
    includes: (route: string) => route.startsWith('models/') && route !== 'models/timeline'
  },
  { label: 'Обзоры направлений', includes: (route: string) => route.startsWith('research/') },
  { label: 'Хронология', includes: (route: string) => route === 'models/timeline' }
] as const;

const SIDEBAR_SECTION_ORDER = ['textbook', 'models', 'sources', 'practice', 'questions'] as const;

function visibleInSidebar(route: string): boolean {
  return !route.includes('/legacy/')
    && !route.startsWith('sources/papers/')
    && !route.startsWith('sources/courses/');
}

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;
const EMBED = /!\[\[([^\]\n]+)\]\]/g;

function removeLeadingSourceHeading(markdown: string): string {
  const heading = markdown.match(/^(?:[ \t]*\r?\n)*#\s+.+?(?:\r?\n|$)/);
  if (!heading) return markdown;
  return markdown.slice(heading[0].length);
}

export function normalizeObsidianHeading(heading: string): string {
  return heading.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ');
}

interface MarkdownHeading {
  text: string;
  normalized: string;
  offset: number;
}

export function parseMarkdownHeadings(markdown: string): MarkdownHeading[] {
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
    return `![${alt}](${publicationBasePath()}/assets/${href})`;
  });
  return { markdown: converted, assets };
}

export async function buildPublication(options: BuildOptions): Promise<void> {
  const rootDir = resolve(options.rootDir);
  const outputOffset = relative(rootDir, resolve(options.outputDir));
  if (outputOffset === '') throw new Error('Output directory must be below rootDir');
  const outputDir = contained(rootDir, outputOffset, 'Output');
  const manifest = loadManifest(options.manifestPath);
  const allowlist = loadLinkAllowlist(join(dirname(options.manifestPath), 'link-allowlist.json'));
  const entries = manifest.sections.flatMap((section) => section.pages);
  const parsed = await Promise.all(entries.map(async (entry) => {
    const sourcePath = contained(rootDir, entry.source, 'Source');
    const raw = await readFile(sourcePath, 'utf8');
    return { entry, page: readPage(entry.source, raw) };
  }));
  const registry = createRouteRegistry(parsed.map(({ entry, page }) => ({
    sourcePath: entry.source,
    route: publicationHref(entry.route),
    title: page.title
  })), { allowAmbiguousBasenames: true });
  const titles = new Map(parsed.map(({ entry, page }) => [entry.route, page.title]));
  const statuses = new Map(parsed.map(({ entry, page }) => [entry.route, page.status]));
  const sidebar = [...manifest.sections]
    .sort((left, right) =>
      SIDEBAR_SECTION_ORDER.indexOf(left.id) - SIDEBAR_SECTION_ORDER.indexOf(right.id)
    )
    .map((section) => {
    if (section.id === 'models') {
      const referencePages = manifest.sections
        .filter((candidate) => candidate.id === 'models' || candidate.id === 'sources')
        .flatMap((candidate) => candidate.pages);
      return {
        label: section.title,
        items: REFERENCE_SIDEBAR_GROUPS.map((group) => ({
          label: group.label,
          items: referencePages
            .filter((entry) => group.includes(entry.route)
              && visibleInSidebar(entry.route)
              && !['redirect', 'legacy'].includes(statuses.get(entry.route) ?? ''))
            .map((entry) => ({
              label: titles.get(entry.route)!,
              slug: entry.route
            }))
        }))
      };
    }

    if (section.id !== 'sources') {
      return {
        label: section.title,
        items: section.pages.filter((entry) => visibleInSidebar(entry.route) && statuses.get(entry.route) !== 'redirect').map((entry) => ({
          label: titles.get(entry.route)!,
          slug: entry.route
        }))
      };
    }

    return {
      label: section.title,
      items: section.pages
        .filter((entry) => entry.route.startsWith('sources/')
          && visibleInSidebar(entry.route)
          && !['redirect', 'legacy'].includes(statuses.get(entry.route) ?? ''))
        .map((entry) => ({
          label: titles.get(entry.route)!,
          slug: entry.route
        }))
    };
  });

  const assetsByPublicPath = new Map<string, Asset>();
  const preparedPages = parsed.map(({ entry, page }) => {
    const prepared = prepareAssets(rootDir, page.body);
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
    return { entry, page, prepared };
  });
  for (const asset of assetsByPublicPath.values()) {
    try {
      await access(asset.source);
    } catch (error) {
      throw new Error(`Missing asset: ${relative(rootDir, asset.source)}`, { cause: error });
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
  const report = { pages: [] as Array<{
    source: string;
    route: string;
    unresolved: string[];
    allowlisted: Array<{ target: string; reason: string }>;
  }> };
  const headingsByRoute = new Map(preparedPages.map(({ entry, prepared }) => [
    publicationHref(entry.route),
    parseMarkdownHeadings(prepared.markdown)
  ]));
  const fragmentAnchors = new Map<string, Map<string, string>>();
  for (const { entry, prepared } of preparedPages) {
    for (const match of prepared.markdown.matchAll(/!?\[\[([^\]\n]+)\]\]/g)) {
      const expression = match[1].split('|', 1)[0];
      const headingAt = expression.indexOf('#');
      if (headingAt === -1) continue;
      const target = expression.slice(0, headingAt);
      const heading = expression.slice(headingAt + 1);
      if (!heading) continue;
      const route = target === '' ? publicationHref(entry.route) : registry.routeForWikiTarget(target);
      if (!route) continue;
      const exact = headingsByRoute.get(route)?.filter(
        (candidate) => candidate.normalized === normalizeObsidianHeading(heading)
      ) ?? [];
      if (exact.length !== 1) continue;
      const anchors = fragmentAnchors.get(route) ?? new Map<string, string>();
      // Astro/Starlight already assigns this slug to the Markdown heading.
      // Linking to that native anchor avoids injecting raw HTML before a
      // heading, which can make the page body disappear during rendering.
      anchors.set(heading, wikiHeadingSlug(heading));
      fragmentAnchors.set(route, anchors);
    }
  }

  for (const { entry, page, prepared } of preparedPages) {
    const currentRoute = publicationHref(entry.route);
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
    const markdown = removeLeadingSourceHeading(convertCallouts(converted.markdown));
    const target = contained(outputDir, `${entry.route}.md`, 'Output');
    await mkdir(dirname(target), { recursive: true });
    const metadata: Record<string, string | Date> = {
      title: page.title,
      description: page.title,
      slug: entry.route
    };
    if (page.lastUpdated) metadata.lastUpdated = new Date(page.lastUpdated);
    await writeFile(target, matter.stringify(markdown, metadata), 'utf8');

    if (unresolved.length > 0 || allowlisted.length > 0) {
      report.pages.push({
        source: entry.source,
        route: entry.route,
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
