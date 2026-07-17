import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import { convertCallouts } from './callouts.js';
import { readPage } from './frontmatter.js';
import { convertWikiSyntax } from './links.js';
import { loadManifest } from './manifest.js';
import { createRouteRegistry } from './routes.js';

export interface BuildOptions {
  rootDir: string;
  manifestPath: string;
  outputDir: string;
  reportPath: string;
}

interface Asset {
  source: string;
  publicPath: string;
}

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;
const EMBED = /!\[\[([^\]\n]+)\]\]/g;

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
    return `![[Assets/${asset.publicPath}]]`;
  });
  return { markdown: converted, assets };
}

function editUrl(source: string): string {
  return source.replaceAll('\\', '/');
}

export async function buildPublication(options: BuildOptions): Promise<void> {
  const rootDir = resolve(options.rootDir);
  const outputOffset = relative(rootDir, resolve(options.outputDir));
  if (outputOffset === '') throw new Error('Output directory must be below rootDir');
  const outputDir = contained(rootDir, outputOffset, 'Output');
  const manifest = loadManifest(options.manifestPath);
  const entries = manifest.sections.flatMap((section) => section.pages);
  const parsed = await Promise.all(entries.map(async (entry) => {
    const sourcePath = contained(rootDir, entry.source, 'Source');
    const raw = await readFile(sourcePath, 'utf8');
    return { entry, page: readPage(entry.source, raw) };
  }));
  const registry = createRouteRegistry(parsed.map(({ entry, page }) => ({
    sourcePath: entry.source,
    route: `/${entry.route}/`,
    title: page.title
  })));

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });
  const assetDir = join(rootDir, 'site', 'public', 'assets');
  const report = { pages: [] as Array<{
    source: string;
    route: string;
    unresolved: string[];
  }> };

  for (const { entry, page } of parsed) {
    const prepared = prepareAssets(rootDir, page.body);
    const converted = convertWikiSyntax(prepared.markdown, registry);
    const markdown = convertCallouts(converted.markdown);
    const target = contained(outputDir, `${entry.route}.md`, 'Output');
    await mkdir(dirname(target), { recursive: true });
    const metadata: Record<string, string> = {
      title: page.title,
      description: page.title,
      editUrl: editUrl(entry.source)
    };
    if (page.lastUpdated) metadata.lastUpdated = page.lastUpdated;
    await writeFile(target, matter.stringify(markdown, metadata), 'utf8');

    for (const asset of prepared.assets) {
      const destination = contained(assetDir, asset.publicPath, 'Asset');
      await mkdir(dirname(destination), { recursive: true });
      try {
        await copyFile(asset.source, destination);
      } catch (error) {
        throw new Error(`Missing asset: ${relative(rootDir, asset.source)}`, { cause: error });
      }
    }

    if (converted.unresolved.length > 0) {
      report.pages.push({
        source: entry.source,
        route: entry.route,
        unresolved: [...new Set(converted.unresolved)]
      });
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
