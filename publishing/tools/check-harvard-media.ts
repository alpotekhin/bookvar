import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parse } from 'yaml';

const course = '05 Источники/Courses/Harvard ML Systems';
const mediaPrefix = 'Sources/Harvard ML Systems/';
const courseRoute = /^\/(?:en\/)?sources\/courses\/harvard-ml-systems(?:\/|$)/;
const origin = 'https://bookvar.invalid';
const root = resolve(import.meta.dirname, '../..');

interface Chapter { images: { file: string; sha256: string; source_url: string }[] }
interface Slide {
  state: 'local-original' | 'external-source-link';
  local_path?: string;
  sha256?: string;
  source: string;
  source_commit: string;
  git_blob_sha1: string;
}
interface Media { kind: 'chapters' | 'slides'; sha256: string; sourcePath: string; gitBlob?: string }
interface UpstreamTree { sha: string; truncated: boolean; tree: { type: string; path: string; sha: string }[] }
interface Upstream { chapters: Map<string, string>; slides: Map<string, string> }

function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(file, 'utf8')) as T;
}

// Optional, immutable Git evidence for the imported editions. No network call
// is needed for routine checks of the built media against the import manifests.
export function readHarvardUpstreamTrees(directory: string): Upstream {
  const readTree = (file: string, sha: string, prefix = '') => {
    const t = readJson<UpstreamTree>(join(directory, file));
    if (t.sha !== sha || t.truncated || !Array.isArray(t.tree))
      throw new Error(file + ': expected the complete pinned tree ' + sha);
    return t.tree.filter(x => x.type === 'blob').map(x => [prefix + x.path, x.sha] as const);
  };
  return {
    chapters: new Map([
      ...readTree('vol1-tree.json', '259aad6d6e151c1769aaacdd9e884efcea5ac2cb', 'vol1/'),
      ...readTree('vol2-tree.json', '9cffef1270bbb694a194337d700b2bd9d07285b2', 'vol2/'),
    ]),
    slides: new Map(readTree('slides-tree.json', '45ecc8d82fcae70c149cdce550d3b3d3411df913')),
  };
}

function portableMediaPath(value: string): string {
  const file = value.replace(/^Assets\//, '');
  if (!file.startsWith(mediaPrefix) || file.split('/').some(x => x === '' || x === '.' || x === '..'))
    throw new Error('Invalid Harvard media path: ' + value);
  return file;
}

function sourceLibraryPath(file: string, dist: string): string {
  return '/' + relative(dist, file).split(sep).join('/').replace(/index\.html$/, '');
}

export function auditHarvardMedia({
  dist, chapters, slides, retired, basePath = '', upstream,
}: {
  dist: string; chapters: Chapter[]; slides: Slide[]; retired: string[];
  basePath?: string; upstream?: Upstream;
}) {
  const manifest = new Map<string, Media>();
  const withheld = new Set(retired.map(portableMediaPath));
  const intentionallyExternal = new Set<string>();
  const errors: string[] = [];
  const register = (file: string, info: Media) => {
    if (manifest.has(file)) throw new Error('Duplicate Harvard media record: ' + file);
    if (!/^[0-9a-f]{64}$/.test(info.sha256)) throw new Error('Missing media SHA-256: ' + file);
    manifest.set(file, info);
  };
  for (const chapter of chapters) for (const image of chapter.images) {
    const url = new URL(image.source_url);
    if (url.origin !== 'https://mlsysbook.ai') throw new Error('Unexpected chapter source: ' + url.href);
    register(portableMediaPath(image.file), {
      kind: 'chapters', sha256: image.sha256, sourcePath: decodeURIComponent(url.pathname).slice(1),
    });
  }
  for (const slide of slides) {
    if (slide.state === 'external-source-link') {
      if (slide.local_path) throw new Error('External slide asset has a local path: ' + slide.source);
      intentionallyExternal.add(portableMediaPath(mediaPrefix + slide.source));
      continue;
    }
    if (slide.state !== 'local-original' || !slide.local_path || !slide.sha256 ||
        !/^[0-9a-f]{40}$/.test(slide.git_blob_sha1))
      throw new Error('Incomplete slide media record: ' + slide.source);
    if (upstream && slide.source_commit !== '45ecc8d82fcae70c149cdce550d3b3d3411df913')
      throw new Error('Slide source revision does not match upstream tree: ' + slide.source);
    register(portableMediaPath(slide.local_path), {
      kind: 'slides', sha256: slide.sha256, sourcePath: slide.source, gitBlob: slide.git_blob_sha1,
    });
  }
  if (!manifest.size) errors.push('No Harvard media records loaded.');
  for (const file of intentionallyExternal) if (manifest.has(file))
    errors.push('External-only slide asset also registered for distribution: ' + file);

  const base = basePath.trim().replace(/^\/+|\/+$/g, '');
  const prefix = base ? '/' + base : '';
  const visible = new Map<string, Set<string>>();
  const scanned = new Set<string>();
  const published = new Set<string>();
  let coursePages = 0;
  for (const entry of readdirSync(dist, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || entry.name !== 'index.html') continue;
    const file = join(entry.parentPath, entry.name);
    const route = sourceLibraryPath(file, dist);
    const coursePage = courseRoute.test(route);
    const main = readFileSync(file, 'utf8').match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)?.[1];
    if (!main) continue;
    if (coursePage) coursePages++;
    const html = main.replace(/<(pre|code|script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, '');
    for (const m of html.matchAll(/<img\b[^>]*?\ssrc\s*=\s*(?:"([^"]+)"|'([^']+)')/gi)) {
      const src = (m[1] ?? m[2]).replaceAll('&amp;', '&');
      const url = new URL(src, origin + prefix + route);
      if (url.origin !== origin || (prefix && !url.pathname.startsWith(prefix + '/'))) continue;
      const path = decodeURIComponent(url.pathname.slice(prefix.length));
      if (!path.startsWith('/assets/')) continue;
      const rel = path.slice('/assets/'.length).replace(/^Assets\//, '');
      if (!rel.startsWith(mediaPrefix)) continue;
      const key = portableMediaPath(rel);
      if (coursePage) published.add(key);
      const paths = visible.get(key) ?? new Set<string>();
      paths.add(path);
      visible.set(key, paths);
      if (withheld.has(key) || intentionallyExternal.has(key)) errors.push(route + ': embeds excluded ' + key);
      if (!manifest.has(key)) errors.push(route + ': image missing from both course manifests: ' + key);
    }
  }
  if (!coursePages) errors.push('No built Harvard course pages found.');
  for (const file of [...withheld, ...intentionallyExternal]) {
    for (const base of ['assets', 'assets/Assets']) if (existsSync(join(dist, base, file)))
      errors.push('Excluded asset is distributed: ' + base + '/' + file);
  }
  const matching = { chapters: 0, slides: 0 };
  const upstreamMatching = { chapters: 0, slides: 0 };
  for (const [file, row] of manifest) {
    if (withheld.has(file) || intentionallyExternal.has(file)) continue;
    if (!published.has(file)) errors.push('Not visible in any Harvard course page: ' + file);
    const paths = visible.get(file);
    if (!paths) continue;
    let matches = true, upstreamMatches = true;
    for (const path of paths) {
      const disk = join(dist, path);
      if (!existsSync(disk)) {
        errors.push('Missing rendered image: ' + path);
        matches = false; upstreamMatches = false;
        continue;
      }
      scanned.add(path);
      const bytes = readFileSync(disk);
      if (createHash('sha256').update(bytes).digest('hex') !== row.sha256) {
        errors.push('SHA-256 mismatch: ' + path);
        matches = false;
      }
      if (row.gitBlob || upstream) {
        const actualBlob = createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');
        const upstreamBlob = upstream?.[row.kind].get(row.sourcePath);
        if ((row.gitBlob && actualBlob !== row.gitBlob) || (upstream && actualBlob !== upstreamBlob)) {
          errors.push('Git blob mismatch or missing source: ' + row.sourcePath + ' (' + path + ')');
          upstreamMatches = false;
        }
      }
    }
    if (matches) matching[row.kind]++;
    if (upstream && upstreamMatches) upstreamMatching[row.kind]++;
  }
  return {
    coursePages, visibleCourseMedia: published.size, checkedDistributionPaths: scanned.size,
    matching, upstreamMatching: upstream ? upstreamMatching : null,
    retired: withheld.size, externalOnlySlides: intentionallyExternal.size, errors: [...new Set(errors)].sort(),
  };
}

export function auditBuiltHarvardMedia(dist: string, basePath = '', upstream?: Upstream) {
  const registry = parse(readFileSync(join(root, '05 Источники/asset-registry.yml'), 'utf8'),
    { merge: true });
  const retired = registry.assets.filter((x: { asset?: string; publication_status?: string }) =>
    x.asset?.startsWith('Assets/' + mediaPrefix) && x.publication_status === 'retired')
    .map((x: { asset: string }) => x.asset);
  return auditHarvardMedia({
    dist, basePath,
    chapters: readJson<Chapter[]>(join(root, course, 'import-manifest.json')),
    slides: readJson<Slide[]>(join(root, course, 'slide-assets-manifest.json')),
    retired, upstream,
  });
}

function main() {
  const [option, directory, ...extra] = process.argv.slice(2);
  if (option && (option !== '--upstream' || !directory || extra.length))
    throw new Error('Usage: check:harvard-media [--upstream DIRECTORY]');
  const result = auditBuiltHarvardMedia(join(root, 'site/dist'), process.env.PUBLICATION_BASE_PATH,
    directory ? readHarvardUpstreamTrees(directory) : undefined);
  console.log(JSON.stringify(result, null, 2));
  if (result.errors.length) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main();
