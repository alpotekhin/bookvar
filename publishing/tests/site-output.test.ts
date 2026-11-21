import { access, readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { findBrokenBuiltLinks } from '../adapter/check-built-links.js';
import { loadManifest } from '../adapter/manifest.js';
import { hseCatalogueEntry } from '../adapter/hse-catalogue.js';
import { readingPaths } from '../adapter/reading-paths.js';
import { auditBuiltHarvardMedia } from '../tools/check-harvard-media.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';
import { verifyCourseLibraryOriginals } from './course-library-output.js';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const distDir = join(rootDir, 'site', 'dist');
const pagefindDir = join(distDir, 'pagefind');

interface PagefindResult {
  data(): Promise<{ meta: { title: string }; url: string }>;
}

interface PagefindApi {
  destroy(): Promise<void>;
  options(options: Record<string, unknown>): Promise<void>;
  search(term: string): Promise<{ results: PagefindResult[] }>;
}

let pagefind: PagefindApi;
let originalFetch: typeof globalThis.fetch;

function expectCanonicalEditionNotice(html: string) {
  expect(html.match(/aria-label="English edition status"/g)).toHaveLength(1);
  expect(html).toContain('Canonical-language chapter.');
  expect(html).toContain('Read the canonical chapter');
  expect(html).not.toContain('Partial English edition.');
  expect(html).not.toContain('This content is not available in your language yet.');
}

function expectCourseOriginalAlias(html: string, route: string) {
  expect(html).toContain(`0;url=/${route}/`);
  expect(html).toContain(`href="/${route}/"`);
  expect(html).toContain('location.search + location.hash');
  expect(html).toContain('noindex');
}

beforeAll(async () => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
    if (!url.startsWith('file:')) return originalFetch(input, init);
    const contents = await readFile(fileURLToPath(url));
    return new Response(contents as unknown as BodyInit, {
      status: 200,
      headers: { 'content-type': url.endsWith('.pagefind') ? 'application/wasm' : 'application/octet-stream' }
    });
  };

  const basePath = pathToFileURL(`${pagefindDir}/`).href;
  pagefind = await import(`${basePath}pagefind.js`) as PagefindApi;
  await pagefind.options({ basePath, language: 'ru' });
});

afterAll(async () => {
  await pagefind?.destroy();
  globalThis.fetch = originalFetch;
});

describe('static handbook output', () => {
  it('serves all 23 imbalanced-learn gallery figures locally with their original attribution', async () => {
    const source = join(rootDir, '05 Источники/imbalanced-learn/0.14.2');
    const { figures, galleries, docs_base } = JSON.parse(await readFile(join(source, 'figures.json'), 'utf8')) as {
      figures: Record<string, { asset: string; gallery: string; sha256: string }>;
      galleries: Record<string, { author: string }>; docs_base: string;
    };
    const { pages } = JSON.parse(await readFile(join(source, 'published/import-manifest.json'), 'utf8')) as {
      pages: Array<{ route: string }>;
    };
    const seen = new Set<string>();
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const { route } of pages) {
      for (const prefix of ['', 'en/']) {
        const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
        expect(html).not.toMatch(/<img\b[^>]*src="https?:\/\/imbalanced-learn\.org\//);
        if (prefix && html.includes('0;url=')) {
          expect(html).toContain(`0;url=${base}/${route}/`);
          continue;
        }
        for (const [name, figure] of Object.entries(figures)) {
          const asset = figure.asset.replace(/^Assets\//, '');
          const url = `${base}/assets/${asset.split('/').map(encodeURIComponent).join('/')}`;
          if (!html.includes(`src="${url}"`)) continue;
          expect(html).toContain(`href="${url}"`);
          expect(html).toContain(`href="${docs_base}${figure.gallery}"`);
          expect(html).toContain(`href="${docs_base}_images/${name}"`);
          expect(html).toContain(galleries[figure.gallery].author);
          expect(createHash('sha256').update(await readFile(join(distDir, 'assets', asset))).digest('hex'))
            .toBe(figure.sha256);
          seen.add(name);
        }
      }
    }
    expect(seen).toEqual(new Set(Object.keys(figures)));
    expect(seen.size).toBe(23);
    const modelSelection = await readFile(join(distDir, 'sources/imbalanced-learn/0-14-2/user-guide/model-selection/index.html'), 'utf8');
    expect(modelSelection).toContain('Top: InstanceHardnessCV. Bottom: StratifiedKFold.');
    expect(modelSelection.match(/<pre\b/g)).toHaveLength(5);
  });

  it('links all 67 HSE originals and excludes the notebook archive media from the public artifact', async () => {
    const records = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections
      .find(section => section.id === 'sources')!.pages
      .flatMap(page => {
        const entry = hseCatalogueEntry(page.source, page.route);
        return entry ? [{ ...entry, route: page.route }] : [];
      });
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    expect(records).toHaveLength(67);
    for (const record of records) {
      const html = await readFile(join(distDir, record.route, 'index.html'), 'utf8');
      const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
      expect(main).toContain('href="' + record.href + '"');
      expect(main).toContain(record.title);
      expect(main).not.toMatch(/<pre\b|<img\b[^>]+\bsrc=/i);
      const alias = await readFile(join(distDir, 'en', record.route, 'index.html'), 'utf8');
      expect(alias).toContain('0;url=' + base + '/' + record.route + '/');
    }
    await expect(access(join(distDir, 'assets/Sources/HSE ML course'))).rejects.toMatchObject({ code: 'ENOENT' });
    const published = await readdir(join(distDir, 'sources/courses/hse-ml-course'), { recursive: true });
    expect(published.filter(file => /\.(pdf|ipynb|py)$/.test(file))).toEqual([]);
  });
  it('preserves the four document intake diagrams from the pinned GenAI Agents tutorial', async () => {
    const revision = 'bd681451b254ac1a790e947b581d3997ab35013d';
    const route = 'sources/courses/genai-agents/document-intake-agent-langgraph';
    const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
    const upstream = 'https://github.com/NirDiamant/GenAI_Agents/blob/' + revision;
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const [name, sha] of Object.entries({
      'agent': '1520332846bffa9fc8234544b5220604db322b2c',
      'conversion-flow': 'e5a8bb2ceef2e36360355a937ef882663ff47a05',
      'reading-order': 'f8d6c3c69dbd6b8105eb1852bad04eb7c88054d0',
      'table-loss': 'b30a174e20ebad8b954affd2a2979eec5dda0a13',
    })) {
      const file = 'document-intake-' + name + '.svg';
      const path = '/assets/Sources/GenAI Agents/document-intake-originals/' + file;
      expect(html).toContain('src="' + base + path.replaceAll(' ', '%20') + '"');
      expect(html).toContain('href="' + upstream + '/images/' + file + '"');
      const bytes = await readFile(join(distDir, path));
      expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')).toBe(sha);
    }
    expect(html.match(/Nir Diamant and GenAI Agents contributors/g)).toHaveLength(4);
    expect(html).toContain(upstream + '/LICENSE');
    expect(html).toContain('Comparison with Naive Extraction');
    expect(html).toContain('128.4');
  });
  it('restores teaching diagrams from GenAI notebooks without shipping the unreviewed screenshots', async () => {
    const publicAssets = join(distDir, 'assets/Sources/GenAI Agents/all_agents_tutorials');
    const root = join(rootDir, 'Assets/Sources/GenAI Agents/all_agents_tutorials');
    const course = 'sources/courses/genai-agents';
    const { sha256 } = JSON.parse(await readFile(join(rootDir,
      'publishing/tests/fixtures/genai-notebook-media-sha256.json'), 'utf8')) as {
        sha256: Record<string, string>;
      };
    expect(Object.keys(sha256)).toHaveLength(54);
    for (const [filename, hash] of Object.entries(sha256)) {
      expect(createHash('sha256').update(await readFile(join(root, filename))).digest('hex'), filename).toBe(hash);
    }
    const pages = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections
      .flatMap(section => section.pages).filter(page => page.route.startsWith(course + '/'));
    let diagrams = 0;
    let linkedScreenshots = 0;
    for (const { route } of pages) {
      const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
      expect(html).not.toContain('/_assets/Sources/GenAI Agents/');
      linkedScreenshots += (html.match(/View the screenshot in the original notebook \(cell \d+\)/g) ?? []).length;
      for (const [, src] of html.matchAll(/<img\b[^>]+\bsrc="([^"]+\/all_agents_tutorials\/[^"]*cell-\d+-(?:attachment|markdown-\d+)\.[^"]+)"/g)) {
        const filename = decodeURIComponent(src).split('/all_agents_tutorials/')[1];
        expect(await readFile(join(publicAssets, filename))).toEqual(await readFile(join(root, filename)));
        expect(html).toContain('Unmodified image.');
        diagrams++;
      }
    }
    expect(diagrams).toBe(8);
    expect(linkedScreenshots).toBe(7);
    const emitted = await readdir(publicAssets, { recursive: true });
    expect(emitted.filter(path => /cell-\d+-(?:attachment|markdown-\d+)\./.test(path))).toHaveLength(8);
    const images = emitted.filter(path => /\.(png|jpg|svg)$/.test(path));
    expect(images).toHaveLength(47);
    for (const filename of images) {
      expect(createHash('sha256').update(await readFile(join(publicAssets, filename))).digest('hex'), filename)
        .toBe(sha256[filename]);
    }
  });
  it('serves only reviewed local GenAI course figures and leaves remote screenshots as links', async () => {
    const course = 'sources/courses/genai-agents/';
    const prefix = '/assets/Sources/GenAI Agents/repository-figures/';
    const policy = JSON.parse(await readFile(join(rootDir, 'publishing/adapter/genai-course-figures.json'), 'utf8')) as {
      revision: string;
      license: { blob: string; sha256: string };
      figures: Record<string, { blob: string; sha256: string; stylesheetOmitted: boolean; staticViewport?: boolean }>;
    };
    const upstream = 'https://github.com/NirDiamant/GenAI_Agents/blob/' + policy.revision + '/';
    const license = await readFile(join(rootDir, 'Assets/Sources/GenAI Agents/repository-figures/LICENSE'));
    expect(createHash('sha1').update(`blob ${license.length}\0`).update(license).digest('hex')).toBe(policy.license.blob);
    expect(createHash('sha256').update(license).digest('hex')).toBe(policy.license.sha256);
    const pages = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections
      .flatMap(section => section.pages).filter(page => page.route.startsWith(course));
    const seen = new Set<string>();
    for (const { route } of pages) {
      const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
      const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
      expect(main, route).not.toMatch(/<img\b[^>]*\b(?:src|srcset)=["'](?:https?:)?\/\//i);
      for (const [, src] of main.matchAll(/<img\b[^>]*\bsrc="([^"]+\/repository-figures\/[^"]+)"/g)) {
        const filename = decodeURIComponent(src).split('/').at(-1)!;
        expect(policy.figures[filename], filename).toBeDefined();
        seen.add(filename);
        expect(main).toContain('href="' + upstream + 'images/' + filename + '"');
        expect(main).toContain('href="' + upstream + 'LICENSE"');
        expect(main).toContain('Nir Diamant and GenAI Agents contributors');
        if (policy.figures[filename].staticViewport) {
          expect(main).toContain('The viewport now fits the whole graph.');
        } else if (policy.figures[filename].stylesheetOmitted) {
          expect(main).toContain('External stylesheet reference omitted; drawing and labels unchanged.');
        }
      }
    }
    expect([...seen].sort()).toEqual(Object.keys(policy.figures).sort());
    expect(seen.size).toBe(16);
    const emitted = await readdir(join(distDir, prefix));
    expect(emitted.sort()).toEqual([...seen].sort());
    for (const [filename, row] of Object.entries(policy.figures)) {
      const bytes = await readFile(join(distDir, prefix, filename));
      expect(createHash('sha256').update(bytes).digest('hex'), filename).toBe(row.sha256);
      if (row.stylesheetOmitted) {
        expect(bytes.toString()).not.toMatch(/cdnjs\.cloudflare\.com|@import|<\?xml-stylesheet|<script\b/i);
        if (row.staticViewport) {
          expect(bytes.toString()).toContain('viewBox="0 0 1440 1566"');
          expect(bytes.toString()).not.toContain('id="svg-pan-zoom-controls"');
        }
      } else {
        expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')).toBe(row.blob);
      }
    }
    const scientific = await readFile(join(distDir, course, 'scientific-paper-agent-langgraph/index.html'), 'utf8');
    expect(scientific).toContain('href="https://i.ibb.co/y4Zf4Pc/Screenshot-2024-11-17-at-21-40-21.png"');
    const mcp = await readFile(join(distDir, course, 'mcp-tutorial/index.html'), 'utf8');
    expect(mcp).toContain('href="' + upstream + 'images/Claude_Desktop_with_MCP.png"');
    const art = await readFile(join(distDir, course, 'art-agent/index.html'), 'utf8');
    expect(art).toMatch(/<a href="https:\/\/colab\.research\.google\.com\/github\/[^"]+"[^>]*>Open In Colab<\/a>/i);
  });
  it('renders the Efficient DL Systems archive without hotlinks or a duplicate trace', async () => {
    const course = 'sources/courses/efficient-dl-systems/';
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    const policy = JSON.parse(await readFile(join(rootDir, 'publishing/adapter/edls-course-figures.json'), 'utf8')) as {
      revision: string;
      courseTraces: Record<string, { sha256: string }>;
      reusedOriginals: Record<string, string>;
    };
    const sources = 'https://github.com/mryab/efficient-dl-systems/blob/' + policy.revision + '/';
    const pages = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections.flatMap(section => section.pages)
      .filter(page => page.route.startsWith(course));
    const assets = { ...policy.reusedOriginals, ...Object.fromEntries(Object.entries(policy.courseTraces)
      .map(([name, entry]) => [name, entry.sha256])) };
    const seen = new Map<string, number>();
    let decks = 0;
    let notebooks = 0;
    for (const { route } of pages) {
      const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
      const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
      expect(main, route).not.toMatch(/<(?:img|iframe|video|source)\b[^>]*\b(?:src|srcset)=["'](?:https?:)?\/\//i);
      if (main.includes('Open the original slides (PDF)')) {
        expect(main).toMatch(new RegExp('href="' + sources + '[^"]+\\.pdf">Open the original slides \\(PDF\\)'));
        expect(main).toContain('Extracted slide text');
        decks++;
      }
      notebooks += [...main.matchAll(/href="https:\/\/colab\.research\.google\.com\/github\/mryab\/efficient-dl-systems\/blob\/([^/]+)\/[^"]+"[^>]*>Open in Colab/g)]
        .filter(([, rev]) => rev === policy.revision).length;
      for (const [, src] of main.matchAll(/<img\b[^>]*\bsrc="([^"]+)"[^>]*>/g)) {
        const filename = decodeURIComponent(src).split('/').at(-1)!;
        if (!Object.hasOwn(assets, filename)) continue;
        expect(src).toMatch(new RegExp('^' + base + '/assets/'));
        const bytes = await readFile(join(distDir, decodeURIComponent(src.slice(base.length))));
        expect(createHash('sha256').update(bytes).digest('hex'), filename).toBe(assets[filename]);
        seen.set(filename, (seen.get(filename) ?? 0) + 1);
      }
      if (route.endsWith('/week06-dl-arithmetic/seminar')) {
        expect(main).toContain('the same image for the compiled run');
        expect(main).toContain('href="' + sources + 'week06_dl_arithmetic/seminar/cross_entropy/profile_compiled_ce.py"');
        expect(main).toContain('href="https://arxiv.org/html/2410.10989v3#S3.F1"');
        expect(main).not.toMatch(/<img\b[^>]*src="[^"]*compiled_ce\.png/);
      }
      if (route.endsWith('/week02-fast-pipelines/seminar')) {
        expect(main).toContain('href="#edls-fp8-scaling"');
        expect(main).toContain('Each real MXFP8 block has 32 values');
        expect(main).toContain('NVIDIA Transformer Engine contributors');
      }
    }
    expect(decks).toBe(10);
    expect(notebooks).toBe(3);
    expect(seen).toEqual(new Map(Object.keys(assets).map(name => [name, 1])));
  });
  it('serves the pinned Inria glossary figures and credits the separate Palmer Penguins artwork', async () => {
    const course = 'sources/courses/scikit-learn-mooc';
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    const prefix = '/assets/Sources/Scikit-learn MOOC/figures/';
    const { commit, blobs } = JSON.parse(await readFile(join(rootDir,
      'publishing/tests/fixtures/scikit-learn-mooc-blobs.json'), 'utf8'));
    const upstream = 'https://github.com/INRIA/scikit-learn-mooc/blob/' + commit + '/';
    const figures = ['lin_separable.svg', 'dt_fit.svg', 'boosting0.svg', 'linear_fit_red.svg', 'api_diagram-predictor.fit.svg'];
    const figureDir = join(distDir, prefix);
    const emitted = await readdir(figureDir, { withFileTypes: true, recursive: true });
    const courseMedia = emitted.filter(entry => entry.isFile());
    expect(courseMedia.length).toBeGreaterThanOrEqual(159);
    for (const entry of courseMedia) {
      const file = join(entry.parentPath, entry.name);
      const sourcePath = 'figures/' + relative(figureDir, file).split(sep).join('/');
      expect(blobs[sourcePath], sourcePath).toMatch(/^[a-f0-9]{40}$/);
      const bytes = await readFile(file);
      expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex'), sourcePath)
        .toBe(blobs[sourcePath]);
    }
    const getPage = async (page: string) => {
      const alias = await readFile(join(distDir, 'en', course, page, 'index.html'), 'utf8');
      expect(alias).toContain(`0;url=${base}/${course}/${page}/`);
      const html = await readFile(join(distDir, course, page, 'index.html'), 'utf8');
      return html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
    };
    const glossary = await getPage('jupyter-book/appendix/glossary');
    const imgs = [...glossary.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*\salt="([^"]+)"/g)];
    expect(imgs).toHaveLength(figures.length);
    for (const [index, file] of figures.entries()) {
      const [, src, alt] = imgs[index];
      expect(decodeURIComponent(src)).toBe(base + prefix + file);
      expect(alt).not.toBe('img');
      expect(glossary).toContain('href="' + upstream + 'figures/' + file + '"');
      await expect(access(join(figureDir, file))).resolves.toBeUndefined();
    }
    const trees = await getPage('python-scripts/trees-dataset');
    expect(trees).not.toContain('/raw/main/man/figures/');
    expect(trees.match(/Artwork by @allison_horst/g)).toHaveLength(2);
    const art = JSON.parse(await readFile(join(rootDir, 'publishing/adapter/scikit-mooc-palmerpenguins.json'), 'utf8')) as {
      revision: string;
      figures: Record<string, { anchor: string; sha256: string }>;
    };
    const imgsByName = new Map([...trees.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*\salt="([^"]+)"/g)]
      .map(([, src, alt]) => [src.split('/').at(-1), { src: decodeURIComponent(src), alt }]));
    for (const [file, row] of Object.entries(art.figures)) {
      const src = '/assets/Sources/Scikit-learn MOOC/palmerpenguins/' + file;
      expect(imgsByName.get(file)?.src).toBe(base + src);
      expect(imgsByName.get(file)?.alt).toBeTruthy();
      expect(createHash('sha256').update(await readFile(join(distDir, src))).digest('hex')).toBe(row.sha256);
      expect(trees).toContain('href="https://allisonhorst.github.io/palmerpenguins/articles/art.html#' + row.anchor + '"');
      expect(trees).toContain('href="https://raw.githubusercontent.com/allisonhorst/palmerpenguins/' + art.revision + '/man/figures/' + file + '"');
    }
    expect(trees.match(/Teaching-use permission/g)).toHaveLength(2);
  });
  it('renders scikit-learn MOOC figures and presenter notes without loading third-party iframes', async () => {
    const course = 'sources/courses/scikit-learn-mooc/';
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    const pages = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections
      .flatMap(section => section.pages).filter(page => page.route.startsWith(course));
    let videos = 0, slides = 0;
    for (const { route } of pages) {
      const html = await readFile(join(distDir, route.replace(/\/index$/, ''), 'index.html'), 'utf8');
      const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
      expect(main, route).not.toMatch(/<(?:img|iframe|video|source)\b[^>]*\b(?:src|srcset)=["'](?:https?:)?\/\//i);
      videos += (main.match(/>Watch the course video on YouTube<\/a>/g) ?? []).length;
      slides += (main.match(/>Read the slides and speaker notes<\/a>/g) ?? []).length;
    }
    expect(videos).toBe(13);
    expect(slides).toBe(10);
    const bagging = await readFile(join(distDir, course, 'jupyter-book/ensemble/bagging-slides/index.html'), 'utf8');
    expect(bagging).toContain('href="https://www.youtube.com/watch?v=SnvdnIlOsHQ"');
    expect(bagging).toContain('href="' + base + '/' + course + 'slides/bagging/"');
    expect(bagging).not.toContain('first click on the slides');
    const deck = await readFile(join(distDir, course, 'slides/bagging/index.html'), 'utf8');
    const main = deck.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
    expect(main).not.toMatch(/(?:\.pull-left\[|\.column1\[|class: titlepage|\?\?\?)/);
    expect(main).toMatch(/<summary>Speaker notes<\/summary>[\s\S]*?separating circles from squares/);
    expect(main).toMatch(/<img\b[^>]*src="[^"]*bagging_reg_data\.svg"/);
    expect(main).toMatch(/<h2 id="bagging-for-classification-1">/);
    expect(deck).toContain('href="#bagging-for-classification-1"');
    const crossValidation = await readFile(join(distDir, course, 'slides/intro-cross-validation/index.html'), 'utf8');
    expect(crossValidation).toMatch(/<img\b[^>]*src="[^"]*test_scores_h_shuffle=false\.svg"/);
    expect(crossValidation).not.toContain('test_scores_h_shuffle%3Dfalse.svg');
  });
  it('renders the mlcourse.ai PCA and clustering figures, code and results together', async () => {
    const route = 'sources/courses/advanced-unsupervised-classical/mlcourse-topic7';
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
    const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)![1];
    const provenance = JSON.parse(await readFile(join(rootDir, 'publishing/adapter/mlcourse-topic7-evidence.json'), 'utf8')) as {
      plots: Record<string, { asset: string }>; thirdParty: Record<string, { sha256: string }>;
    };
    const expected = new Map(Object.entries(provenance.plots).map(([hash, {asset}]) => [asset, hash]));
    expected.set('GaussianScatterPCA.svg', provenance.thirdParty.pca.sha256);
    expected.set('spectral-clustering-comparison.png', provenance.thirdParty.clustering.sha256);
    const found = new Set<string>();
    for (const [, src, alt] of main.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*\salt="([^"]+)"/g)) {
      const filename = src.split('/').at(-1)!;
      expect(src).toMatch(new RegExp('^' + base + '/assets/'));
      expect(alt).not.toContain('_images/');
      expect(expected.has(filename), filename).toBe(true);
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src.slice(base.length))))).digest('hex'))
        .toBe(expected.get(filename));
      found.add(filename);
    }
    expect(found).toEqual(new Set(expected.keys()));
    expect(main).not.toMatch(/<(?:img|iframe|video|source)\b[^>]*\b(?:src|srcset)=["'](?:https?:)?\/\//i);
    expect(main).toMatch(/<table>[\s\S]*?0\.794003[\s\S]*?<\/table>/);
    expect(main).toContain('test rows influence the projection');
    expect(main).toContain('known digit labels');
    expect((main.match(/class="katex-display"/g) ?? []).length).toBe(12);
    expect(main).not.toContain('katex-error');
    const alias = await readFile(join(distDir, 'en', route, 'index.html'), 'utf8');
    expect(alias).toContain('0;url=' + base + '/' + route + '/');
  });
  it('reconciles the visible Harvard library images against the chapter and slide manifests', () => {
    const media = auditBuiltHarvardMedia(distDir, process.env.PUBLICATION_BASE_PATH);
    expect(media.errors).toEqual([]);
    expect(media.coursePages).toBeGreaterThan(0);
    expect(media.visibleCourseMedia).toBe(media.matching.chapters + media.matching.slides);
    expect(media.matching.chapters).toBeGreaterThan(0);
    expect(media.matching.slides).toBeGreaterThan(0);
  });
  it('links the six credited Harvard originals and embeds only the reviewed replacement figures', async () => {
    const audit = JSON.parse(await readFile(join(rootDir,
      'docs/audit/2026-11-11-harvard-credited-illustrations-additions.json'), 'utf8')) as {
        placements: { route: string; archived: string; archive_url: string; replacement: string | null }[];
        figures: { asset: string; sha256: string }[];
      };
    expect(audit.placements).toHaveLength(6);
    for (const placement of audit.placements) {
      const html = await readFile(join(distDir, placement.route, 'index.html'), 'utf8');
      expect(html).toContain(`href="${placement.archive_url}"`);
      const images = [...html.matchAll(/<img[^>]+src="([^"]+)"/g)].map(match => decodeURIComponent(match[1]));
      const file = placement.archived.split('/').at(-1);
      expect(images.some(src => src.endsWith('/' + file))).toBe(false);
      if (placement.replacement) {
        const target = images.find(src => src.endsWith('/' + placement.replacement));
        expect(target, placement.route).toBeDefined();
        const expected = audit.figures.find(figure => figure.asset === placement.replacement);
        expect(createHash('sha256').update(await readFile(join(distDir, target!))).digest('hex'))
          .toBe(expected!.sha256);
      }
      for (const path of [placement.archived, placement.archived.replace(/^Assets\//, '')]) {
        await expect(readFile(join(distDir, 'assets', path))).rejects.toMatchObject({ code: 'ENOENT' });
      }
    }
  });
  it('opens the selected authored Markdown on GitHub instead of a generated page', async () => {
    const base = 'https://github.com/alpotekhin/bookvar/edit/dev/';
    const entries = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections.flatMap(section => section.pages);
    for (const [locale, route] of [
      ['', 'textbook/models/gpt-1'],
      ['en/', 'textbook/models/gpt-1'],
      ['en/', 'textbook/modern-llm/rope'],
      ['en/', 'practice/rope'],
    ]) {
      const entry = entries.find(page => page.route === route)!;
      const source = locale === 'en/' ? entry.sourceEn ?? entry.source : entry.source;
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const editLinks = Array.from(html.matchAll(/<a\b[^>]*\bhref="(https:\/\/github\.com\/alpotekhin\/bookvar\/edit\/dev\/[^"]+)"/g));
      expect(editLinks, locale + route).toHaveLength(1);
      expect(decodeURIComponent(editLinks[0][1].slice(base.length))).toBe(source);
      expect(html).toContain(locale === 'en/' ? 'Edit page' : 'Редактировать страницу');
    }
    const library = await readFile(join(distDir, 'sources/courses/berkeley-advanced-llm-agents-spring-2025/index.html'), 'utf8');
    expect(library).not.toContain(base);
    const protectedArchive = join(distDir, 'sources/courses/berkeley-advanced-llm-agents-spring-2025/Lectures');
    for (const file of ['meeting-06-slides.pdf', 'meeting-06-slides.pages.txt']) {
      await expect(access(join(protectedArchive, file))).rejects.toMatchObject({ code: 'ENOENT' });
    }
  });

  it('renders the three licensed scaling originals instead of the course crops', async () => {
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, 'textbook/pretraining/scaling-laws/index.html'), 'utf8');
      for (const file of ['repeated-data-jmlr-figure1.png', 'mup-sp-vs-mup.png', 'step-law-figure1.png'])
        expect(html).toContain('/scaling-source-2026-11-10/' + file + '"');
      for (const file of ['repeated-data-effective-tokens-chart.png', 'learning-rate-scaling-chart.png', 'optimizer-search-surface-chart.png'])
        expect(html).not.toContain(file);
      expect(html).toContain('https://jmlr.org/papers/volume26/24-1000/24-1000.pdf#page=2');
      expect(html).toContain('/licenses/mup-19814971-MIT.txt');
      expect(html).toContain('https://step-law.github.io/');
      expect(html).toContain('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(html).not.toContain('katex-error');
    }
  });
  it('publishes the original cooldown experiment and links to the exact historical plots', async () => {
    const figures = [
      ['hagele-cosine-optimality.png', 'S2.F1', '4f9efb64de395b5ba5da945498bb0ebf29955f7590eac5573f34ed3edf545d13'],
      ['hagele-scaling-comparison.png', 'S5.F12', '32dd70de469761e452f1e56577cd440f769723098cfe42f07c4b26c29977e725'],
    ];
    const historical = [
      ['data-power-law-chart.png', 'https://arxiv.org/html/2001.08361v1#S1.F1'],
      ['parameter-count-convention-chart.png', 'https://arxiv.org/html/2001.08361v1#S3.F6'],
      ['critical-batch-vs-loss-chart.png', 'https://arxiv.org/html/2001.08361v1#S5.F10'],
      ['minicpm-cosine-campaign-cost-chart.png', 'https://arxiv.org/html/2203.15556v1#A2.F1'],
      ...['deepseek-lr-batch-grid-chart.png', 'deepseek-hyperparameter-fits.png',
        'deepseek-isoflop-curves-chart.png', 'deepseek-loss-prediction-chart.png']
        .map((file, i) => [file, 'https://arxiv.org/html/2401.02954v1#S3.F' + (i + 2)]),
      ['optimizer-extrapolation-failure.png', 'https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong'],
    ];
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, 'textbook/pretraining/scaling-laws/index.html'), 'utf8');
      for (const [file, figure, digest] of figures) {
        const path = [...html.matchAll(/<img\b[^>]*\bsrc="([^"]+)"[^>]*>/g)]
          .map(match => match[1]).find(path => path.endsWith('/' + file));
        expect(path).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(path!)))).digest('hex')).toBe(digest);
        expect(html).toContain('https://arxiv.org/html/2405.18392v3#' + figure);
      }
      for (const [file, link] of historical) {
        expect(html).not.toContain(file);
        expect(html).toContain('href="' + link + '"');
      }
      expect(html).toContain('id="step-law-grid"');
      expect(html).toContain('2{,}50625');
      expect(html).toContain('11,18');
      expect(html).not.toContain('katex-error');
    }
  });
  it('publishes both licensed DistServe panels in the benchmarking chapter in each locale', async () => {
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, 'textbook/inference/benchmarking-slo/index.html'), 'utf8');
      expect(html).toMatch(/<img[^>]+src="[^"]*\/distserve-original-teaser\.svg"[^>]*alt="DistServe:/);
      expect(html).not.toContain('distserve-figure1-original.png');
      expect(html).toContain('https://arxiv.org/html/2401.09670v3#S1.F1');
      expect(html).toContain('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(html).not.toContain('katex-error');
    }
  });
  it('connects the four Gemma generations to original figures and the worked configuration note', async () => {
    for (const locale of ['', 'en']) {
      const get = (page: string) => readFile(join(distDir, locale, page, 'index.html'), 'utf8');
      const [atlas, note, timeline, index] = await Promise.all([
        'models/semeystva/gemma', 'sources/papers/gemma-3-4', 'models/timeline', 'sources',
      ].map(get));
      for (const n of [144, 624, 832, 276]) expect(atlas).toContain(n + ' MiB');
      for (const name of ['original-eval_chart.svg', 'gemma-3-kv-cache.svg'])
        expect(atlas).toContain('/gemma/' + name + '"');
      expect(atlas).not.toContain('gemma-capability-comparison.png');
      expect(note).toContain('/gemma/gemma-4-mtp.png"');
      expect(note).toContain('https://arxiv.org/html/2607.02770v1#S2.F1');
      expect(atlas).not.toContain('katex-error');
      expect(note).not.toContain('katex-error');
      for (const anchor of ['gemma-cache-comparison', 'gemma4-kv-sharing', 'gemma4-mtp']) {
        expect(atlas).toMatch(new RegExp('href="/(?:en/)?sources/papers/gemma-3-4/#' + anchor + '"'));
        expect(note).toContain('id="' + anchor + '"');
      }
      expect(timeline).toMatch(/href="\/(?:en\/)?models\/semeystva\/gemma\/"/);
      expect(timeline).toMatch(/href="\/(?:en\/)?sources\/papers\/gemma-3-4\/#gemma4-vision-audio"/);
      expect(index).toMatch(/href="\/(?:en\/)?sources\/papers\/gemma-3-4\/"/);
    }
  });
  it('publishes licensed paper figures and routes Self-RAG to its canonical explanation', async () => {
    for (const locale of ['', 'en']) {
      const get = (name: string) => readFile(join(distDir, locale, 'sources/papers', name, 'index.html'), 'utf8');
      const [gemma, rag, astra] = await Promise.all(['gemma', 'self-rag', 'astra'].map(get));
      for (const [page, files] of [
        [gemma, ['eval_chart.svg', 'gemma_by_model_size_final.svg', 'gemma_both_by_corpus_sensitive_final.svg', 'gemma_both_by_corpus_amem_final.svg']],
        [astra, ['sft-pipeline.png', 'env.png']],
      ] as const) {
        for (const file of files) expect(page).toContain('/original-' + file + '"');
      }
      expect(gemma).toContain('https://arxiv.org/html/2403.08295v4#S6.F2');
      expect(gemma.indexOf('original-gemma_by_model_size_final.svg')).toBeGreaterThan(gemma.indexOf('Что измеряли в тестах на запоминание'));
      expect(rag).toMatch(/href="\/(?:en\/)?textbook\/rag\/full-pipeline\/#rag-self-rag"/);
      expect(rag).toContain('Порог задаёт система, а не модель');
      expect(astra).toContain('/licenses/astra-bdf6a46-Apache-2.0.txt');
      expect(astra).toContain('https://github.com/LianjiaTech/astra/blob/bdf6a46a130118e039aadcaeb46724d278383589/assets/env.png');
    }
  });
  it('connects the worked Self-RAG example and its two original figures from both reader indexes', async () => {
    for (const locale of ['', 'en']) {
      const get = (page: string) => readFile(join(distDir, locale, page, 'index.html'), 'utf8');
      const [chapter, ref] = await Promise.all([get('textbook/rag/full-pipeline'), get('reference')]);
      const begin = chapter.indexOf('<a id="rag-self-rag"></a>');
      const end = chapter.indexOf('<a id="rag-openscholar"></a>');
      expect(begin).toBeGreaterThan(chapter.indexOf('<a id="rag-conflicting-sources"></a>'));
      expect(end).toBeGreaterThan(begin);
      const section = chapter.slice(begin, end);
      for (const asset of ['teaser_self_rag_v8.svg', 'training_examples.svg'])
        expect(section).toContain('/original-' + asset + '"');
      for (const refId of ['S1.F1', 'S3.F2'])
        expect(section).toContain('https://arxiv.org/html/2310.11511v1#' + refId);
      expect(section).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(section).toContain('run_long_form_static.py#L211-L244');
      expect(section).toContain('finetune.py#L250-L291');
      expect(section).not.toContain('katex-error');
      expect(section).toMatch(/href="\/(?:en\/)?sources\/papers\/self-rag\/"/);
      expect(ref).toMatch(/href="\/(?:en\/)?textbook\/rag\/full-pipeline\/#rag-self-rag"/);
    }
  });
  it('uses the canonical source image in the GPT and k1.5 atlas reader routes', async () => {
    for (const locale of ['', 'en']) {
      const gpt = await readFile(join(distDir, locale, 'models/families/gpt/index.html'), 'utf8');
      const kimi = await readFile(join(distDir, locale, 'models/semeystva/kimi/index.html'), 'utf8');
      expect(gpt).toMatch(/<img[^>]+src="[^"]*\/gpt-30\/gpt2-transformer-block-vectors-2\.png"/);
      expect(gpt).not.toContain('gpt-decoder-block.png');
      expect(gpt).toContain('https://jalammar.github.io/illustrated-gpt2/#a-journey-up-the-stack');
      for (const anchor of ['gpt1-decoder-architecture', 'gpt2-pre-norm', 'gpt3-full-local-attention', 'gpt-llama-comparison'])
        expect(gpt).toMatch(new RegExp('href="/(?:en/)?textbook/[^"]+/#' + anchor + '"'));
      expect(kimi).toMatch(/<img[^>]+src="[^"]*\/kimi-rl-system\.svg"/);
      expect(kimi).not.toContain('kimi-k15-rl-system.png');
      expect(kimi).toContain('https://arxiv.org/html/2501.12599v1#S2.F3.sf1');
      expect(kimi).toMatch(/href="\/(?:en\/)?textbook\/post-training\/grpo-deepseek-r1\/#grpo-kimi-partial-rollouts"/);
    }
  });
  it('connects the Kimi generations, original diagram and four worked mechanisms', async () => {
    const route = 'sources/papers/kimi-k3-k31/';
    for (const locale of ['', 'en']) {
      const read = (p: string) => readFile(join(distDir, locale, p, 'index.html'), 'utf8');
      const [atlas, paper, timeline, library] = await Promise.all(
        ['models/semeystva/kimi', route, 'models/timeline', 'sources'].map(read),
      );
      for (const anchor of ['kimi-k3-kda', 'kimi-k3-attnres', 'kimi-k3-latent-moe', 'kimi-k31-sparse-mla']) {
        expect(atlas).toContain(`href="/${route}#${anchor}"`);
        expect(paper).toContain(`id="${anchor}"`);
      }
      expect(atlas).toMatch(/<img[^>]*src="[^"]*\/figure-2-architecture\.png"[^>]*alt="[^"]+"/);
      expect(atlas).toContain('https://arxiv.org/html/2607.24653v1#S2.F2');
      expect(atlas).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
      expect(paper).toContain('href="/models/semeystva/kimi/#kimi-k3-architecture"');
      for (const value of ['22,125 МиБ', '5,25 МиБ', 'Quantile Balancing', '2048', 'mla_use_nope'])
        expect(paper).toContain(value);
      expect(paper).not.toContain('katex-error');
      expect(timeline).toContain(`href="/${route}"`);
      expect(library).toContain(`href="/${route}"`);
    }
  });
  it('places the worked block-routing comparison between DSA and main-cache compression', async () => {
    const chapter = 'textbook/modern-llm/long-context/';
    const note = 'sources/papers/qwen3-8-flash-next/';
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, chapter, 'index.html'), 'utf8');
      const start = html.indexOf('<a id="block-routing-moba-qsa"></a>');
      const end = html.indexOf('<a id="compressed-sparse-attention-csa-hca"></a>');
      const dsa = html.indexOf('<a id="dsa-content-adaptive-selection"></a>');
      expect(dsa).toBeGreaterThan(0);
      expect(start).toBeGreaterThan(dsa);
      expect(end).toBeGreaterThan(start);
      const section = html.slice(start, end);
      expect(section).toMatch(/<img[^>]*src="[^"]*\/moba-running-example\.png"[^>]*alt="[^"]+"/);
      expect(section).toContain('/licenses/moonshot-moba-b5d58363-MIT.txt');
      expect(section).toContain('https://arxiv.org/html/2608.30320v1#S2.F3');
      expect(section).toContain(`href="${locale ? '/en' : ''}/${note}#qsa-block-selection"`);
      expect(section).toContain('8192 сводных ключа');
      expect(section).not.toContain('katex-error');
      const paper = await readFile(join(distDir, locale, note, 'index.html'), 'utf8');
      expect(paper).toContain(`href="/${chapter}#block-routing-moba-qsa"`);
    }
  });
  it('connects the Flash-Next report to the atlas, chronology, mechanism map and source library', async () => {
    const route = 'sources/papers/qwen3-8-flash-next/';
    for (const locale of ['', 'en']) {
      const read = (p: string) => readFile(join(distDir, locale, p, 'index.html'), 'utf8');
      const [paper, atlas, timeline, map, library] = await Promise.all(
        [route, 'models/semeystva/qwen', 'models/timeline', 'models/sravneniya/karta-arhitekturnyh-liniy', 'sources'].map(read),
      );
      for (const anchor of ['qsa-block-selection', 'qwen-gated-residual', 'qwen-ngram-lookup']) {
        expect(atlas).toContain(`href="/${route}#${anchor}"`);
        expect(paper).toContain(`id="${anchor}"`);
      }
      expect(timeline).toContain(`href="/${route}"`);
      expect(library).toContain(`href="/${route}"`);
      expect(map).toContain(`href="/${route}#qsa-block-selection"`);
      expect(paper).toContain('href="/models/semeystva/qwen/#qwen38-flash-next"');
      for (const value of ['0–3, 8–11 и 12–14', '8192', '0,75 GiB', '24 MiB', '108 MiB', '5120 байт',
        'https://arxiv.org/html/2608.30320v1#S2.F3', 'https://arxiv.org/html/2601.07372v1#S2.F2',
        'https://creativecommons.org/licenses/by/4.0/']) expect(paper).toContain(value);
      expect(paper).toMatch(/<img[^>]*src="[^"]*\/engram-system\.svg"[^>]*alt="[^"]+"/);
      expect(paper).not.toContain('katex-error');
    }
  });
  it('shows both cache components and both author attention maps on the Qwen family page', async () => {
    for (const locale of ['', 'en']) {
      const atlas = await readFile(join(distDir, locale, 'models/semeystva/qwen/index.html'), 'utf8');
      const section = atlas.split('<a id="qwen3-next-comparison"></a>')[1];
      expect(section).toBeDefined();
      for (const term of ['0,75 GiB', '72 MiB', '48 таких слоёв кэша', '0{,}4,1{,}8',
        'Это не карты 80B', 'https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-Next/model_architecture.png',
        'https://arxiv.org/html/2505.09388v1#S4.F1',
        'https://arxiv.org/html/2505.06708v1#S2',
        '/licenses/qiu-gated-attention-f4c2a5f6-MIT.txt']) expect(section).toContain(term);
      for (const figure of ['baseline_selected_layer_attention_maps', 'gate_elementwise_selected_layer_attention_maps'])
        expect(section).toMatch(new RegExp(`<img[^>]*src="[^"]*/${figure}\\.png"[^>]*alt="[^"]+"`));
      expect(atlas).not.toMatch(/<img[^>]+src="[^"]*(qwen-next-block-original|qwen-next-hybrid-architecture|qwen3-post-training-pipeline)\.(jpg|png)"/);
      expect(section).not.toContain('katex-error');
    }
  });
  it('separates the 72B Qwen architecture, base results and instruct results on the published route', async () => {
    const route = 'sources/papers/qwen-2-5-technical-report/';
    for (const locale of ['', 'en']) {
      const paper = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const atlas = await readFile(join(distDir, locale, 'models/semeystva/qwen/index.html'), 'utf8');
      // Both fallback links point to the canonical Russian text.
      expect(atlas).toContain(`href="/${route}#qwen-72b-architecture"`);
      expect(atlas).toContain(`href="/${route}#qwen25-pretraining-evidence"`);
      for (const page of [paper, atlas]) {
        expect(page).toContain('https://arxiv.org/html/2412.15115v2#S0.F1');
        expect(page).not.toMatch(/<img[^>]*src="[^"]*qwen25-figure1-hq\.png"/);
        expect(page).not.toContain('katex-error');
      }
      expect(paper).toMatch(/<img[^>]*src="[^"]*\/mha-gqa-mqa\.png"[^>]*alt="[^"]+"/);
      expect(paper).toContain('в центре одна пара на группу запросов; справа все головы делят одну пару');
      expect(paper).toContain('средняя (GQA)');
      expect(paper).toContain('https://arxiv.org/html/2305.13245v3#S2.F2');
      expect(paper).toContain('https://creativecommons.org/licenses/by/4.0/');
      const text = paper.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ');
      for (const claim of ['64 / 64', '64 / 8', '24 576', '29 568', '80 GiB', '10 GiB',
        '117,4 млн', '50,9', '62,1', '64,6', '59,1', 'Llama 3.1 405B', '84,1 против 86,0',
        '50,9 с YaRN и DCA и 27,0 без них', '8192 токена генерации']) {
        expect(text).toContain(claim);
      }
    }
  });
  it('renders the complete attributed DeepSeek originals instead of historical crops', async () => {
    const expected = [
      ['sources/papers/deepseek-v3-technical-report', ['v3-figure1-benchmarks.png', 'v3-figure2-block.png'], ['/licenses/deepseek-v3-d5c08b3-MIT.txt', '0{,}5625']],
      ['sources/papers/deepseek-r1-reasoning-via-rl', ['r1-zero-nature-figure1.png', 'figure2-pipeline.png'], ['65 536', '86,7%', 'https://www.nature.com/articles/s41586-025-09422-z/figures/1']],
      ['models/families/deepseek', ['v2-author-architecture.png', 'r1-zero-nature-figure1.png'], ['/licenses/deepseek-v2-ec98ee3c-MIT.txt', '/sources/papers/deepseek-v3-technical-report/#v3-block']],
      ['textbook/modern-llm/sequence-model-alternatives', ['v3-figure2-block.png'], ['/licenses/deepseek-v3-d5c08b3-MIT.txt']],
    ] as const;
    for (const [route, files, facts] of expected) {
      const page = await readFile(join(distDir, route, 'index.html'), 'utf8');
      for (const name of files) {
        const src = page.match(new RegExp('<img[^>]+src="([^"]*/' + name.replace('.', '\\.') + ')"[^>]*alt="[^"]+"'))?.[1];
        expect(src, route + ': ' + name).toBeDefined();
        await access(join(distDir, decodeURIComponent(src!)));
      }
      for (const fact of facts) expect(page, route).toContain(fact);
      expect(page).not.toMatch(/deepseek-(?:r1-figure1-hq|v3-figure1-hq|v3-figure2-architecture-hq|v2-mla-moe|v3-architecture)\.png/);
      expect(page).not.toContain('katex-error');
    }
  });
  it('does not load the imported GenAI Agents view counter in any notebook page', async () => {
    const prefix = 'sources/courses/genai-agents/';
    const pages = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections
      .flatMap(section => section.pages).filter(page => page.route.startsWith(prefix));
    const counter = 'europe-west1-genai-agents-views-tracker.cloudfunctions.net';
    let archivedCounters = 0;
    for (const page of pages) {
      const source = await readFile(join(rootDir, page.source), 'utf8');
      if (source.includes(counter)) archivedCounters++;
      const html = await readFile(join(distDir, page.route, 'index.html'), 'utf8');
      expect(html, page.route).not.toMatch(/<img\b[^>]*\bsrc="[^"]*genai-agents-views-tracker\.cloudfunctions\.net/);
    }
    expect(archivedCounters).toBe(53);
    const overview = await readFile(join(distDir, prefix, 'overview/index.html'), 'utf8');
    expect(overview).toContain('github.com/NirDiamant/GenAI_Agents');
    expect(overview).toContain('>PRs Welcome</a>');
    expect(overview).toContain('>LinkedIn</a>');
  });
  it('connects54 DDP/checkpointing subjects to complete chapters and nineteen source figures in both interfaces',async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-ddp-checkpoint-subject-additions.json'),'utf8'));
    const topics=[{route:'textbook/pretraining/processes-collectives-ddp',subjects:33,programs:15,figures:14,displays:16},
      {route:'textbook/pretraining/gradient-checkpointing-offload',subjects:21,programs:9,figures:5,displays:6}];
    for(const prefix of ['', 'en/']){
      const index=await readFile(join(distDir,prefix,'reference/index.html'),'utf8');
      for(const [i,topic] of topics.entries()){
        const raw=await readFile(join(rootDir,journal.files[i].source),'utf8');
        const metadata=YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)![1]);
        expect(metadata.reference_terms).toHaveLength(topic.subjects);
        const html=await readFile(join(distDir,prefix,topic.route,'index.html'),'utf8');
        for(const {anchor} of metadata.reference_terms){
          expect(html).toContain('id="'+anchor+'"');
          expect(index).toContain('href="/'+prefix+topic.route+'/#'+anchor+'"');
        }
        expect(html.match(/<pre data-language="python"/g)).toHaveLength(topic.programs);
        expect(html.match(/<img\b[^>]*\ssrc="/g)).toHaveLength(topic.figures);
        expect(html.match(/class="katex-display"/g)).toHaveLength(topic.displays);
        expect(html).not.toContain('katex-error');
        if(prefix)expectCanonicalEditionNotice(html);
      }
    }
  });
  it('connects all33 diffusion and selection subjects to full explanations in both interfaces',async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-diffusion-selection-subject-additions.json'),'utf8'));
    const topics=[{route:'textbook/pretraining/masked-diffusion-language-models',programs:4,figures:4},
      {route:'textbook/pretraining/selection-engineering-evaluation',programs:10,figures:5}];
    for(const prefix of ['', 'en/']){
      const index=await readFile(join(distDir,prefix,'reference/index.html'),'utf8');
      for(const [i,topic] of topics.entries()){
        const raw=await readFile(join(rootDir,journal.files[i].source),'utf8');
        const metadata=YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)![1]);
        expect(metadata.reference_terms).toHaveLength(i===0?13:20);
        const html=await readFile(join(distDir,prefix,topic.route,'index.html'),'utf8');
        for(const {anchor} of metadata.reference_terms){
          expect(html).toContain('id="'+anchor+'"');
          expect(index).toContain('href="/'+prefix+topic.route+'/#'+anchor+'"');
        }
        expect(html.match(/<pre data-language="python"/g)).toHaveLength(topic.programs);
        expect(html.match(/<img\b[^>]*\ssrc="/g)).toHaveLength(topic.figures);
        expect(html.match(/class="katex-display"/g)).toHaveLength(8);
        expect(html).not.toContain('katex-error');
        expect(html).not.toContain('This content is not available in your language yet.');
      }
    }
  });
  it('renders38 data-selection subjects,19 complete programs and nine original figures in both interfaces',async()=>{
    for(const prefix of ['', 'en/']){
      for(const [route,programs,figures,anchors] of [
        ['data-selection-scores',8,4,['el2n','grand','cl-class-thresholds','cl-joint-calibration','cl-paper-software']],
        ['active-learning',5,2,['annotation-loop','al-vote-entropy','al-consensus-disagreement']],
        ['geometric-coresets',6,3,['approximation-guarantee','grid-compression','herding','mean-error']]
      ] as const){
        const html=await readFile(join(distDir,prefix+'textbook/pretraining/'+route+'/index.html'),'utf8');
        expect(html.match(/<pre data-language="python"/g)).toHaveLength(programs);
        expect(html.match(/<img\b[^>]*\ssrc="/g)).toHaveLength(figures);
        expect(html).not.toContain('katex-error');
        for(const anchor of anchors)expect(html).toContain('id="'+anchor+'"');
      }
    }
  });
  it('publishes separate kernel/compiler chapters with all45 programs,16 figures and compatibility links in both interfaces',async()=>{
    for(const prefix of ['', 'en/']){
      const kernels=await readFile(join(distDir,prefix+'textbook/ml-systems/gpu-kernels-triton/index.html'),'utf8');
      const compiler=await readFile(join(distDir,prefix+'textbook/ml-systems/graph-compilation-pytorch-jax/index.html'),'utf8');
      expect(kernels.match(/<pre data-language="python"/g)).toHaveLength(10);
      expect(compiler.match(/<pre data-language="python"/g)).toHaveLength(35);
      expect(kernels.match(/<img\b[^>]*\ssrc="/g)).toHaveLength(11);
      expect(compiler.match(/<img\b[^>]*\ssrc="/g)).toHaveLength(5);
      expect(kernels).not.toContain('compile_records');
      expect(compiler).not.toContain('def chunked_linear_ce');
      expect(kernels).toContain('<summary>Перенесённые разделы о компиляции графов</summary>');
      expect(kernels).toContain('href="/'+prefix+'textbook/ml-systems/graph-compilation-pytorch-jax/#torch-compile-graphs-guards-recompilation"');
      expect(compiler).toContain('href="/'+prefix+'textbook/ml-systems/gpu-kernels-triton/"');
      for(const html of [kernels,compiler]){
        expect(html).not.toContain('katex-error');
        expect(html).not.toContain('<p>![[');
        if(prefix) expectCanonicalEditionNotice(html);
      }
      for(const anchor of ['compile-donation-changes-buffer-ownership',
        'compile-device-placement-and-collective-communication','cuda-graphs-capture-replay'])
        expect(compiler).toContain('id="'+anchor+'"');
    }
  });
  it('publishes the licensed replacements and all configuration data in both interfaces', async () => {
    const registry=YAML.parse(await readFile(join(rootDir,'05 Источники/asset-registry.yml'),'utf8'));
    const retiredAssets=registry.assets.filter((row: { publication_status?: string }) => row.publication_status === 'retired');
    expect(retiredAssets.map((row: { id: string }) => row.id)).toEqual(expect.arrayContaining([
      'curated-grpo-84b71426c8c2',
      'curated-gpt-3-paper-416b4bb6d87b',
      'curated-gpt-3-paper-table-12ca47fd68d1',
      'curated-vlm2026-clip-figure1-ac050e1cf247',
      'curated-vlm2026-llava-figure1-27391332e106',
      'curated-vlm2026-cs231n-winoground-d72312341628',
      'curated-stanford-cs336-multimodal-siglip-parallelism-21fbe73a2d33',
      'curated-stanford-cs336-multimodal-llava-anyres-ba1d449afcd3',
    ]));
    for(const prefix of ['', 'en/']){
      const gpt=await readFile(join(distDir,prefix+'textbook/models/gpt-3/index.html'),'utf8');
      const table=[...gpt.matchAll(/<table\b[\s\S]*?<\/table>/g)].map(x=>x[0]).find(x=>x.includes('GPT-3 Small'));
      expect(table).toBeDefined();
      expect(table!.match(/<th(?:\s|>)/g)).toHaveLength(8);
      expect(table!.match(/<td(?:\s|>)/g)).toHaveLength(64);
      for(const label of ['GPT-3 Small','GPT-3 Medium','GPT-3 Large','GPT-3 XL',
        'GPT-3 2.7B','GPT-3 6.7B','GPT-3 13B','GPT-3 175B']) expect(table).toContain(label);
      expect(gpt).toContain('gpt3-demonstrations-frame-171.png');
      expect(gpt).toContain('https://jalammar.github.io/images/gpt3/09-gpt3-generating-react-code-example.gif');
      expect(gpt).not.toContain('katex-error');
      const grpo=await readFile(join(distDir,prefix+'textbook/post-training/grpo-deepseek-r1/index.html'),'utf8');
      expect(grpo).toContain('ppo-framework.png');
      expect(grpo).toContain('group-advantages.png');
      expect(grpo).toContain('/licenses/rlhf-book-MIT.txt');
      if(prefix) expectCanonicalEditionNotice(grpo);
      for(const retired of retiredAssets){
        const name=retired.asset.split('/').at(-1)!;
        expect(gpt).not.toContain(name);
        expect(grpo).not.toContain(name);
      }
    }
    for(const retired of retiredAssets)
      await expect(access(join(distDir,'assets',retired.asset))).rejects.toThrow();
  });

  it('keeps Plan-Seq-Learn attribution optional while retaining the source explanation and figures', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + 'textbook/agents/memory-planning-orchestration/index.html'), 'utf8');
      const disclosure = html.match(/(<details[^>]*>)\s*<summary>Source text and attribution<\/summary>([\s\S]*?)<\/details>/);
      expect(disclosure).not.toBeNull();
      expect(disclosure![1].includes('open')).toBe(false);
      expect(disclosure![2].includes('Ниже сохранены')).toBe(true);
      expect(disclosure![2].includes('<blockquote>')).toBe(false);
      const start = html.indexOf('id="plan-seq-learn"');
      const end = html.indexOf('id="всегда-ли-дополнительный-планировщик-полезен"', start);
      const section = html.slice(start, end);
      for (const figure of ['method.png', 'nut-assembly.png', 'sequencing.png'])
        expect(section.includes('/planseqlearn-2026-10-26/' + figure)).toBe(true);
      for (const text of ['Planning Module: Zero-Shot High-level Planning',
        'Sequencing Module: Vision-based Plan Tracking', 'Learning Module: Efficiently Learning Local Control',
        'Stage Termination Details', 'There are several limitations of PSL'])
        expect(section.includes(text)).toBe(true);
      expect(html.includes('katex-error')).toBe(false);
      if (prefix) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the three modern-route corrections in both interfaces without dropping examples', async () => {
    for (const prefix of ['', 'en/']) {
      const ffn = await readFile(join(distDir, prefix + 'textbook/modern-llm/dense-ffn/index.html'), 'utf8');
      const long = await readFile(join(distDir, prefix + 'textbook/modern-llm/long-context/index.html'), 'utf8');
      const distributed = await readFile(join(distDir, prefix + 'textbook/pretraining/distributed-systems/index.html'), 'utf8');
      expect(ffn.includes('Краткие формулы собраны отдельно')).toBe(false);
      expect(long.includes('Все четыре блока NumPy')).toBe(false);
      expect(long).toContain('Примеры NumPy в главе можно');
      const prose = distributed.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
      expect(prose.includes('Предыдущая глава о масштабировании')).toBe(false);
      expect(distributed).toContain('Prerequisites');
      for (const route of ['modern-llm/transformer-shapes-resources',
        'ml-systems/transformer-moe-arithmetic', 'pretraining/scaling-laws'])
        expect(distributed).toContain('href="/' + prefix + 'textbook/' + route + '/"');
      for (const [html, count] of [[ffn, 3], [long, 6], [distributed, 3]] as const) {
        expect(html.match(/<pre data-language=/g)).toHaveLength(count);
        expect(html.includes('katex-error')).toBe(false);
        if (prefix) expectCanonicalEditionNotice(html);
      }
    }
  });
  it('keeps canonical-language textbook navigation in EN while labeling the body honestly', async () => {
    const source = await readFile(join(rootDir,
      'site/src/content/docs/generated/en/textbook/evaluation/models-contamination.md'), 'utf8');
    const data = YAML.parse(source.match(/^---\n([\s\S]*?)\n---/)![1]);
    expect(data.translationStatus).toBe('canonical');
    expect(data.canonicalHref).toBe('/textbook/evaluation/models-contamination/');
    expect(data.translationReviewed).toBeUndefined();
    const html = await readFile(join(distDir, 'en/textbook/evaluation/models-contamination/index.html'), 'utf8');
    expectCanonicalEditionNotice(html);
    for (const route of ['inference/benchmarking-slo', 'agents/evaluation',
      'pretraining/dedup-pii-corpus-quality', 'inference/decoding']) {
      expect(html).toContain('href="/en/textbook/' + route + '/"');
      expect(html).not.toContain('href="/textbook/' + route + '/"');
    }
    expect(html).toContain('href="/textbook/evaluation/models-contamination/"');
    expect(html).toContain('Две модели правильно ответили');
    expect(html).toContain('bert-score-recall.png');
  });
  it('shows one edition notice per textbook page, or none for current/native English content', async () => {
    const manifest = loadManifest(join(rootDir, 'publishing/navigation.yml'));
    const textbook = manifest.sections.find(section => section.id === 'textbook')!;
    for (const page of textbook.pages) {
      const file = page.route.replace(/\/index$/, '') + '/index.html';
      const raw = await readFile(join(rootDir, page.source), 'utf8');
      const canonical = YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
      const root = await readFile(join(distDir, file), 'utf8');
      const english = await readFile(join(distDir, 'en', file), 'utf8');
      expect(root, page.route).not.toContain('aria-label="English edition status"');
      expect(english, page.route).not.toContain('This content is not available in your language yet.');
      const rootChapterLinks = [...english.matchAll(
        /<a\b[^>]*\bhref="(\/textbook\/[^\"]*)"[^>]*>([\s\S]*?)<\/a>/g)]
        .filter(match => match[2].replace(/<[^>]*>/g, '').trim() !== 'Read the canonical chapter')
        .map(match => ({ href: match[1], text: match[2].replace(/<[^>]*>/g, '').trim() }));
      expect(rootChapterLinks, page.route + ': unexpected locale switch').toEqual([]);
      if (!page.sourceEn && canonical.source_language !== 'en') {
        expectCanonicalEditionNotice(english);
        const media = (text: string) => [...text.matchAll(
          /<(?:img|video|source)\b[^>]*\bsrc="([^"]+)"/g)].map(match => match[1]);
        const programs = (text: string) => [...text.matchAll(/<pre\b[^>]*>([\s\S]*?)<\/pre>/g)]
          .map(match => match[1].replace(/<[^>]*>/g, ''));
        expect(media(english), page.route + ': original figures/media').toEqual(media(root));
        expect(programs(english), page.route + ': published code').toEqual(programs(root));
        expect(english.match(/<math\b[\s\S]*?<\/math>/g) ?? [], page.route + ': formulas')
          .toEqual(root.match(/<math\b[\s\S]*?<\/math>/g) ?? []);
      } else if (!page.sourceEn) {
        expect(english, page.route).not.toContain('aria-label="English edition status"');
      } else {
        const text = await readFile(join(rootDir, page.sourceEn), 'utf8');
        const translation = YAML.parse(text.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
        const current = translation.translation_status === 'current'
          && translation.canonical_revision === createHash('sha256').update(raw).digest('hex');
        expect(english.match(/aria-label="English edition status"/g) ?? [], page.route)
          .toHaveLength(current ? 0 : 1);
        if (!current) expect(english, page.route).toContain('Partial English edition.');
      }
    }
    const other = manifest.sections.find(section => section.id === 'models')!.pages
      .find(page => !page.sourceEn)!;
    const outside = await readFile(join(distDir, 'en', other.route.replace(/\/index$/, ''), 'index.html'), 'utf8');
    expect(outside).toContain('This content is not available in your language yet.');
  });
  it('renders the complete nanoGPT worked reference in both existing practice routes', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'practice/20-sobrat-yazykovuyu-model-s-nulya/index.html'), 'utf8');
      const text = html.replace(/<[^>]+>/g, '');
      expect(html).toContain('id="nanogpt-worked-reference"');
      expect(html).toContain('nanogpt-2026-10-30/loss-original.png');
      expect(html).toContain('nanogpt-3adf61e1-MIT.txt');
      for (const fragment of ['96 training targets', '110,392-byte', '48 official tests',
        'Checkpoint contents and update boundaries', 'Original nanoGPT excerpt',
        'optimizer.zero_grad(set_to_none=True)', 'dtype']) {
        expect(text).toContain(fragment);
      }
      expect(html).not.toContain('katex-error');
    }
  });
  it('does not leak temporary code-protection markers into any published page', async () => {
    const files = (await readdir(distDir, { recursive: true })).filter(file => file.endsWith('index.html'));
    const leaked: string[] = [];
    for (const file of files) {
      if (/BOOKVARPROTECTEDCODE\d+TOKEN/.test(await readFile(join(distDir, file), 'utf8'))) leaked.push(file);
    }
    expect(leaked).toEqual([]);
  });
  it('keeps all eight complete-Transformer code blocks in both locales', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/transformer/full-transformer/index.html'), 'utf8');
      expect(html.match(/<pre\b/g)).toHaveLength(8);
      expect(html).toContain('id="transformer-dropout"');
      expect(html.replace(/<[^>]+>/g,'')).toContain('A surviving weight can exceed one');
      for (const name of ['TinyTranslation', 'TinyCausalLM', 'class EncoderBlock', 'class DecoderBlock'])
        expect(html.replace(/<[^>]+>/g, '')).toContain(name);
    }
  });
  it('renders complete BLEU, chrF and original BERTScore reasoning in both locale routes', async () => {
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'textbook/evaluation/models-contamination/index.html'),'utf8');
      for (const anchor of ['reference-based-text-evaluation','bleu-reference-overlap',
        'chrf-character-overlap','bertscore-token-alignment'])
        expect(html).toContain('id="'+anchor+'"');
      for (const value of ['bleu_from_tokens','Corpus BLEU','weighted_recall',
        'bert-score-recall.png','bertscore-19e7f551-MIT.txt','53.7285'])
        expect(html).toContain(value);
      expect(html.match(/<pre\b/g)).toHaveLength(13);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('<p>![');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders Coconut inside TTC, including the native diagram, complete example and mixed-language fallback', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/reasoning/test-time-compute/index.html'), 'utf8');
      expect(html).toContain('id="coconut-continuous-thought"');
      expect(html).toContain('coconut-2026-10-29/coconut.png');
      expect(html).toContain('coconut-27273cb8-MIT.txt');
      expect(html).toContain('coconut_linear_step');
      expect(html).toContain('coconut_stage');
      expect(html.match(/<pre\b/g)).toHaveLength(7);
      for (const anchor of ['tree-of-thoughts', 'tot-creative-writing', 'tot-crossword-dfs'])
        expect(html).toContain('id="' + anchor + '"');
      for (const file of ['tot-creative-search.png', 'tot-creative-results.png', 'tot-crossword-dfs.png'])
        expect(html).toContain('tot-reconciliation-2026-10-30/' + file);
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('serves all declared source-native English pages without pretending they need translation', async () => {
    const manifest = loadManifest(join(rootDir, 'publishing/navigation.yml'));
    const native = [];
    for (const page of manifest.sections.flatMap(section => section.pages)) {
      if (hseCatalogueEntry(page.source, page.route)) continue;
      const raw = await readFile(join(rootDir, page.source), 'utf8');
      const metadata = YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
      if (!page.sourceEn && metadata.source_language === 'en') native.push(page);
    }
    expect(native).toHaveLength(40);
    expect(native.some(page => page.route === 'sources/papers/deepspeed-fastgen')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/wildbench')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/length-controlled-alpacaeval')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/eagle-3')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/isoflop-parabola-bias')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/piecewise-linear-regions')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/what-in-context-learning-learns')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/prolong')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/contaccum')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/tied-language-model-architectures')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/delayed-conversion-feedback')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/implicit-feedback-weights-sampling')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/sequence-windows-sampled-evaluation')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/appworld')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/toolsandbox')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/gradcache')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/complexity-trap-context-management')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/lambdarank-lambdaloss')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/fedbuff')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/membership-memorization-audits')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/label-shift-certified-robustness')).toBe(true);
    expect(native.some(page => page.route === 'sources/papers/gender-shades-selective-labels')).toBe(true);
    for (const page of native) {
      const file = page.route.replace(/\/index$/, '') + '/index.html';
      const canonical = await readFile(join(distDir, file), 'utf8');
      const english = await readFile(join(distDir, 'en', file), 'utf8');
      expect(english, page.route).not.toContain('This content is not available in your language yet.');
      expect(english, page.route).not.toContain('Canonical-language chapter.');
      expect(english, page.route).not.toContain('Partial English edition.');
      expect(english.match(/<img\b[^>]*\bsrc=/g) ?? [], page.route)
        .toHaveLength((canonical.match(/<img\b[^>]*\bsrc=/g) ?? []).length);
      expect(english.match(/<pre\b/g) ?? [], page.route)
        .toHaveLength((canonical.match(/<pre\b/g) ?? []).length);
      expect(english.match(/class="katex-display"/g) ?? [], page.route)
        .toHaveLength((canonical.match(/class="katex-display"/g) ?? []).length);
      expect(english, page.route).not.toContain('katex-error');
    }
  });
  it('shows the original candidate matrix comparison before the GradCache example', async () => {
    const route = 'textbook/retrieval/embeddings-metric-learning/';
    const paper = 'sources/papers/contaccum/';
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="emb-gradcache"><\/a>[\s\S]*?(?=<a id="emb-simcse")/)?.[0];
      expect(section).toBeDefined();
      expect(section).toMatch(/<img[^>]+src="[^"]*\/contaccum-figure1\.png"[^>]+alt="Three panels:/);
      for (const value of ['0.7437', '0.3133', 'not zero-valued logits', 'five negatives',
        'https://arxiv.org/html/2406.12356v3#S2.F1', 'https://creativecommons.org/licenses/by/4.0/'])
        expect(section).toContain(value);
      expect(section).toContain(`href="/${locale}${paper}"`);
      const note = await readFile(join(distDir, locale, paper, 'index.html'), 'utf8');
      expect(note).toContain(`href="/${locale}${route}#emb-gradcache"`);
      expect(section).not.toContain('katex-error');
    }
  });
  it('publishes the same PEFT chapter, thirteen source figures and nineteen examples in both locale routes', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/fine-tuning-peft/index.html'), 'utf8');
      expect(html.match(/<h1(?:\s|>)/g)).toHaveLength(1);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(13);
      expect(html.match(/<pre\b/g)).toHaveLength(19);
      for (const name of ['houlsby-adapter.png', 'lora-animated.gif', 'lora-merge.png', 'lora-gpt3-figure2.png',
        'qlora-training-original.png', 'qlora-figure2-target-modules.svg',
        'qlora-figure6-training-memory.svg', 'prompt-tuning.png', 'prefix-tuning.png',
        'asapp-overview.png', 'mosbach-stability.png',
        'l2sp-target-comparison.png', 'l2sp-retention.png'])
        expect(html).toContain(name);
      for (const id of ['choosing-trainable-parameters', 'bottleneck-adapters', 'lora-update',
        'merging-and-serving', 'qlora-training', 'qlora-frozen-gradient-example', 'soft-prompts', 'prefix-keys-values',
        'small-data-fine-tuning', 'fine-tuning-bias-correction', 'fine-tuning-reinitialization',
        'fine-tuning-update-budget', 'intermediate-task-transfer',
        'retaining-capabilities', 'fine-tuning-replay', 'l2-sp',
        'elastic-weight-consolidation', 'fisher-for-ewc'])
        expect(html).toContain('id="' + id + '"');
      expect(html).toContain('href="/licenses/peft-fee923eb-Apache-2.0.txt"');
      expect(html).toContain('href="/licenses/asapp-finetuning-0aa4f4e-LICENSE.txt"');
      expect(html).toContain('href="/licenses/mosbach-finetuning-3cf27e8-Apache-2.0.txt"');
      expect(html).toContain('CC BY-NC-SA 4.0');
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders the same-budget GQA comparison and both original QLoRA plots', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/fine-tuning-peft/index.html'), 'utf8');
      const table = html.match(/id="lora-gqa-budget"[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/)?.[1];
      expect(table?.match(/3,407,872/g)).toHaveLength(2);
      expect(table).toContain('20,971,520');
      for (const [id, image, url, digest] of [
        ['qlora-target-placement', 'qlora-figure2-target-modules.svg', 'S4.F2', '6690d5f00e1708f7044b777d5635130e9b9e306fcb0ff5da199407016b80411e'],
        ['qlora-memory-breakdown', 'qlora-figure6-training-memory.svg', 'A7.F6', 'cc100be66d62507eb60507ee98600dc769aac28dd145b92b78462dd294fb7e21'],
      ]) {
        const section = html.match(new RegExp('id="' + id + '"[\\s\\S]*?(?=<h2\\b|<pre\\b)'))?.[0];
        expect(section).toBeDefined();
        const src = section?.match(new RegExp('<img[^>]+src="([^"]+/' + image.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(src).toBeDefined();
        expect(section).toContain('https://arxiv.org/html/2305.14314v1#' + url);
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex')).toBe(digest);
      }
    }
  });
  it('keeps the licensed LoRA originals on their reader routes and preserves the paper-note anchors', async () => {
    const route = 'textbook/post-training/fine-tuning-peft/';
    const license = 'licenses/microsoft-lora-c4593f06-MIT.txt';
    const hash = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
    const read = (path: string) => readFile(join(distDir, path, 'index.html'), 'utf8');
    expect(hash(await readFile(join(distDir, license)))).toBe('c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383');
    for (const locale of ['', 'en/']) {
      const chapter = await read(locale + route);
      const note = await read(locale + 'sources/papers/lora/');
      for (const [html, file, figure, digest] of [
        [chapter, 'lora-gpt3-figure2.png', 'S5.F2', '55d8c903fd6d94527701b4c187e9b89abb0f77c2a785fcef968bd492b117ea93'],
        [note, 'lora-gpt2-table3.png', 'S5.T3', '2f8ef35986bc3bfec8ee16e7cd3226454b9d261d83e25d0bed5ff593936c08e4'],
      ]) {
        const src = [...html.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/g)]
          .map(match => match[1]).find(url => url.endsWith('/' + file));
        expect(src).toBeDefined();
        expect(hash(await readFile(join(distDir, decodeURIComponent(src!))))).toBe(digest);
        expect(html).toContain('https://arxiv.org/html/2106.09685v2#' + figure);
        expect(html).toContain('href="/' + license + '"');
        expect(html).not.toContain('katex-error');
      }
      expect(note).toContain('href="/' + route + '#lora-gpt3-budget-plot"');
      for (const id of ['tldr', 'problem', 'method', 'ключевая-гипотеза',
        'lora-reparametrization-41', 'применение-в-transformer-42', 'key-results',
        'roberta--deberta-на-glue-table-2', 'gpt-2-на-e2e-nlg-table-3', 'gpt-3-175b-table-4',
        'inference-latency-table-1-lorafig5png', 'ablation-studies',
        'какие-матрицы-адаптировать-table-5-71', 'оптимальный-rank-r-table-6-72',
        'subspace-analysis-73-lorafig3png-fig4png', 'my-notes'])
        expect(note).toContain('id="' + id + '"');
      for (const link of ['S1.F1', 'S7.F3', 'S7.F4', 'A2.F5'])
        expect(note).toContain('https://arxiv.org/html/2106.09685v2#' + link);
      expect(note).toContain('354,92');
      expect(note).toContain('1482,0');
      expect(note).not.toContain('Оптимум ≈');
      expect(note).not.toContain('/curated/lora/fig');
    }
    const answer = await read('questions/100-nlp/source-answers/');
    expect(answer).toContain('/peft-2026-10-29/lora-animated.gif');
    expect(answer).toContain('href="/' + route + '#lora-update"');
    expect(answer).not.toContain('/curated/lora/fig1.png');
  });
  it('renders masked diffusion with four source figures, executable programs and module navigation', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/pretraining/masked-diffusion-language-models/index.html'), 'utf8');
      for (const anchor of ['masking-process', 'weighted-mask-loss', 'reverse-transition', 'complete-toy-sampler',
        'llada-training-and-selection', 'block-diffusion-and-kv', 'remasking-and-refinement', 'likelihood-and-serving'])
        expect(html).toContain(`id="${anchor}"`);
      for (const name of ['mdlm-overview.png', 'block-diffusion-overview.svg', 'remasking-overview.svg', 'remasking-loop.svg'])
        expect(html).toContain('masked-diffusion-2026-10-29/' + name);
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html.match(/class="katex-display"/g)).toHaveLength(8);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('CC BY 4.0');
      expect(html).toContain('textbook/pretraining/mae-masked-autoencoders/');
      if (locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders MAE source figures, complete programs and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/mae-masked-autoencoders/index.html'),'utf8');
      for(const anchor of ['asymmetric-mae','patch-layout','mask-and-restore','masked-pixel-loss',
        'normalized-targets','mae-training-step','reading-reconstructions','mae-cost-and-evaluation'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['architecture.png','pixel-reconstruction.png','gan-comparison.png'])
        expect(html).toContain('mae-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(2);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/mae-efb2a80-CC-BY-NC-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders MoCo queue, EMA, source diagrams and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/moco-queue-momentum/index.html'),'utf8');
      for(const anchor of ['dictionary','momentum-encoder','fifo-queue','moco-loss',
        'queue-snapshot','complete-step','distributed-shuffle','cost-and-versions'])
        expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('moco-2026-10-28/moco-dictionary.png');
      expect(html).toContain('simclr-2026-10-28/framework.png');
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(3);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/moco-132aea9-MIT.txt');
      expect(html).toContain('/licenses/moco-aea97d4-CC-BY-NC-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('connects all55 pretext subjects to unchanged complete chapters in both interfaces',async()=>{
    const topics=[{"route":"textbook/pretraining/semi-supervised-fixmatch","anchors":["pseudo-labels","noisy-student","noisy-targets","noisy-filtering","consistency","virtual-adversarial-training","vat-curvature","vat-step","uda","uda-augmentation","training-signal-annealing","mixtext","tmix-branches","mixtext-targets","mixtext-implementation","fixmatch","training-step","confirmation-bias","label-propagation","ssl-structural-assumptions","harmonic-label-propagation","published-results","cost-accounting","self-supervised-boundary","dai-le-sequence-pretraining"],"programs":14,"figures":8,"displays":22},{"route":"textbook/pretraining/simclr-representations","anchors":["two-view-encoder","augmentation-task","positive-index","nt-xent","shared-training-step","collapse-and-negatives","projection-versus-representation","linear-evaluation","simclr-probe-protocol"],"programs":5,"figures":4,"displays":2},{"route":"textbook/pretraining/moco-queue-momentum","anchors":["dictionary","moco-versus-simclr","momentum-encoder","fifo-queue","moco-loss","queue-snapshot","complete-step","moco-step-order","distributed-shuffle","cost-and-versions"],"programs":6,"figures":2,"displays":3},{"route":"textbook/pretraining/mae-masked-autoencoders","anchors":["asymmetric-mae","patch-layout","mask-and-restore","masked-pixel-loss","normalized-targets","mae-training-step","reading-reconstructions","mae-gan-demo","mae-cost-and-evaluation","mae-alternative-targets","mae-masking-recipe"],"programs":6,"figures":3,"displays":2}];
    for(const locale of ['', 'en/']){
      const index=await readFile(join(distDir,locale,'reference/index.html'),'utf8');
      for(const topic of topics){
        const html=await readFile(join(distDir,locale,topic.route,'index.html'),'utf8');
        for(const anchor of topic.anchors){
          expect(html).toContain('id="'+anchor+'"');
          expect(index).toContain('href="/'+locale+topic.route+'/#'+anchor+'"');
        }
        expect(html.match(/<pre\b/g)).toHaveLength(topic.programs);
        expect(html.match(/class="katex-display"/g)).toHaveLength(topic.displays);
        expect(html.match(/<img[^>]+src="[^"]*\/curated\//g)).toHaveLength(topic.figures);
        expect(html).not.toContain('katex-error');
        expect(html).not.toContain('This content is not available in your language yet.');
      }
    }
  });
  it('renders SimCLR source diagrams, programs, math and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/simclr-representations/index.html'),'utf8');
      for(const anchor of ['two-view-encoder','augmentation-task','positive-index','nt-xent',
        'shared-training-step','collapse-and-negatives','projection-versus-representation','linear-evaluation'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['framework.png','crop-views.png','augmentations.png','projection-head.png'])
        expect(html).toContain('simclr-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html.match(/class="katex-display"/g)).toHaveLength(2);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/uva-c7d9e18-MIT.txt');
      expect(html).toContain('/licenses/simclr-PMLR-CC-BY-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders semi-supervised mechanisms, unchanged original figures and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/semi-supervised-fixmatch/index.html'),'utf8');
      for(const anchor of ['pseudo-labels','consistency','fixmatch','training-step','confirmation-bias',
        'label-propagation','published-results','cost-accounting','self-supervised-boundary'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['fixmatch-diagram.png','circles-raw.png','circles-spread.png'])
        expect(html).toContain('semi-supervised-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(14);
      expect(html.match(/class="katex-display"/g)).toHaveLength(22);
      for(const anchor of ['noisy-student','mixtext','dai-le-sequence-pretraining','ssl-structural-assumptions'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['tmix-original.png','mixtext-original.png'])
        expect(html).toContain('semi-supervised-extensions-2026-10-29/'+name);
      for(const anchor of ['virtual-adversarial-training','uda','training-signal-annealing'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['vat.gif','uda-fixmatch.png'])
        expect(html).toContain('consistency-training-2026-10-29/'+name);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/fixmatch-d4985a1-Apache-2.0.txt');
      expect(html).toContain('/licenses/semi-supervised-sklearn-BSD-3-Clause.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders embedding adversarial training, FreeLB and SMART in both locales',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/robustness/index.html'),'utf8');
      for(const anchor of ['embedding-adversarial-training','freelb','smart'])
        expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('consistency-training-2026-10-29/smart-boundaries.png');
      expect(html).toContain('0.8036');
      expect(html).toContain('0.7892');
      expect(html.match(/<pre\b/g)).toHaveLength(3);
      expect(html.match(/class="katex-display"/g)).toHaveLength(16);
      expect(html).not.toContain('katex-error');
    }
  });
  it('publishes review capacity and selective-label auditing in both reader shells',async()=>{
    const route='textbook/evaluation/imbalanced-classification/';
    for(const prefix of ['', 'en/']){
      const chapter=await readFile(join(distDir,prefix,route,'index.html'),'utf8');
      const ref=await readFile(join(distDir,prefix,'reference/index.html'),'utf8');
      for(const anchor of ['imbalance-review-capacity','imbalance-selective-labels']){
        expect(chapter).toContain(`id="${anchor}"`);
        expect(ref).toContain(`href="/${prefix}${route}#${anchor}"`);
      }
      expect(chapter).toMatch(/href="(?:[^"]*\/)?#imbalance-review-capacity"/);
      expect(chapter).toContain('3332.5');
      expect(chapter).toContain('threshold-tradeoff.png');
      expect(chapter).toContain('kdd17-selective.pdf');
      expect(chapter).not.toContain('katex-error');
    }
  });
  it('renders geometric selection with original figures, math, code and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/geometric-coresets/index.html'),'utf8');
      for(const anchor of ['coverage-objective','farthest-first','approximation-guarantee',
        'implementation','representation-and-metric','outliers-and-labels',
        'coreset-versus-centres','herding']) expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('geometric-coresets-2026-10-28/erickson-farthest-first.png');
      expect(html).toContain('selection-metric-2026-11-04/mooc-kmeans-not-scaled.svg');
      expect(html).toContain('selection-metric-2026-11-04/mooc-kmeans-scaled.svg');
      expect(html).toContain('id="coverage-trace"');
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(10);
      expect(html).not.toContain('katex-error');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders active-learning mechanisms, original images, complete programs and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/active-learning/index.html'),'utf8');
      for(const anchor of ['annotation-loop','uncertainty','committee','expected-gradient',
        'executable-loop','batch-and-diversity','evaluation-and-cost']) expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['active-learning.png','motivating-example.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*active-learning-2026-10-28/'+figure.replace('.','\\.')));
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html).toContain('/licenses/modal-bba6f6fd-MIT.txt');
      expect(html.match(/class="katex-display"/g)).toHaveLength(8);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('oracle_calls');
      expect(html).toContain('consensus entropy');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders labelled selection and Confident Learning with four source figures and eight complete programs',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/data-selection-scores/index.html'),'utf8');
      for(const anchor of ['selection-problem','el2n','grand','forgetting','stable-identifiers',
        'reproduction','selection-controls','confident-learning','selection-cost']) expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['joost_train_input_norm_vs_grad_norm_sorted.png','figure11_repro_cifar10_hlb.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*data-selection-2026-10-28/'+figure.replace('.','\\.')));
      for(const figure of ['figure1-process.png','figure2-joint-estimation.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*confident-learning-2026-10-29/'+figure.replace('.','\\.')));
      expect(html.match(/<pre\b/g)).toHaveLength(8);
      expect(html.match(/class="katex-display"/g)).toHaveLength(17);
      expect(html).not.toContain('katex-error');
      for(const fact of ['never_correct','score_by_id','shaded bands','Apache-2.0','CC BY-NC-SA 4.0'])
        expect(html).toContain(fact);
      for(const license of ['data-diet-aee462ea-Apache-2.0.txt','data-diet-reproduction-fda604dc-Apache-2.0.txt'])
        await access(join(distDir,'licenses',license));
      expect(html).toContain('/licenses/confident-learning-JAIR-v1.html');
      await access(join(distDir,'licenses/confident-learning-JAIR-v1.html'));
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders original GaLoRE figures, the three memory examples and explicit EN fallback',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/optimizers-schedules/index.html'),'utf8');
      for(const anchor of ['memory-efficient-optimizers','adafactor-factor-the-second-moment-not-the-weight-matrix',
        '8-bit-optimizer-states-fewer-bytes-for-the-same-coordinates',
        'galore-run-adam-in-a-smaller-gradient-space','account-for-the-remaining-tensors'])
        expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['galore_code_box.png','subspace_learning.png','memory_breakdown.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*galore-2026-10-28/'+figure.replace('.','\\.')));
      expect(html).toContain('/licenses/galore-2cc66f88-Apache-2.0.txt');
      expect(html).toContain('Первый шаг Adam на весах MLP');
      for (const value of ['−0,02384','−0,05960','1,00238','2,00596'])
        expect(html).toContain(value);
      expect(html).toMatch(/<details>\s*<summary>Открыть AGC,[^<]+<\/summary>[\s\S]*?id="agc-unitwise"[\s\S]*?id="matrix-optimizers"[\s\S]*?id="optimizer-memory-accounting"[\s\S]*?<\/details>/);
      expect(html).toContain('id="galore-memory-breakdown"');
      expect(html).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(html.match(/<pre\b/g)).toHaveLength(10);
      for(const anchor of ['minibatch-sgd','learning-rate-curvature','momentum',
        'nesterov','adagrad-rmsprop','adam-moments','adamw','optimizer-step-comparison',
        'schedule-boundary','critical-batch-size','optimizer-loss-scaling',
        'optimizer-checkpoint','matrix-optimizers','adafactor','optimizer-8bit-states',
        'galore-projection','optimizer-memory-accounting']) expect(html).toContain('id="'+anchor+'"');
      const core=html.indexOf('id="optimizer-checkpoint"');
      expect(core).toBeLessThan(html.indexOf('id="agc-unitwise"'));
      expect(core).toBeLessThan(html.indexOf('id="matrix-optimizers"'));
      for (const anchor of ['agc-unitwise','autoclip'])
        expect(html).toContain('id="'+anchor+'"');
      for (const name of ['nfnet-agc-ablation.png','autoclip-mi.gif'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*gradient-clipping-2026-10-29/'+name.replace('.','\\.')));
      expect(html).toContain('/licenses/autoclip-dfb622c3-MIT.txt');
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('LoRA, QLoRA и PEFT');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders rare-gradient arithmetic, the original lookup figure and the PyTorch example', async () => {
    const anchor = 'optimizer-sparse-gradients';
    const route = 'textbook/neural-networks/optimizers-schedules/';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = html.match(/id="optimizer-sparse-gradients"[\s\S]*?(?=<a id="adamw")/)?.[0];
      expect(section).toBeDefined();
      const src = section?.match(/<img[^>]+src="([^"]*\/alammar-word2vec-lookup-embeddings\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex'))
        .toBe('cef0f9dd8bc6bcc5a2bab1301e2337202ef88539c233111b7fd021d34c5d9759');
      for (const value of ['0.00995', '0.31623', '0.09804', '0.06701', 'SparseAdam',
        '#emb-sgns-step', 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'https://docs.pytorch.org/docs/2.15/generated/torch.optim.SparseAdam.html'])
        expect(section).toContain(value);
      expect(section).toContain('<pre');
      expect(html).toMatch(/<details>\s*<summary>Редкие градиенты и SparseAdam[^<]+<\/summary>[\s\S]*?id="optimizer-sparse-gradients"[\s\S]*?<\/details>/);
      expect(section).not.toContain('katex-error');
      const reference = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
      expect(reference).toContain('href="/'+locale+route+'#'+anchor+'"');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the source optimizer trajectories and links out to the retired animation', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale,
        'textbook/neural-networks/optimizers-schedules/index.html'), 'utf8');
      const adaptive = html.match(/id="optimizer-adaptive-trajectories"[\s\S]*?(?=<a id="adam-moments")/)?.[0];
      const saddle = html.match(/id="optimizer-saddle-noise"[\s\S]*?(?=<a id="schedule-boundary")/)?.[0];
      expect(adaptive).toBeDefined();
      expect(saddle).toBeDefined();
      for (const [name, upstream] of [
        ['d2l-adagrad-trajectory.svg', 'output_adagrad_2fb0ed_3_1.svg'],
        ['d2l-rmsprop-trajectory.svg', 'output_rmsprop_251805_15_1.svg'],
      ]) {
        expect(adaptive).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replace('.', '\\.')+'"'));
        expect(adaptive).toContain('https://d2l.ai/_images/'+upstream);
      }
      expect(adaptive).toContain('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(saddle).toMatch(/<img[^>]+src="[^"]*\/staib19a-figure1\.png"/);
      expect(saddle).toContain('https://proceedings.mlr.press/v97/staib19a/staib19a.pdf#page=8');
      expect(saddle).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(saddle).toContain('href="https://cs231n.github.io/assets/nn3/opt1.gif"');
      expect(html).not.toContain('cs231n-optimizer-trajectories.gif');
      expect(html).not.toContain('katex-error');
      expect(html.match(/id="schedule-boundary"/g)).toHaveLength(1);
      if (locale) expectCanonicalEditionNotice(html);
    }
    for (const base of ['assets', 'assets/00 Учебник/Assets']) {
      await expect(readFile(join(distDir, base,
        'Figures/curated/source-first-7-14/cs231n-optimizer-trajectories.gif')))
        .rejects.toMatchObject({code: 'ENOENT'});
    }
  });
  it('renders initialization derivations, original normalization figures and source BatchNorm code',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/initialization-normalization-residuals/index.html'),'utf8');
      for(const anchor of ['initialization-symmetry','initialization-second-moment',
        'xavier-initialization','he-initialization','normalization-axes','batchnorm-train-eval',
        'normalization-causal-leakage','token-layernorm','rmsnorm-forward-precision',
        'normalization-backward','residual-addition','residual-gradients',
        'residual-zero-projection','pre-post-normalization','residual-variance',
        'normalization-diagnostics','batchnorm-source-implementation']) expect(html).toContain('id="'+anchor+'"');
      expect(html.match(/<pre\b/g)).toHaveLength(4);
      for(const figure of ['d2l-residual-block.svg','d2l-resnet-block.svg','cs336-prenorm-block.png',
        'xiong-pre-post-norm.png','karpathy-activation-distribution.png','karpathy-gradient-distribution.png'])
        expect(html).toContain(figure);
      expect(html).toContain('/licenses/karpathy-nn-zero-to-hero-73c3fcc7-MIT.txt');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders the original activation comparison and the normalization-axis figure where they are explained', async () => {
    const route = 'textbook/neural-networks/initialization-normalization-residuals/index.html';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, route), 'utf8');
      const distributions = html.match(/<a id="activation-init-distributions"><\/a>[\s\S]*?(?=<a id="xavier-initialization")/)?.[0];
      const axes = html.match(/<a id="normalization-axes"><\/a>[\s\S]*?(?=<a id="batchnorm-train-eval")/)?.[0];
      expect(distributions).toBeDefined();
      expect(axes).toBeDefined();
      for (const file of ['uva-activation-std-001.svg', 'uva-activation-std-01.svg', 'uva-activation-fan-in.svg'])
        expect(distributions).toMatch(new RegExp('<img[^>]+src="[^"]*/'+file.replace('.', '\\.')+'"[^>]+alt="UvA:'));
      expect(distributions).toContain('/licenses/uvadlc-c7d9e18f-MIT.txt');
      for (const value of ['0{,}077930', '0{,}000083', '262{,}5', '0{,}87'])
        expect(distributions).toContain(value);
      expect(axes).toMatch(/<img[^>]+src="[^"]*\/fleuret-normalization-axes\.png"[^>]+alt="Флёре:/);
      expect(axes).toContain('https://fleuret.org/public/lbdl.pdf#page=83');
      expect(axes).toContain('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(axes).not.toContain('katex-error');
    }
  });
  it('renders the original TensorFlow source and the actually checked SavedModel-to-Lite boundary',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/ml-workflow/index.html'),'utf8');
      for(const anchor of ['tensorflow-saved-graphs-variables-and-callable-signatures',
        'what-survives-saving-and-loading','named-endpoints-and-the-exported-state',
        'conversion-to-a-smaller-runtime-is-another-checked-boundary','keep-a-tested-migration-path'])
        expect(html).toContain(`id="${anchor}"`);
      for(const fact of ['two-layer-network.png','CustomModule','checked_scale','4.98186',
        'BUILTIN_WITHOUT_DEFAULT_DELEGATES','flattened signature callable']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt');
      expect(html).not.toContain('katex-error');
    }
    expect(await readFile(join(distDir,'licenses/tensorflow-docs-9beed83c-Apache-2.0.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt'),'utf8'));
  });
  it('publishes the worked model-export boundary and full original provider figure in both routes',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/ml-workflow/index.html'),'utf8');
      expect(html).toContain('id="exporting-a-model-does-not-export-the-application"');
      expect(html).toContain('id="which-provider-executes-each-part"');
      expect(html).toContain('onnx-runtime-providers.png');
      for(const fact of ['MissingKernel','CPUExecutionProvider','Batch 9','1454×910']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/onnxruntime-e3d000e3-MIT.txt');
      expect(html).not.toContain('katex-error');
    }
    expect(await readFile(join(distDir,'licenses/onnxruntime-e3d000e3-MIT.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/onnxruntime-e3d000e3-MIT.txt'),'utf8'));
  });
  it('renders the camera-trap memory, deadline and disjoint-state energy calculation', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'practice/18a-architecture-selection/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="a-camera-trap-memory-event-latency-and-battery-energy"');
      for(const value of ['65.045056','75 ms','865.425','43.27125','50.325','8.64 kWh',
        'same MAC/FLOP convention','class counts alone do not prove']) expect(text).toContain(value);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
      expect(html).toContain('harvard-architecture-selection.svg');
      expect(html).toContain('harvard-architecture-budgets.svg');
      expect(html).toContain('/examples/architecture-lab-source-checks-2026-10-26.json');
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders complete Winograd examples, native figure and original downloads in both routes', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,
        'textbook/neural-networks/cnn-lenet-resnet/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toMatch(/<img\b[^>]*src="[^"]*\/winograd-2026-10-27\/dolz-workflow\.png"/);
      for(const fact of ['Four products recover two outputs', 'Channels become sixteen',
        '2049', '0.5 instead of 1', 'not a measured speedup', 'torch.bmm', 'CC BY 4.0'])
        expect(text).toContain(fact);
      expect(html).toContain('href="/examples/wincnn-c2bf6305.py"');
      expect(html).toContain('href="/licenses/wincnn-c2bf6305-Apache-2.0.txt"');
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
    expect(await readFile(join(distDir,'examples/wincnn-c2bf6305.py')))
      .toEqual(await readFile(join(rootDir,'site/public/examples/wincnn-c2bf6305.py')));
  });

  it('renders the original Harvard window diagram with the corrected orientation in both routes', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,
        'textbook/neural-networks/cnn-lenet-resnet/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="two-channels-window-rows-and-filter-vector"');
      expect(html).toMatch(/<img\b[^>]*src="[^"]*\/nn-native-2026-10-27\/harvard-window-rows\.svg"/);
      for(const fact of ['356', '392', '464', '500', 'each channel',
        'This explanatory caption corrects', 'No timing measurement', 'CC BY-NC-SA 4.0'])
        expect(text.toLowerCase()).toContain(fact.toLowerCase());
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders rooted/all-rank destinations and complete variable-split routing', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale,
        'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
      const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
      expect(html).toContain('id="collective-destinations-gather-scatter-all-to-all"');
      for (const name of ['scatter.png', 'gather.png', 'all_gather.png'])
        expect(html).toMatch(new RegExp('<img\\b[^>]*src="[^"]*/collective-destinations-2026-10-27/'
          + name.replaceAll('.', '\\.') + '"'));
      for (const fact of ['input_split_sizes', 'output_split_sizes', 'sends five',
        'receives six', 'CPU/Gloo', 'experimental']) expect(text).toContain(fact);
      expect(html).toContain('/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt');
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders indexed gather/scatter, original SVG, gradients and both API contracts', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="indexed-gather-scatter-and-gradients"');
    expect(html).toMatch(/<img\b[^>]*\bsrc="[^"]*\/scatter-add\.svg"/);
    for (const phrase of ['GatherElements-13', 'Scatter-add combines values',
      'initial.scatter_add', 'fp32_access_units', 'full one-to-one permutation',
      'not measured HBM traffic']) expect(text).toContain(phrase);
    expect(html).toContain('/licenses/pytorch-scatter-f514c10-MIT.txt');
    expect(html).not.toContain('katex-error');
  });

  it('renders the original reduction animation, worked tree and copied CUDA helper', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="local-reduction-tree-work-depth-synchronization"');
    expect(html).toMatch(/<img\b[^>]*\bsrc="[^"]*\/binomial-tree\.gif"/);
    for (const phrase of ['reduction_trace', 'Independent additions', 'cg::sync(cta)',
      'warpReduceSum', '__shfl_down_sync', 'all 32 lanes to participate',
      'not a complete kernel or an executed CUDA test', 'Flogr']) expect(text).toContain(phrase);
    expect(html).toContain('/licenses/nvidia-cuda-samples-5443602-BSD-3-Clause.txt');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(10);
    expect(html).not.toContain('katex-error');
  });

  it('renders the Harvard Transformer figure joined to a bare DOI and verifies the archived revision', async () => {
    const path='05 Источники/Courses/Harvard ML Systems/vol1/nn_architectures.md';
    const source=await readFile(join(rootDir,path),'utf8');
    expect(createHash('sha256').update(beforeTopicConsolidation(source,path,rootDir)).digest('hex'))
      .toBe('52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac');
    const route = 'sources/courses/harvard-ml-systems/vol1/network-architectures';
    const html=await readFile(join(distDir,route,'index.html'),'utf8');
    const main=html.slice(html.indexOf('<main'),html.indexOf('</main>'));
    expect(main.match(/<img\b[^>]*\bsrc="/g)).toHaveLength(24);
    expect(main).toMatch(/<img\b[^>]*\bsrc="[^"]*8f27999af14450c9623159ca4a3f23e5463a4c45\.svg"/);
    expect(main).toContain('href="https://doi.org/10.18653/v1/w19-4828"');
    expect(main).not.toContain('w19-4828.%5B!%5B');
    expectCourseOriginalAlias(await readFile(join(distDir,'en',route,'index.html'),'utf8'), route);
  });

  it('renders full LSTM/GRU direction, bias and projection explanations in both locales',async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-03-lstm-gru-contract-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^    anchor: ([a-z-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(21);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/recurrent-networks/lstm-gru/index.html'),'utf8');
      for(const id of anchors)expect(html.split('id="'+id+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
      expect(html.match(/<img\b[^>]+\bsrc="\/assets\//g)).toHaveLength(9);
      for(const phrase of ['biased_gru','bidirectional_lstm','projected_lstm',
        'manual_gradients','library_gradients','246 rather than 240',
        'original English excerpt','/d2l-birnn.svg','CC BY-SA 4.0',
        '/lstm-hidden-state.svg','/lstm-gates.svg','/d2l-lstm-1.svg','/d2l-lstm-2.svg',
        '/d2l-gru-2.svg','/cho-gru-figure2.png','CC BY-NC-SA 3.0',
        '0.653275','D14-1179.pdf#page=3','#step-by-step-lstm-walk-through',
        '#variants-on-long-short-term-memory','/img/LSTM3-chain.png'])
        expect(html).toContain(phrase);
      expect(html).not.toMatch(/<img\b[^>]*\bsrc="[^"]*LSTM3-/);
      expect(html).not.toContain('class="katex-error');
      expect(html).toContain('/textbook/recurrent-networks/seq2seq-bottleneck/');
      if(locale)expect(html).toContain('Canonical-language chapter.');
    }
  });

  it('renders RNN shape/bias/packing examples and subject anchors in both locales',async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-03-rnn-library-contract-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^    anchor: (rnn-[a-z-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(22);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/recurrent-networks/rnn-bptt/index.html'),'utf8');
      for(const id of anchors)expect(html.split('id="'+id+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
      expect(html.match(/<img\b[^>]+\bsrc="\/assets\//g)).toHaveLength(6);
      for(const phrase of ['effective_after_two_biases','manual_gradients','packed_output',
        'packed_final','restored_lengths','dense_final','100 parameters instead of 96',
        'Рекуррентная связь и развёртка RNN с общими весами по времени'])
        expect(html).toContain(phrase);
      expect(html).not.toContain('class="katex-error');
      expect(html).toContain('https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html');
      expect(html).toContain('/textbook/recurrent-networks/lstm-gru/');
      if(locale)expect(html).toContain('Canonical-language chapter.');
    }
  });

  it('renders recurrent input projections, layer dependencies and the complete original D2L figure', async () => {
    const html=await readFile(join(distDir,'textbook/recurrent-networks/rnn-bptt/index.html'),'utf8');
    for(const id of ['parallel-work-inside-a-recurrent-network','full-sequence-chunks-and-a-live-stream',
      'several-layers-form-a-dependency-grid','weight-gradients-can-be-reduced-across-time'])
      expect(html).toContain('id="'+id+'"');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/d2l-deep-rnn\.svg"/);
    expect(html.match(/<img\b[^>]+src="\/assets\//g)).toHaveLength(6);
    expect(html.match(/<pre\b/g)).toHaveLength(9);
    for(const fact of ['62.5 MiB','1.25 MiB','Original English excerpt','projection chunk:',
      'states and all gradients match','not a measured twofold GPU speedup','CC BY-SA 4.0'])
      expect(html).toContain(fact);
    expect(html).not.toContain('katex-error');
  });
  it('renders the DDP overlap explanation, original bucketing figure and stream examples', async () => {
    const html = await readFile(join(distDir,
      'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="ddp-overlap-readiness-streams-compilation"');
    expect(html).toContain('gradient-buckets.svg');
    for (const phrase of ['ordered_bucket_schedule', 'DDPOptimizer', 'record_stream',
      'whole_backward', 'split_backward', 'collective order',
      'the exposed tail is 5 ms', 'not an NCCL simulator', 'not an executed test'])
      expect(text).toContain(phrase);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(15);
    expect(html).toContain('/textbook/ml-systems/graph-compilation-pytorch-jax/#torch-compile-graphs-guards-recompilation');
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('<p>![[');
  });
  it('renders the Dynamo source explanation, original diagram and three frontend experiments', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/graph-compilation-pytorch-jax/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="torch-compile-graphs-guards-recompilation"');
    expect(html).toContain('torch-dynamo-pipeline.png');
    for (const value of ['compile_records', 'piece_records', 'dynamic_records',
      'strict_boundary', 'fullgraph=True', 'AOTDispatcher',
      'not kernel launches', 'not AOTAutograd or Inductor code generation']) expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(35);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('dynamo-summary.png');
  });
  it('renders source tracing failures and explicit export control flow', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/graph-compilation-pytorch-jax/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="tracing-a-data-dependent-branch"');
    for (const value of ['Tracing Silent Failure',
      'branch_cases', 'trace_nodes', 'check_inputs', 'compiled_branch',
      'fresh compiler cache', 'torch.compiler.reset', 'Exporting an explicit conditional',
      'M_old', 'M_new', 'exported_branch', 'higher_order.cond', '0.841471',
      '0.540302', 'prototype features', 'not deployment'])
      expect(text).toContain(value);
    expect(html).toContain('/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(35);
    expect(html).not.toContain('katex-error');
  });
  it('renders the integrated CUDA Graphs explanation, original figures and source code', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/graph-compilation-pytorch-jax/index.html'), 'utf8');
    for (const name of ['launch-vs-replay.png', 'mask-rcnn-before.png', 'mask-rcnn-after.png'])
      expect(html).toContain(name);
    for (const phrase of ['cuda-graphs-capture-replay', 'Why CUDA Graphs?',
      'replay_fixed_buffer', 'launch_bound_finish', 'static_input.copy_',
      'Memory pools are part of the cost', '31.662 ms', '6.078 ms',
      'not yet supported by the Inductor backend', 'not measurements of CUDA'])
      expect(html).toContain(phrase);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(35);
    expect(html).toContain('/licenses/pytorch-v2.14.1-LICENSE.txt');
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('<p>![[');
  });
  it('renders Harvard slide math, native SVGs and separate original speaker notes', async () => {
    const records = JSON.parse(await readFile(join(rootDir,
      '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'), 'utf8'));
    const slides = records.filter((row: {kind: string}) => row.kind === 'slides');
    expect(slides).toHaveLength(35);
    for (const row of slides) {
      const html = await readFile(join(distDir, row.route, 'index.html'), 'utf8');
      expect(html, row.source).not.toContain('katex-error');
      expect(html, row.source).not.toContain('<p>![[');
      expect(html, row.source).toContain('Original speaker notes (TeX)');
      expect(html, row.source).toContain('Complete original Beamer source');
      expect(html, row.source).toContain('Original slide 1');
    }
    const edge = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/slides/vol2/11-edge-intelligence/index.html'), 'utf8');
    for (const figure of ['cloud-vs-edge-learning.svg', 'training-memory-amplifier.svg',
      'fedavg-protocol.svg', 'edge-device-spectrum.svg', 'adaptation-strategies.svg',
      'federated-systems-challenges.svg']) expect(edge).toContain(figure);
    expect(edge).toContain('Original SVG, copied without redrawing or cropping.');
    expect(edge).toContain('Source figure limitation');
    expect(edge).toContain('cannot be read as additive memory amounts');
    expect(edge).toContain('θ');
    expect(edge).toContain('<table>');
    // Lecture instructions belong to collapsed code, not narrative paragraphs.
    expect(edge).not.toMatch(/<p>[^<]*%[^<]*NARRATE:/);
    const architectures = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/slides/vol1/06-nn-architectures/index.html'), 'utf8');
    expect(architectures).toContain('Source arithmetic mismatch');
    expect(architectures).toContain('86,973,087,744');
    expect(architectures).toContain('356,241,767,399,424');
  });
  it('keeps the reading sequence separate from the common course library', async () => {
    const html=await readFile(join(distDir,'textbook/index.html'),'utf8');
    const curriculum = html.match(/<h2[^>]*id="основной-курс"[\s\S]*?(?=<a id="gpt-decoder-route")/)?.[0];
    expect(curriculum?.match(/<h3\b/g)).toHaveLength(23);
    for(const part of ['Основной курс','Углублённые модули','II. Нейронные сети','IV. Transformer, BERT и T5','Рекомендательные системы'])
      expect(html).toContain(part);
    for(const alias of ['iv-нейронные-сети','iii-рекомендательные-системы','другие-режимы'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/sources/kursy/');
    expect(html).not.toContain('/sources/courses/stanford-cs336-spring-2026/');
    expect(html).not.toContain('/sources/courses/berkeley-advanced-llm-agents-spring-2025/');
    const catalog=await readFile(join(distDir,'sources/kursy/index.html'),'utf8');
    expect(catalog).toContain('/sources/courses/stanford-cs336-spring-2026/');
    expect(catalog).toContain('/sources/courses/berkeley-advanced-llm-agents-spring-2025/');
    const svm=await readFile(join(distDir,'textbook/classical-ml/kernels-svm-rff/index.html'),'utf8');
    expect(svm).not.toContain('>08 Gaussian mixture и EM</a>');
    expect(svm).toMatch(/<a\b(?=[^>]*\brel="next")(?=[^>]*\bhref="\/textbook\/#)[^>]*>/);
  });
  it('opens the course in the textbook sidebar without expanding unrelated sections', async () => {
    const open = (html: string, label: string) => {
      const group = [...html.matchAll(/<details([^>]*)><summary[^>]*><span[^>]*><span[^>]*>([^<]+)<\/span>/g)]
        .find(match => match[2] === label);
      expect(group, label).toBeDefined();
      return /\bopen\b/.test(group![1]);
    };
    for (const [lang, core, reference, modules] of [
      ['', 'Основной курс', 'Справочник', 'Углублённые модули'],
      ['en', 'Core course', 'Reference', 'Specialist modules'],
    ]) {
      const html = await readFile(join(distDir, lang, 'textbook/index.html'), 'utf8');
      expect(open(html, core)).toBe(true);
      expect(open(html, reference)).toBe(false);
      expect(open(html, modules)).toBe(false);
      const index = await readFile(join(distDir, lang, 'reference/index.html'), 'utf8');
      expect(open(index, reference)).toBe(true);
    }
  });
  it('publishes previous and next buttons for all chapters in both languages', async () => {
    const book = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections.find(s => s.id === 'textbook')!;
    const publishedRoute = (route: string) => route.replace(/\/index$/, '');
    let checked = 0;
    for (const lang of ['', 'en/']) for (const path of readingPaths(book)) for (const [i, route] of path.routes.entries()) {
      const html = await readFile(join(distDir, lang, publishedRoute(route), 'index.html'), 'utf8');
      const anchors = html.match(/<a\b[^>]*>/g) ?? [];
      const modules = 'textbook/#' + (lang ? 'specialist-modules' : 'углублённые-модули');
      const contents = 'textbook/#' + (lang ? 'curriculum' : 'полный-маршрут');
      const prev = path.routes[i - 1] ? publishedRoute(path.routes[i - 1]) + '/'
        : path.id === 'main' ? 'textbook/how-to-use/' : modules;
      const next = path.routes[i + 1] ? publishedRoute(path.routes[i + 1]) + '/'
        : path.id === 'main' ? contents : modules;
      for (const [rel, expected] of [['prev', prev], ['next', next]]) {
        const button = anchors.find(a => a.includes(`rel="${rel}"`));
        expect(button, `${route}: ${rel} button`).toBeTruthy();
        expect(decodeURI(button!.match(/href="([^"]+)"/)![1]), `${lang}${route}: ${rel}`).toBe('/' + lang + expected);
      }
      checked++;
    }
    expect(checked).toBe(306);
  });
  it('renders the complete original GAN/DCGAN reader and four-route gradient explanation in both locales',async()=>{
    const source=await readFile(join(rootDir,'00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор.md'),'utf8');
    const anchors=[...source.matchAll(/^    anchor: ((?:gan|dcgan)-[a-z-]+)$/gm)].map(x=>x[1]);
    expect(anchors).toHaveLength(18);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/gan-dcgan/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(15);
      expect(html.match(/<img\b[^>]+\bsrc="\/assets\//g)).toHaveLength(5);
      for(const phrase of ['frozen_routes','ordinary','frozen_D','detached_fake','no_grad_D',
        'restore D','no optimizer step','three BatchNorm visits','CC BY-SA 4.0',
        'not a plot produced by the local loop'])
        expect(text).toContain(phrase);
      for(const name of ['gan.svg','gaussian-data.svg','gaussian-training.svg',
        'leaky-relu.svg','transpose-convolution.gif'])expect(html).toContain(name);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/textbook/neural-networks/autoencoder-vae/');
      expect(html).toContain('/textbook/multimodal/diffusion-flow-images/');
    }
  });
  it('renders the UvA comparisons, retrieval example and24 subject anchors in both AE/VAE routes',async()=>{
    const source=await readFile(join(rootDir,'00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md'),'utf8');
    const anchors=[...source.matchAll(/^    anchor: ((?:ae|vae)-[a-z-]+)$/gm)].map(x=>x[1]);
    expect(anchors).toHaveLength(24);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/autoencoder-vae/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(15);
      expect(html.match(/<img\b[^>]+\bsrc="\/assets\//g)).toHaveLength(10);
      for(const asset of ['uva-latent-width.svg','uva-noise-reconstruction.svg',
        'uva-pattern-reconstruction.svg','uva-random-codes.svg'])
        expect(html).toContain(asset);
      for(const phrase of ['coordinate_map','torch.cdist','recovered_codes',
        'not a measured 48-fold file compression ratio','not samples from a VAE'])
        expect(text).toContain(phrase);
      expect(html).not.toContain('katex-error');
      if(locale)expect(text).toContain('Canonical-language chapter.');
    }
  });
  it('renders AE/VAE derivations, original course code, fifteen examples and ten complete figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/autoencoder-vae/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(15);
    for (const name of ['chervinskii-autoencoder.png', 'uva-autoencoder.svg', 'transpose-convolution.gif',
      'prince-vae-posterior-figure17-8.png', 'vae-blocks.png', 'prince-vae-reparameterization-figure17-11.png'])
      expect(html).toContain(name);
    for (const phrase of ['class Encoder', 'class Decoder', 'class VAE', 'def loss_function',
      'posterior_gap', 'fixed_epsilon', 'output_padding=1', 'Normal.rsample()', 'interpolation_codes',
      'два KL с разными аргументами', 'стандартного отклонения', '0,596574',
      'У истинного posterior получаются два пика', 'Шум', 'мягкие цели',
      'среднему восстановлению']) expect(text).toContain(phrase);
    for (const id of ['почему-обычный-latent-space-неудобен-для-генерации',
      'elbo-reconstruction-и-regularization-latent-space', 'краткие-итоги']) expect(html).toContain(`id="${id}"`);
    for (const name of ['uva-c7d9e18f-MIT.txt', 'pytorch-examples-acc295dc-BSD-3-Clause.txt',
      'conv-arithmetic-af6f818b-MIT.txt']) {
      expect(html).toContain(name);
      await access(join(distDir, 'licenses', name));
    }
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('Это исходный пример прежней редакции');
    expect(html).toContain('f_\\theta(g_\\phi(x_i))');
    expect(html).toContain('/textbook/neural-networks/cnn-lenet-resnet/');
    expect(html).toContain('/textbook/text-representation/numerical-representation/');
  });
  it('renders the complete neural-LM path and original Bengio architecture in both locales',async()=>{
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix,'textbook/language-modeling/next-token-probability/index.html'),'utf8');
      for(const anchor of ['nlm-sequence-nll','nlm-head','nlm-window','nlm-target-shift',
        'nlm-causality','nlm-teacher-forcing','nlm-cross-entropy','nlm-vocabulary-gradient',
        'nlm-stability','nlm-perplexity','nlm-loss-reduction','nlm-entropy-floor',
        'nlm-objective-limits','nlm-base-assistant','nlm-eos-pad-mask','nlm-window-training',
        'nlm-bengio-architecture','nlm-context-permutation','autoregressive-objective',
        'stable-cross-entropy','objective-evidence','от-биграммы-к-длинному-контексту',
        'минимальная-реализация-функции-потерь']) expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
      for(const name of ['d2l-softmaxreg.svg','d2l-language-model-data.png','d2l-gpt-decoder.png','bengio-original-figure1.png'])
        expect(html).toContain(name);
      for(const alt of ['Полносвязная выходная проекция из четырёх признаков в три оценки',
        'Посимвольные входы и цели со сдвигом на одну позицию',
        'Сдвинутые цели GPT и доступные причинные зависимости'])expect(html).toContain('alt="'+alt+'"');
      for(const phrase of ['source_vectors','lookup_gradient','WindowLM','batch_loss','128'])
        expect(html).toContain(phrase);
      expect(html).toContain('/licenses/bengio03a-figure1-jmlr-policy.txt');
      expect(html).not.toContain('katex-error');
      if(prefix)expect(html).toContain('Canonical-language chapter.');
    }
  });
  it('renders the N-gram derivations, original count matrix and source code in both locales',async()=>{
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix,'textbook/language-modeling/ngram/index.html'),'utf8');
      for(const anchor of ['ngram-chain-rule','ngram-boundaries','ngram-markov','ngram-counts',
        'ngram-mle-derivation','ngram-sequence-probability','ngram-zero-events','ngram-add-k',
        'ngram-interpolation','ngram-backoff','ngram-stupid-backoff','ngram-kn-continuation',
        'ngram-kn-discount','ngram-kn-recursion','ngram-perplexity','ngram-corpus-metrics',
        'ngram-generation','ngram-sparsity','ngram-neural-bridge','ngram-makemore-matrix',
        'ngram-makemore-normalization']) expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
      for(const name of ['bjc-this-is-a-sentence.png','heafield-backoff-figure1.png','moore-smoothing-figure1.png','karpathy-bigram-counts.png'])
        expect(html).toContain(name);
      expect(html).not.toContain('/voita-language-modeling/');
      for(const alt of [
        'Фраза This is a sentence даёт четыре униграммы, три биграммы и две перекрывающиеся триграммы.',
        'Поиск самого длинного сохранённого суффикса с накоплением коэффициентов backoff.',
        'Верхняя строка для наблюдавшегося ответа имеет прямой и младший вклады; нижняя строка для невиданного ответа имеет только младший.',
      ]) expect(html).toContain('alt="'+alt+'"');
      for(const url of [
        'https://aclanthology.org/D12-1107.pdf#page=4',
        'https://aclanthology.org/P09-2088.pdf#page=2',
        'https://lena-voita.github.io/nlp_course/language_modeling.html#smoothing_interpolation',
      ]) expect(html).toContain(url);
      for(const phrase of ['Reading makemore','Row normalization','row_denominators','33,885','32,033'])
        expect(html).toContain(phrase);
      expect(html).toContain('/licenses/karpathy-nn-zero-to-hero-73c3fcc7-MIT.txt');
      expect(html).not.toContain('katex-error');
      if(prefix)expect(html).toContain('Canonical-language chapter.');
    }
  });
  it('renders CNN mechanisms, twenty-two executable examples and fifteen original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/cnn-lenet-resnet/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(22);
    for (const name of ['d2l-cross-correlation.svg', 'cs231n-local-connectivity.jpeg',
      'conv-multi-in.svg', 'conv-1x1.svg', 'cs231n-stride-padding.jpeg', 'pooling.svg',
      'lenet.svg', 'vgg.svg', 'd2l-resnet-block.svg', 'resnet18.svg',
      'escnn-feature-fields.png', 'escnn-vectorfield.gif', 'escnn-conventional-cnn.gif',
      'harvard-window-rows.svg', 'dolz-workflow.png']) expect(html).toContain(name);
    for (const phrase of ['4,9152', '61706', 'F.unfold', 'make_resnet18',
      'zero_gradient', 'не означает ошибку', 'не гарантирует', 'Height и width',
      'код AlexNet в D2L задаёт', 'реальных пикселей в ней меньше']) expect(text).toContain(phrase);
    for (const id of ['двумерная-cross-correlation', 'padding-stride-и-размер-выхода',
      'receptive-field', 'lenet-5', 'alexnet-vgg-и-переход-к-глубине', 'resnet',
      'связь-cnn-с-текстом-и-современными-моделями', 'краткие-итоги']) expect(html).toContain(`id="${id}"`);
    for (const phrase of ['lift_c4', 'roll', 'Rotation-equivariant feature maps',
      'Copyright and source license', 'NO EXPRESS OR IMPLIED LICENSES',
      'Qualcomm Innovation Center']) expect(text).toContain(phrase);
    for (const id of ['rotation-equivariant-feature-maps',
      'four-orientations-from-one-learned-filter', 'read-the-original-steerable-cnn-demonstration'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/neural-networks/dropout-regularization/"');
    expect(html).toContain('href="/textbook/neural-networks/autoencoder-vae/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders nineteen CNN mechanism anchors and the uninterrupted core in both locales',async()=>{
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix,'textbook/neural-networks/cnn-lenet-resnet/index.html'),'utf8');
      for(const anchor of ['cnn-cross-correlation','cnn-shared-gradients','cnn-learned-filter',
        'cnn-channels','cnn-pointwise','cnn-geometry','cnn-depthwise','cnn-pooling',
        'cnn-receptive-field','cnn-im2col','cnn-lenet','cnn-alexnet-vgg','cnn-resnet',
        'cnn-residual-shapes','cnn-residual-gradient','cnn-resnet18','cnn-causal','cnn-c4','cnn-winograd'])
        expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      const content=html.slice(html.indexOf('id="cnn-cross-correlation"'));
      const sequence=['id="cnn-pooling"','id="cnn-receptive-field"','id="cnn-im2col"',
        'id="cnn-lenet"','id="cnn-resnet"','id="cnn-causal"','id="cnn-c4"','id="cnn-winograd"']
        .map(id=>content.indexOf(id));
      expect(sequence).toEqual([...sequence].sort((a,b)=>a-b));
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(22);
      expect(html).toContain('causal_conv1d');
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders dropout derivations, eleven runnable examples and five full source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/dropout-regularization/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
    for (const name of ['dropout.svg', 'prince-dropout-figure9-9.png', 'data_augmentation.png', 'd2l-training.svg', 'dropconnect-original.png'])
      expect(html).toContain(name);
    for (const phrase of ['all_masks', 'linear_predictions', 'grad=None', 'state_dict()',
      'restore_best_weights', 'val_acc', 'checkpoint_model', 'training=self.training']) expect(text).toContain(phrase);
    for (const id of ['inverted-dropout', 'один-прямой-проход-в-числах', 'weight-decay',
      'data-augmentation', 'early-stopping', 'краткие-итоги', 'dropconnect', 'stochastic-depth']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('/licenses/timm-beabdc3a-Apache-2.0.txt');
    expect(html).not.toContain('katex-error');
  });
  it('renders the four original AE, VAE and dropout illustrations with attribution and no retired copies', async () => {
    const modules = [
      ['dropout-regularization', ['prince-dropout-figure9-9.png'], ['cs231n-dropout-train.jpeg'], ['#page=163']],
      ['autoencoder-vae', ['chervinskii-autoencoder.png', 'prince-vae-posterior-figure17-8.png',
        'prince-vae-reparameterization-figure17-11.png'],
      ['weng-autoencoder.png', 'weng-vae-graphical-model.png', 'weng-reparameterization.png'],
      ['#page=351', '#page=353', 'commons.wikimedia.org/wiki/File:Autoencoder_structure.png']],
    ] as const;
    for (const locale of ['', 'en/']) {
      for (const [route, images, retired, sources] of modules) {
        const html = await readFile(join(distDir, locale, 'textbook/neural-networks', route, 'index.html'), 'utf8');
        for (const name of images) {
          const src = html.match(new RegExp('<img[^>]*src="([^"]*/' + name.replace('.', '\\.') + ')"[^>]*alt="[^"]+"'))?.[1];
          expect(src, name).toBeDefined();
          await access(join(distDir, decodeURIComponent(src!)));
        }
        for (const old of retired) {
          expect(html).not.toContain(old);
          await expect(access(join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/source-first-7-14', old))).rejects.toMatchObject({ code: 'ENOENT' });
        }
        for (const url of sources) expect(html).toContain(url);
        expect(html).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
        expect(html).not.toContain('katex-error');
      }
    }
  });
  it('keeps the regularization core and sixteen subject anchors in both locale routes',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/dropout-regularization/index.html'),'utf8');
      for(const anchor of ['inverted-dropout-expectation','dropout-contract','dropout-forward-backward',
        'dropout-nonlinearity','dropout-train-eval','channel-dropout','dropout-objective','weight-decay',
        'data-augmentation','early-stopping','dropout-learning-curves','regularization-diagnostics',
        'regularization-regimes','regularization-control','dropconnect','stochastic-depth'])
        expect(html).toContain('id="'+anchor+'"');
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
      const ids=['dropout-objective','weight-decay','early-stopping','regularization-control','dropconnect','stochastic-depth'];
      for(let i=1;i<ids.length;i++)
        expect(html.indexOf('id="'+ids[i-1]+'"')).toBeLessThan(html.indexOf('id="'+ids[i]+'"'));
      expect(html).toContain('data_augmentation.png');
      expect(html).toContain('dropconnect-original.png');
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the native-English augmentation owner with its original composition gallery',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/data-augmentation/index.html'),'utf8');
      for(const id of ['augmentation-target','augmentation-crop-scale','augmentation-objective',
        'augmentation-views','augmentation-fill-units','mixup','mixup-beta','cutmix',
        'cutmix-full-step','augmentation-composition','autoaugment-controller','randaugment-n-m',
        'text-audio-augmentation','augmentation-parent-split','augmentation-evaluation','tta-averaging'])
        expect(html).toContain('id="'+id+'"');
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(12);
      for(const figure of ['d2l-random-crops.svg','augmentations.png','cutmix-comparison.png','d2l-composed-transforms.svg'])
        expect(html).toContain(figure);
      expect(html.indexOf('id="augmentation-composition"')).toBeLessThan(html.indexOf('id="autoaugment-controller"'));
      expect(html.indexOf('id="autoaugment-controller"')).toBeLessThan(html.indexOf('id="randaugment-n-m"'));
      expect(html.replace(/<[^>]+>/g,'')).toContain('Complete original eight-panel PyTorch output');
      expect(html).not.toContain('Canonical-language chapter.');
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders evaluation calculations, complete request accounting and four source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/evaluation-operations/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    for (const figure of ['precision-recall.svg', 'diversity.svg', 'linux-foundation-recommenders-architecture.png', 'li-sampled-ranks-figure1.png'])
      expect(html).toContain(figure);
    for (const phrase of ['AP учитывает точность', 'macro average', 'пустой набор', 'col_relevance',
      'recommendForAllUsers(10)', 'intention-to-treat', 'совместимый набор']) expect(text).toContain(phrase);
    for (const id of ['rating-metrics', 'ranking-metrics', 'candidate-set--часть-результата',
      'beyond-accuracy', 're-ranking-как-constrained-optimization', 'production-architecture', 'ab-test'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/sequential-feature-aware/"');
    expect(html).toContain('href="/textbook/neural-networks/mlp/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders sequential windows, causal masks, SSE distinctions and twelve executable examples', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/sequential-feature-aware/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(12);
    for (const name of ['rec-seq-data.svg', 'rec-caser.svg', 'petrov-sasrec.svg', 'petrov-bert4rec.svg']) expect(html).toContain(name);
    const objectiveComparison=html.match(/<a id="sasrec-bert4rec-objectives"><\/a>[\s\S]*?(?=<h3\b)/)?.[0];
    expect(objectiveComparison).toBeDefined();
    for(const image of ['petrov-sasrec.svg','petrov-bert4rec.svg'])
      expect(objectiveComparison).toMatch(new RegExp('<img[^>]+src="[^"]*/'+image.replaceAll('.','\\.')+'"'));
    expect(objectiveComparison).toContain('https://arxiv.org/html/2207.07483v1#S3.F2');
    expect(objectiveComparison).toContain('https://creativecommons.org/licenses/by/4.0/');
    expect(html).not.toContain('cell-001-attachment.png');
    const notebookRoute='sources/courses/lf-recommenders/sasrec-amazon/';
    const notebook=await readFile(join(distDir,notebookRoute,'index.html'),'utf8');
    expect(notebook).toContain('href="https://arxiv.org/html/1808.09781v1#S1.F1"');
    expect(notebook).not.toContain('cell-001-attachment.png');
    const englishNotebook=await readFile(join(distDir,'en',notebookRoute,'index.html'),'utf8');
    expect(englishNotebook).toContain(`content="0;url=/${notebookRoute}"`);
    expect(englishNotebook).not.toContain('cell-001-attachment.png');
    for (const phrase of ['третье полное окно', 'SSE-PT', 'такой случайной замены embeddings нет',
      'сумму координат', 'Нет настоящих targets', 'available_at', 'pad_left']) expect(text).toContain(phrase);
    for (const id of ['как-матрица-теряет-задачу', 'сначала-split-потом-окна', 'caser-свёртки-по-истории',
      'sasrec-causal-self-attention', 'padding-и-mask--разные-вещи', 'long-term-short-term-и-context',
      'feature-rich-ctr-prediction', 'worked-example-field-interactions', 'признаки-без-утечки',
      'cold-start', 'failure-modes-sequence-models', 'production-contract', 'первоисточники-и-продолжение'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/ranking-neural-models/"');
    expect(html).toContain('href="/textbook/recommender-systems/evaluation-operations/"');
    expect(html).not.toContain('katex-error');
  });
  it('connects the cache and sampled-rank examples to the bilingual reference and original sources', async () => {
    const root='/textbook/recommender-systems/';
    const note='/sources/papers/sequence-windows-sampled-evaluation/';
    const routes=[
      ['sequential-feature-aware/', 'sasrec-rolling-window-cache', 'stale_input'],
      ['evaluation-operations/', 'sampled-ranking-order-reversal', 'expected_sampled_hit1'],
    ];
    for(const prefix of ['','/en']){
      const ref=await readFile(join(distDir,prefix,'reference/index.html'),'utf8');
      const sources=await readFile(join(distDir,prefix,note,'index.html'),'utf8');
      for(const [slug,anchor,code] of routes){
        const route=root+slug;
        const html=await readFile(join(distDir,prefix,route,'index.html'),'utf8');
        expect(html).toContain(`id="${anchor}"`);
        expect(html).toContain(code);
        expect(html).toContain(`href="${prefix}${note}"`);
        expect(ref).toContain(`href="${prefix}${route}#${anchor}"`);
        expect(sources).toContain(`href="${prefix}${route}#${anchor}"`);
      }
    }
  });
  it('renders ranking, DLRM and cold-ID selection with twenty-four examples and seven original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/ranking-neural-models/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(24);
    expect(html).toContain('id="cold-embedding-ids"');
    expect(html).toContain('torchrec-zch-dataflow.png');
    for (const name of ['rec-ranking.svg', 'rec-neumf.svg', 'rec-deepfm.svg']) expect(html).toContain(name);
    for (const name of ['torchrec-sharding.png', 'torchrec-forward.png',
      'torchrec-full-training-loop.png']) expect(html).toContain(name);
    for (const phrase of ['Which pairwise products reach the top MLP',
      'Backpropagation through pooling and dot products', '476.84 GiB',
      'Incorrect mean of shard means', 'original DLRM paper']) expect(text).toContain(phrase);
    for (const name of ['dlrm-MIT.txt','torchrec-BSD-3-Clause.txt']) {
      expect(html).toContain('href="/licenses/'+name+'"');
      expect((await readFile(join(distDir,'licenses',name),'utf8')).length).toBeGreaterThan(1000);
    }
    for (const phrase of ['попарного loss', '0,3133', 'частотно-взвешенная регуляризация',
      'маску наблюдений', 'натуральный логарифм', 'Суммарная производная logit']) expect(text).toContain(phrase);
    for (const id of ['pointwise-pairwise-и-listwise-постановки', 'bayesian-personalized-ranking',
      'как-loss-реагирует-на-margin', 'bpr-не-превращает-unseen-в-настоящий-dislike',
      'hinge-ranking-loss', 'autorec-восстановление-строки-матрицы', 'neural-collaborative-filtering',
      'формы-тензоров', 'почему-comparison-часто-нечестен', 'factorization-machines-interactions-между-полями',
      'deepfm-low-order-и-high-order-interactions', 'сквозной-training-protocol-для-ranker',
      'failure-modes', 'что-должно-быть-в-отчёте', 'практика-и-первоисточники']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/collaborative-filtering-matrix-factorization/"');
    expect(html).toContain('href="/textbook/recommender-systems/sequential-feature-aware/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders full collaborative-filtering examples, three source figures and the weighted ALS explanation', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/collaborative-filtering-matrix-factorization/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(12);
    for(const name of ['sar-schema.svg','rec-mf.svg','weighted-objectives.svg']) expect(html).toContain(name);
    for(const phrase of ['Weighted ALS','0,40625','вклад','глобальной регуляризацией',
      'всех факторов','матрицы пользовательских']) expect(text).toContain(phrase);
    for(const id of ['матрица-взаимодействий','три-baseline-до-factorization','memory-based-collaborative-filtering',
      'worked-example-sar','matrix-factorization','один-шаг-sgd','implicit-feedback-ноль-состоит-из-разных-причин',
      'от-обучения-к-выдаче','где-модель-ломается','контрольный-эксперимент','первоисточники-и-продолжение'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/task-data-pipeline/"');
    expect(html).toContain('href="/textbook/recommender-systems/ranking-neural-models/"');
    expect(html).not.toContain('katex-error');
  });
  it('links the implicit-weight and sampler examples from the reference and source note', async () => {
    const root='/textbook/recommender-systems/';
    const source='/sources/papers/implicit-feedback-weights-sampling/';
    const subjects=[
      ['collaborative-filtering-matrix-factorization/', 'als-availability-weights', 'als_without_unavailable'],
      ['ranking-neural-models/', 'bpr-hard-negative-probabilities', 'uniform_target'],
    ];
    for(const prefix of ['','/en']){
      const ref=await readFile(join(distDir,prefix,'reference/index.html'),'utf8');
      const note=await readFile(join(distDir,prefix,source,'index.html'),'utf8');
      for(const [slug,anchor,example] of subjects){
        const route=root+slug;
        const html=await readFile(join(distDir,prefix,route,'index.html'),'utf8');
        expect(html).toContain(`id="${anchor}"`);
        expect(html).toContain(example);
        expect(html).toContain(`href="${prefix}${source}"`);
        expect(ref).toContain(`href="${prefix}${route}#${anchor}"`);
        expect(note).toContain(`href="${prefix}${route}#${anchor}"`);
      }
    }
  });
  it('renders the recommendation-data chapter with eight executable examples and seven source figures', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/task-data-pipeline/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    for(const name of ['google-process.svg','feast-features.png','feast-decisions.png',
      'feast-asof.png','feast-joined.png','rec-ranking.svg','yang-esdfm-figure1.png']) expect(html).toContain(name);
    for(const value of ['999','watermark','filter_by_created_timestamp','available_at',
      'не зарегистрирован целевой отклик']) {
      expect(text.toLowerCase()).toContain(value.toLowerCase());
    }
    for(const id of ['сначала-определить-событие-и-решение','четыре-разные-задачи',
      'rating-prediction','top-kkk-recommendation','ctr-или-response-prediction',
      'next-item-prediction','двухступенчатый-serving','split-должен-имитировать-будущее',
      'worked-example-временного-split','как-строятся-negatives','candidate-generation--часть-задачи',
      'data-contract','baseline-раньше-архитектуры','bias-baseline-как-проверка-dataset',
      'типичные-ошибки-постановки','практика-и-первоисточники']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/collaborative-filtering-matrix-factorization/"');
    expect(html).toContain('href="/licenses/feast-1e5e90ca-Apache-2.0.txt"');
    expect(html).not.toContain('katex-error');
  });
  it('connects the delayed purchase example to the map, reference indexes and original paper', async () => {
    const route='/textbook/recommender-systems/task-data-pipeline/';
    const anchor='#recommendation-delayed-labels';
    const source='/sources/papers/delayed-conversion-feedback/';
    for(const prefix of ['','/en']){
      for(const path of ['textbook/recommender-systems/index.html','reference/index.html']){
        const entry=await readFile(join(distDir,prefix,path),'utf8');
        expect(entry).toContain(`href="${prefix}${route}${anchor}"`);
      }
      const html=await readFile(join(distDir,prefix,route,'index.html'),'utf8');
      const section=html.match(/<a id="recommendation-delayed-labels"><\/a>[\s\S]*?(?=<a id="двухступенчатый-serving")/)?.[0];
      expect(section).toBeDefined();
      expect(section).toMatch(/<img[^>]+src="[^"]*\/yang-esdfm-figure1\.png"/);
      expect(section).toContain('https://arxiv.org/html/2012.03245v1#Sx3.F1');
      expect(section).toContain(`href="${prefix}${source}"`);
      expect(section).toContain('purchase_status');
      expect(section).not.toContain('katex-error');
    }
    const note=await readFile(join(distDir,source,'index.html'),'utf8');
    expect(note).toContain(`href="${route}${anchor}"`);
  });
  it('renders the recommender map with prerequisites, original feedback and process figures', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/index.html'),'utf8');
    for(const name of ['rec-intro.svg','google-process.svg']) expect(html).toContain(name);
    for(const route of ['task-data-pipeline','collaborative-filtering-matrix-factorization',
      'ranking-neural-models','sequential-feature-aware','evaluation-operations'])
      expect(html).toContain(`href="/textbook/recommender-systems/${route}/"`);
    expect(html).toContain('href="/textbook/neural-networks/mlp/"');
    for(const id of ['почему-это-отдельная-дисциплина','три-слоя-системы','минимальный-сквозной-dataset',
      'порядок-чтения','один-запрос-как-маршрут-по-модулю','сквозной-пример','один-лог--разные-targets',
      'два-маршрута-чтения','что-должно-остаться-после-модуля','практика-и-первоисточники'])
      expect(html).toContain(`id="${id}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders kernel and SVM derivations, ten executable examples and ten source figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/kernels-svm-rff/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(10);
    for(const name of ['jake-separators.png','jake-support-vectors.png','jake-support-invariance.png',
      'jake-circles-linear.png','jake-radial-lift.png','jake-circles-rbf.png',
      'jake-soft-margin.png','kernel-approximation.png',
      'sklearn-rbf-decision-grid.png','sklearn-rbf-validation-grid.png']) expect(html).toContain(name);
    for(const phrase of ['Условия KKT','0,0728','псевдообратную',
      'CC BY-NC-ND 3.0 Unported','Как число признаков влияет на ошибку'])
      expect(text).toContain(phrase);
    expect(text.toLowerCase()).toContain('при равенстве');
    for(const id of ['kernel-как-скалярное-произведение','полиномиальный-kernel-без-магии',
      'svm-максимальный-зазор','линейно-разделимый-случай','soft-margin-и-hinge-loss',
      'от-dual-problem-к-kernel-svm','от-объектов-к-gram-matrix',
      'что-означает-kernel-trick-вычислительно','random-fourier-features-явное-приближение-rbf',
      'численный-пример-одного-приближения','алгоритм-rff-pipeline',
      'nyström-другой-путь-к-явным-признакам','практический-выбор',
      'диагностика-и-типичные-ошибки','leakage-и-scaling','совместная-роль-ccc-и-γgammaγ',
      'accuracy-недостаточна','бюджет-inference','случайность-rff','практика-и-первоисточники'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/classical-ml/gmm-em/"');
    expect(html).toContain('href="/textbook/recommender-systems/task-data-pipeline/"');
    const anchors=['kernel-gram','svm-geometric-margin','svm-dual-kkt','kernel-prediction-cost',
      'random-fourier-features','rff-approximation-variance','nystrom-anchor-projection',
      'rbf-c-gamma','rff-fixed-map-scale'];
    for(const anchor of anchors){
      expect(html).toContain('id="'+anchor+'"');
      for(const locale of ['','en/']){
        const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
        expect(index).toContain('href="/'+locale+'textbook/classical-ml/kernels-svm-rff/#'+anchor+'"');
      }
    }
    const english=await readFile(join(distDir,'en/textbook/classical-ml/kernels-svm-rff/index.html'),'utf8');
    expect(english).toContain('Canonical-language chapter.');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(10);
    expect(english).toContain('sklearn-rbf-decision-grid.png');
    expect(english).toContain('sklearn-rbf-validation-grid.png');
    expect(english).not.toContain('katex-error');
    expect(text).toContain('её середина не закреплена в нуле');
    expect(html).not.toContain('katex-error');
  });
  it('renders the Gaussian-mixture derivations, ten executable fences and twelve original figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/gmm-em/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(10);
    for(const name of ['jake-uncertainty.png','jake-covariance-types.png','jake-stretched.png',
      'jake-moons-two.png','jake-moons-sixteen.png','jake-aic-bic.png','gmm-covariances.png'])
      expect(html).toContain(name);
    for(const value of ['Уверенная принадлежность','суммарное обучающее правдоподобие',
      'Stanford CS228','231 новый параметр','CC BY-NC-ND 3.0 Unported','явное ограничение на ковариацию'])
      expect(text).toContain(value);
    expect(html).toContain('id="почему-likelihood-не-уменьшается"');
    const additions=['sklearn-gmm-initialization.png','jake-moons-generated.png',
      'jake-training-digits.png','jake-generated-digits.png','jake-digit-aic.png'];
    for(const name of additions)expect(html).toContain(name);
    const anchors=['gmm-responsibilities','em-bound','gmm-covariance','gmm-covariance-collapse',
      'gmm-initialization','gmm-density-components','gmm-model-selection','gmm-pca-generation'];
    for(const anchor of anchors){
      expect(html).toContain('id="'+anchor+'"');
      for(const locale of ['','en/']){
        const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
        expect(index).toContain('href="/'+locale+'textbook/classical-ml/gmm-em/#'+anchor+'"');
      }
    }
    const english=await readFile(join(distDir,'en/textbook/classical-ml/gmm-em/index.html'),'utf8');
    expect(english).toContain('Canonical-language chapter.');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(10);
    for(const name of additions)expect(english).toContain(name);
    expect(english).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/classical-ml/dimensionality-reduction/"');
    expect(html).toContain('href="/textbook/classical-ml/kernels-svm-rff/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the full PCA/t-SNE/UMAP chapter, nine executable fences and twenty-four original figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/dimensionality-reduction/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
    for(const name of ['jake-pca-projection.png','jake-pca-whitening.png',
      'distill-cluster-sizes-input.png','distill-cluster-sizes-tsne30.png',
      'umap-open-cover.png','umap-adaptive-cover.png','umap-fuzzy-graph.png',
      'pca-iris.png','tsne-perplexity.png']) expect(html).toContain(name);
    for(const value of ['Whitening меняет масштаб','оценка градиента смещена',
      'trustworthiness','CC BY-NC-ND 3.0 Unported','5000 итераций','исходных признаков Iris'])
      expect(text).toContain(value);
    const additions = ['jake-pixel-basis.png','jake-pca-basis.png','jake-cumulative-variance.png',
      'jake-noisy-digits.png','jake-filtered-digits.png','umap-neighbors-2.png',
      'umap-neighbors-200.png','umap-min-dist-0.png','umap-min-dist-099.png',
      'distill-global-150-input.png','distill-global-150-p50.png',
      'distill-global-600-input.png','distill-global-600-p50.png',
      'distill-noise-input.png','distill-noise-p2.png'];
    for (const name of additions) expect(html).toContain('/'+name);
    const english = await readFile(join(distDir,'en/textbook/classical-ml/dimensionality-reduction/index.html'),'utf8');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(9);
    for (const name of additions) expect(english).toContain('/'+name);
    for (const anchor of ['pca-image-basis','pca-cumulative-variance','pca-noise-filtering',
      'pca-whitening','tsne-perplexity','tsne-map-interpretation','umap-neighbors-min-dist','trustworthiness']) {
      expect(html).toContain('id="'+anchor+'"');
      expect(english).toContain('id="'+anchor+'"');
    }
    expect(english).not.toContain('katex-error');
    expect(html).toContain('id="где-pca-обманывает"');
    expect(html).toContain('id="сравнение-без-ложных-обещаний"');
    expect(html).toContain('href="/textbook/classical-ml/spectral-clustering/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the spectral-clustering derivation, original figures and six actual code examples', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/spectral-clustering/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for(const name of ['wiskott-connected-heat.png','wiskott-weighted-constraint.png',
      'wiskott-heat-components.png','sklearn-segmentation-four.png',
      'sklearn-segmentation-two.png','spectral-clustering-comparison.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const value of ['Normalized Cut','Ng–Jordan–Weiss','Shi–Malik',
      '1.9.1','0,516047','теплоёмкостью','стационарного распределения',
      'Fabian Schönfeld','CC BY-SA 4.0','commute time','0,999994',
      'h_chain','L_pinv']) expect(text).toContain(value);
    for(const route of ['clustering','dimensionality-reduction'])
      expect(html).toContain('href="/textbook/classical-ml/'+route+'/"');
    const english = await readFile(join(distDir,'en/textbook/classical-ml/spectral-clustering/index.html'),'utf8');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['wiskott-connected-heat.png', 'wiskott-weighted-constraint.png'])
      expect(english).toContain('/'+name);
    for (const anchor of ['normalized-cut','degree-weighted-constraint','commute-time','njw-row-normalization']) {
      expect(html).toContain('id="'+anchor+'"');
      expect(english).toContain('id="'+anchor+'"');
    }
    expect(english).toContain('href="/en/textbook/classical-ml/dimensionality-reduction/"');
    expect(english).not.toContain('katex-error');
    expect(html).not.toContain('katex-error');
  });
  it('renders all four original Lloyd steps and the normalized-center counterexample', async () => {
    const html = await readFile(join(distDir,'textbook/classical-ml/clustering/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
    expect(text).toContain('Transformed counts:');
    expect(text).toContain('count_positions');
    for (const step of [1,2,3,4])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/lloyd-step-'+step+'\\.svg"'));
    for (const value of ['Weston.pace','CC BY-SA 3.0','0,64','0,4',
      'Dhillon','Нормировка среднего','не могут взаимно уничтожиться',
      'mutual','AMI','approximate_predict']) expect(text).toContain(value);
    expect(html).toContain('href="/textbook/classical-ml/spectral-clustering/"');
    expect(html).not.toContain('katex-error');
  });
  it('connects both scheduling editions to the interactive allocation exercise', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale + 'textbook/inference/scheduling/index.html'), 'utf8');
      expect(html).toContain('href="/' + locale + 'practice/inference-scheduler/"');
      expect(html).toContain('Interactive scheduler and KV-memory lab');
      expect(html).toContain('does not run a neural network or predict GPU latency');
    }
  });
  it('renders a working canonical-language scheduler in EN instead of redirecting to RU', async () => {
    const html = await readFile(join(distDir, 'en/practice/inference-scheduler/index.html'), 'utf8');
    expect(html).not.toContain('http-equiv="refresh"');
    expect(html).toContain('data-scheduler-lab');
    expect(html).toContain('id="scheduler-step"');
    expect(html).toContain('Canonical-language exercise.');
    expect(html).toContain('href="/en/textbook/inference/scheduling/"');
  });
  it('renders both RoPE labs, their static results and their reading-path links', async () => {
    for (const lang of ['', 'en/']) {
      const lab = await readFile(join(distDir, lang, 'practice/rope/index.html'), 'utf8');
      const chapter = await readFile(join(distDir, lang, 'textbook/modern-llm/rope/index.html'), 'utf8');
      const index = await readFile(join(distDir, lang, 'practice/index.html'), 'utf8');
      expect(lab).not.toContain('http-equiv="refresh"');
      expect(lab).not.toContain('katex-error');
      expect(lab).toContain('data-rope-lab');
      for (const id of ['rope-rotation-lab', 'rope-frequency-lab', 'rope-query', 'rope-shift',
        'rope-pair', 'rope-distance', 'rope-reset']) expect(lab).toContain('id="' + id + '"');
      expect(lab).toContain('href="/' + lang + 'textbook/modern-llm/rope/#rope-llama31-scaling"');
      expect(chapter).toContain('href="/' + lang + 'practice/rope/"');
      expect(index).toContain('href="/' + lang + 'practice/rope/"');
      expect(lab).toContain(lang ? '0.86540' : '0,86540');
      expect(lab).toContain(lang ? 'Rotate the query and key' : 'Поверните запрос и ключ');
    }
  });
  it('renders the mixed scheduling walkthrough, two complete HF originals and six executable fences', async () => {
    const html = await readFile(join(distDir,'textbook/inference/scheduling/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['dynamic','continuous'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/scheduling-2026-10-25/hf-'+name+'-batching\\.png"'));
    for (const name of ['vllm-prefix-cache-overview.png','vllm-prefix-release-shared-blocks.png','sglang-radix-tree-original.jpg'])
      expect(html).toContain(name);
    for (const value of ['packed_causal_mask','key_start','CC BY-NC-SA 4.0',
      'Rémi Ouazan Reboul','пятнадцати столбцов','num_computed_tokens'])
      expect(text).toContain(value);
    expect(html).toContain('href="/textbook/inference/serving-engines/#'+
      encodeURIComponent('проверка-упакованного-внимания-на-cpu')+'"');
    expect(html).not.toContain('katex-error');
  });
  it('uses the licensed original static and chunked schedules in both editions', async () => {
    const figures = [
      ['hf-static-batching.png', 'b27e333c14ad5e63c1576c9e2a20599041723725d5980824546bba0a130d79c0'],
      ['deepspeed-paper-figure3.png', 'ec8a137c8edd27626b87e0ef714d479a9fec28787ba6ec31d4fe377d0d55590c'],
    ];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/inference/scheduling/index.html'), 'utf8');
      for (const [file, hash] of figures) {
        const img = html.match(new RegExp('<img[^>]+src="([^"]*/scheduling-source-2026-11-11/' +
          file.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(img).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(img!)))).digest('hex'))
          .toBe(hash);
      }
      for (const name of ['anyscale-static-batching.png', 'gordic-chunked-prefill.png'])
        expect(html).not.toContain(name);
      for (const link of ['https://www.anyscale.com/blog/continuous-batching-llm-inference',
        'https://www.aleksagordic.com/blog/vllm#chunked-prefill',
        'https://huggingface.co/datasets/huggingface/documentation-images/blob/c8c3cf72b3b724ce734259e4a086d30618a9198b/README.md', 'https://arxiv.org/html/2401.08671v1#S3.F3', '/' + prefix + 'sources/papers/deepspeed-fastgen/'])
        expect(html).toContain(link);
      expect(html).toContain(prefix ? 'early vLLM version' : 'ранний vLLM');
      expect(html).not.toContain('katex-error');
    }
  });
  it('preserves five actual KV-cache examples, complete source diagrams and adjacent routes', async () => {
    const html = await readFile(join(distDir,'textbook/inference/kv-cache-batching-pagedattention/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html).toMatch(/<img[^>]+src="[^"]*attention-review-2026-09\/prefill-decode\.png"/);
    expect(html).not.toMatch(/<img[^>]+src="[^"]*topics-53-59-source-first\/prefill-decode\.png"/);
    for (const name of ['paged-attention-fragmentation','paged-attention-blocks',
      'paged-attention-logical','paged-attention-sharing','paged-attention-copy-on-write'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name+'\\.png"'));
    for (const value of ['Baolin Li','DynamicCache','cache_position','position_ids',
      'cached_logits','generation_cache','padding_cache','131072','copy-on-write'])
      expect(text).toContain(value);
    expect(html).toContain('href="/textbook/inference/llm-inference-physics/"');
    expect(html).toContain('href="/textbook/inference/scheduling/"');
    expect(html).not.toContain('katex-error');
  });
  it('links the CS336 prefill figure to the same licensed original and cache states in both editions', async () => {
    const route = 'textbook/inference/kv-cache-batching-pagedattention';
    const anchor = 'prefill-decode-cache-states';
    const figure = 'attention-review-2026-09/prefill-decode.png';
    const digest = '947a70b7cef4d1875d295692731cd4216f189daaf5ac7b586106fc542626e3a6';
    const hub = 'sources/courses/stanford-cs336-spring-2026';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const start = html.indexOf('id="' + anchor + '"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start).split(/<\/section>|<h2\b[^>]*id="(?:что-именно-сохраняет-kv-cache|what-the-kv-cache-stores)"/)[0];
      const src = [...section.matchAll(/<img[^>]+src="([^"]+)"/g)]
        .map(match => match[1]).filter(url => url.endsWith(figure));
      expect(src).toHaveLength(1);
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src[0])))).digest('hex')).toBe(digest);
      for (const value of ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4',
        'S+1', 'S+3', 'CC BY 4.0', 'https://arxiv.org/html/2407.12391v1#S2.F2'])
        expect(section).toContain(value);
      expect(section).toMatch(/<code[^>]*>(?:&#x3[Cc];|&lt;)EOS(?:>|&gt;|&#x3[Ee];)<\/code>/);
      expect(section).not.toContain('katex-error');
      expect(html).not.toContain('aria-label="English edition status"');
      const course = await readFile(join(distDir, prefix, prefix ? 'sources/kursy' : hub, 'index.html'), 'utf8');
      expect(course).toContain('href="/' + prefix + route + '/#' + anchor + '"');
    }
  });
  it('renders six executable decoding examples, source illustrations and stop/beam limits', async () => {
    const html = await readFile(join(distDir,'textbook/inference/decoding/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['greedy_search','beam_search','sampling_search',
      'sampling_search_with_temp','top_k_sampling','top_p_sampling'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/hf-'+name+'\\.png"'));
    for (const value of ['probabilities_at','k_then_p','p_then_k','prompt_ignore_length',
      'StopStringCriteria','beam_indices','compute_transition_scores',
      '0,099','0,02','0,68','0,99','0,0816']) {
      expect(text).toContain(value);
    }
    expect(text).not.toContain('Top-k на реальном распределении GPT-2');
    expect(text).not.toContain('chapter 8');
    expect(text).toContain('speculative sampling и Triton');
    expect(html).toContain('href="/textbook/pretraining/synthetic-data-curricula/"');
    expect(html).toContain('href="/textbook/inference/llm-inference-physics/"');
    expect(html).not.toContain('katex-error');
  });
  it('preserves synthetic-data examples, original figures and the linear chapter route', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/synthetic-data-curricula/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    expect(html.match(/<pre data-language="bash"/g)).toHaveLength(1);
    for (const value of ['b-cat', '970 млрд', '10 млн отобранного', '1,6 трлн',
      'Self-Instruct', 'OpenThoughts', 'SWE-Rebench', 'Gerstgrasser'])
      expect(text).toContain(value);
    for (const name of ['nvidia-generate-critique-filter.png', 'self-instruct-pipeline.png',
      'nemotron-transformations.png', 'openthoughts-pipeline.png', 'swe-smith.png',
      'swe-rebench.png', 'swezero-noexec.png', 'olmo2-dolmino.png', 'shumailov-collapse.png',
      'phi1-data-ablation.png', 'phi4-repetitions.png', 'nemotron4-prompts.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['synthetic-transformations', 'teacher-data',
      'posttraining-environments', 'curricula-replay', 'synthetic-evidence',
      'phi-synthetic-recipes', 'nemotron4-synthetic-alignment'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('href="/textbook/reasoning/test-time-compute/"');
    expect(html).toContain('href="/textbook/inference/decoding/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders RLVR reward gradients, prompt/grader contracts and complete Tulu evidence', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/rlvr-verifiers/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const value of ['ascent_gradient', 'updated_reward', 'log_failure', 'include_stop_str_in_output',
      '0.13125', '0.488', 'allow_set_relation_comp', 'градиентный подъём',
      'single_tool_reward', 'completion_efficiency_reward', 'greedily consumes']) expect(text).toContain(value);
    expect(html).toContain('id="agent-reward-design"');
    expect(html).toContain('/licenses/toolrl-8cee13ec-Apache-2.0.txt');
    expect(html).toMatch(/<img[^>]+src="[^"]*toolrl\/reward-design\.png"/);
    for (const name of ['tulu-rlvr-loop.png', 'tulu-rlvr-full-figure19.png',
      'tulu-ifeval-kl-and-generalization.png', 'tulu-value-initialization.png', 'tulu-rlvr-reward-mixing.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*tulu-rlvr-training-curves\.png/);
    for (const id of ['rlvr-contract', 'verifiable-reward', 'parser-vs-verifier', 'grader-contract', 'verifier-audit', 'rlvr-failures', 'главное-различие'])
      expect(html).toContain(`id="${id}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders all RLVR subject links, original figures and rollout-worker reasoning in both locales', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-rlvr-subject-systems-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(30);
    for(const locale of ['', 'en/']){
      const route=locale+'textbook/post-training/rlvr-verifiers';
      const html=await readFile(join(distDir,route+'/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+route+'/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(7);
      expect(html).toContain('second-latest inference');
      expect(html).toContain('/'+locale+'textbook/post-training/grpo-deepseek-r1/#offpolicy-ratios');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('shows the pass@k crossover with the original mathematical comparison in both RLVR routes', async () => {
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/rlvr-verifiers/index.html'), 'utf8');
      const section = html.match(/<a id="rlvr-pass-k-crossover"><\/a>[\s\S]*?(?=<a id="rlvr-tulu-experiment")/)?.[0];
      expect(section).toBeDefined();
      const src = section!.match(/<img[^>]+src="([^"]*\/yue-figure-2\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!.slice(base.length))))).digest('hex'))
        .toBe('408381de34bfbf22e1119f58b395236629996be0ebb1648c4848b56730b873c2');
      for (const text of ['https://arxiv.org/html/2504.13837v5#S2.F2', 'https://creativecommons.org/licenses/by/4.0/',
        'https://arxiv.org/html/2504.13837v5#S4.T2', '90,7%', '53,9%', '480 против 467', 'boxnet',
        `href="${base}/${locale}textbook/reasoning/test-time-compute/"`]) expect(section).toContain(text);
    }
  });
  it('renders actual DPO training, iterative pairing and complete source Tulu figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/dpo/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['optimal_probability', 'recovered_reward', 'model_response_logp',
      'pair_attention', 'reference_pair_logp', 'updated_loss', 'reasoning_pairs',
      '0.713015', '9/13', '0.425557', '120 тысяч']) expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
    for (const name of ['fig1.png', 'reward-kl-frontier.png', 'm02-p74-irpo-loop.png',
      'tulu-preference-pipeline.png', 'tulu-prompt-policy-ablations.png',
      'paired-results.png', 'math-prompts.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['dpo-rhf-objective', 'dpo-derivation', 'dpo-dataflow', 'dpo-offline-boundary', 'dpo-failures', 'проверка-понимания'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders DPO subject anchors and the normalized-policy example in both locale routes', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-dpo-mass-kl-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(26);
    for(const locale of ['', 'en/']){
      const route=locale+'textbook/post-training/dpo';
      const html=await readFile(join(distDir,route+'/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+route+'/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(7);
      expect(html).toContain('mass_exact_kl');
      expect(html).toContain('paired-results.png');
      expect(html).toContain('math-prompts.svg');
      expect(html).toContain('id="dpo-ppo-rollout-prompts"');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders all six executable PPO fences, original figures and stable mechanism anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/ppo/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['baseline_results', 'ppo_targets', 'ppo_clip_loss',
      '0.499425', 'expected_gradient', 'response_forward', 'sampled_logp',
      'clip_grad_norm_', '0.0475'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['value-fn-training.png', 'rlhf-overview.png', 'ppo-clip-viz.png', 'ppo-framework.png',
      'areal-interruptible-timeline.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['reinforce', 'baseline-advantage', 'ppo-roles', 'ppo-clip', 'ppo-reward-shaping', 'ppo-stale-rollouts',
      'ppo-token-update', 'ppo-areal'])
      expect(html).toContain(`id="${anchor}"`);
    const en = await readFile(join(distDir, 'en/textbook/post-training/ppo/index.html'), 'utf8');
    expectCanonicalEditionNotice(en);
    expect(en).toContain('id="ppo-token-update"');
    expect(en).toContain('areal-interruptible-timeline.png');
    expect(html).not.toContain('katex-error');
  });
  it('renders runnable reward modeling, listwise/step examples and the complete calibration figure', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/reward-modeling/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['LlamaModel', 'position_ids', 'left_score', 'plackett_luce_loss',
      '0.380952', '0.355556', 'prm_product', 'calibration_loss', '0.929074',
      '47,4%', '1,949', '0,526', 'Что меняется при PPO'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
    for (const name of ['reward-model-hf.png', 'preference-rm-training.png',
      'prm-step-supervision.png', 'self-rewarding-loop.png', 'm02-p89-meta-rewarding-loop.png',
      'evalplanner-plan-execution-loop.png', 'reward-overoptimization-best-of-n.png', 'reward-overoptimization-ppo.png', 'hh-calibration.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).toContain('id="bradley-terry"');
    expect(html).toContain('id="reward-overoptimization"');
    expect(html).not.toContain('katex-error');
  });
  it('resolves preference/reward subject entries to both complete locale routes', async () => {
    const rows = [
      { route: 'preference-data', code: 5, figures: 6, anchor: 'pref-ranking-weights' },
      { route: 'reward-modeling', code: 7, figures: 9, anchor: 'prm-boundary-loss' },
    ];
    for (const locale of ['', 'en/']) {
      const reference = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
      for (const row of rows) {
        const route = 'textbook/post-training/' + row.route;
        const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
        expect(html.match(/<pre data-language="python"/g)).toHaveLength(row.code);
        expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(row.figures);
        expect(html).toContain('id="' + row.anchor + '"');
        expect(reference).toContain('href="/' + locale + route + '/#' + row.anchor + '"');
        expect(html).not.toContain('katex-error');
      }
    }
  });
  it('renders executable preference preparation with the complete licensed primary figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/preference-data/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['final_turn_pair', 'normalize_verdict', 'ranking_to_pairs',
      'rank_groups', 'validation_ids', 'Chosen', 'Rejected',
      'Do you know why turkeys became the official food of thanksgiving?'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html.match(/<pre data-language="json"/g)).toHaveLength(1);
    for (const name of ['hh-workflow.png', 'hh-feedback-interface.png', 'hh-data-distributions.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(instructgpt-labeling-ui-002|preference-collection-interface)\.png/);
    expect(html).toContain('id="ранжирование-kkk-ответов"');
    expect(text).toContain('cycle_probability');
    expect(text).toContain('9,5%');
    expect(text).toContain('три вероятности');
    expect(html).toContain('id="задания"');
    expect(html).not.toContain('katex-error');
  });
  it('keeps the Tülu preference pipeline and the two data-scaling experiments distinct', async () => {
    for (const locale of ['', 'en/']) {
      const route = locale + 'textbook/post-training/';
      const html = await readFile(join(distDir, route + 'preference-data/index.html'), 'utf8');
      const pipeline = html.match(/id="pref-tulu-rating-to-pair"[\s\S]*?(?=<a id="pref-rankings")/)?.[0];
      const scaling = html.match(/id="pref-tulu-pairs-versus-prompts"[\s\S]*?(?=<a id="pref-bt-observation")/)?.[0];
      expect(pipeline).toContain('href="/' + route + 'reward-modeling/#bradley-terry"');
      expect(pipeline).toContain('4,75');
      expect(scaling).toContain('383');
      expect(scaling).toContain('64 тыс.');
      for (const [section, suffix, ref] of [
        [pipeline, 'tulu3-figure7-preference-pipeline.svg', 'S5.F7'],
        [scaling, 'tulu3-figure8-unique-prompts.svg', 'S5.F8'],
        [scaling, 'tulu3-figure9-more-pairs.svg', 'S5.F9'],
      ]) {
        expect(section).toMatch(new RegExp('<img[^>]*src="[^"]*/' + suffix.replaceAll('.', '\\.') + '"[^>]*alt="[^"]+"'));
        expect(section).toContain('https://arxiv.org/html/2411.15124v5#' + ref);
      }
    }
  });
  it('renders executable SFT loss/update/packing and preserves six attributed illustrations', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/sft-instruction-data/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['assistant_only_loss', 'LlamaConfig', 'padded_assistant',
      'block_mask', 'ordinary_changed', '4.840772', '900 000', '§4.2.1'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    for (const name of ['rlhf-basic.png', 'sft-chat-template.png',
      'self-instruct-pipeline.png', 'm04-p30-training-data-gates.png',
      'm04-p54-voting-filter.png', 'minicpm-two-stage-mixture.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(text).not.toContain('В дополнительной части Assignment 5 упаковка должна сохранять EOS');
    expect(html).not.toContain('katex-error');
  });
  it('renders the original Tulu overview without treating the methods as one compulsory chain', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/module-map/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/tulu3-overview\.png"/);
    expect(text).toContain('не образуют обязательную последовательность');
    expect(text).toContain('Unseen evals');
    expect(html).toContain('id="tülu-как-сквозной-открытый-recipe"');
    expect(text).not.toContain('полезен здесь не как ещё одно имя');
    expect(html).not.toContain('katex-error');
  });
  it('renders the SFT accumulation example, held-out FLAN task and original Tulu sweep', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/sft-instruction-data/index.html'), 'utf8');
      expect(html).toContain(`href="/${locale}textbook/post-training/module-map/#posttraining-one-prompt"`);
      const accumulation = html.match(/id="sft-accumulation-denominator"[\s\S]*?(?=<h2\b)/)?.[0];
      expect(accumulation).toBeDefined();
      for (const claim of ['1/16', '1/10', 'вчетверо', 'huggingface.co/docs/transformers/grad_accumulation'])
        expect(accumulation).toContain(claim);
      const flan = html.match(/id="sft-flan-held-out-task"[\s\S]*?(?=<h3\b)/)?.[0];
      expect(flan).toContain('https://arxiv.org/html/2109.01652v5#S2');
      expect(flan).toContain('A cat sleeps.');
      expect(flan).toContain('при предобучении');
      const sweep = html.match(/id="sft-tulu-loss-sweep"[\s\S]*?(?=<h2\b)/)?.[0];
      expect(sweep).toContain('https://arxiv.org/html/2411.15124v5#S4.F5');
      expect(sweep).toContain('не сравнивает сумму с корректно вычисленным');
      const path = sweep!.match(/<img[^>]+src="([^"]*\/tulu3-figure5-loss-vs-lr\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(path).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(path!)))).digest('hex'))
        .toBe('6cda6319b36a296336236f2bba5e57873fec0b824cc4ce439b24e6ef29baf028');
      expect(html).not.toContain('katex-error');
    }
  });
  it('keeps training rewards separate from Tulu stage evaluation scores', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/module-map/index.html'), 'utf8');
      const exercise = html.match(/id="posttraining-one-prompt"[\s\S]*?(?=<a id="tülu-как-сквозной-открытый-recipe")/)?.[0];
      expect(exercise).toBeDefined();
      const explanation = exercise!.replace(/<[^>]*>/g, '');
      for (const marker of ['84 = 2·30 + 24', '84 − 30 = 6', 'Ответ: 3', 'НОД = 6'])
        expect(explanation).toContain(marker);
      for (const url of ['https://arxiv.org/html/2203.02155#S3.SS1', 'https://arxiv.org/html/2305.18290#S4'])
        expect(exercise).toContain(url);
      const scores = exercise!.match(/<tbody>[\s\S]*?<\/tbody>/)?.[0];
      expect([...scores!.matchAll(/<td[^>]*>([01])<\/td>/g)].map(match => match[1])).toEqual(['1', '1', '0', '0']);
      const compare = html.match(/id="tulu-stage-comparison"[\s\S]*?(?=<h2\b)/)?.[0];
      expect(compare).toBeDefined();
      expect(compare).toContain('https://arxiv.org/html/2411.15124v5#S7.T31');
      for (const row of [/31,5[\s\S]*?42,0[\s\S]*?43,7/, /32,3[\s\S]*?33,0[\s\S]*?35,4/, /11,5[\s\S]*?9,5[\s\S]*?7,4/]) {
        expect(compare).toMatch(row);
      }
      const original = compare!.match(/<img[^>]+src="([^"]*\/tulu3-figure23-final-rlvr\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(original).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(original!)))).digest('hex'))
        .toBe('d1aa186a4c2c7045da01ca5bd1a76a7d654586d8f686281918630972737fb4d5');
    }
  });
  it('keeps the Constitutional AI prerequisite route and complete worked examples in both interfaces', async () => {
    for (const locale of ['', 'en/']) {
      const cases = [
        ['module-map', 'next', 'sft-instruction-data'],
        ['sft-instruction-data', 'prev', 'module-map'],
        ['ppo', 'next', 'alignment-goals'],
        ['alignment-goals', 'prev', 'ppo'],
        ['alignment-goals', 'next', 'dpo'],
        ['dpo', 'prev', 'alignment-goals'],
      ];
      for (const [source, rel, target] of cases) {
        const html = await readFile(join(distDir, locale,
          'textbook/post-training', source, 'index.html'), 'utf8');
        const button = (html.match(/<a\b[^>]*>/g) ?? [])
          .find(tag => tag.includes('rel="' + rel + '"'));
        expect(button, locale + source + ': ' + rel)
          .toContain('href="/' + locale + 'textbook/post-training/' + target + '/"');
      }
      const html = await readFile(join(distDir, locale,
        'textbook/post-training/alignment-goals/index.html'), 'utf8');
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
      for (const image of ['constitutional-ai-overview.png', 'revisions-tradeoff.png',
        'human-elo-frontier.png', 'absolute-harmfulness.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^\"]*/' + image.replaceAll('.', '\\.') + '"'));
      expect(html).toContain('href="/' + locale + 'textbook/post-training/reward-modeling/#bradley-terry"');
      expect(html).toContain('id="cai-human-elo"');
      expect(html).toContain('id="cai-harmfulness-score"');
      expect(html).not.toContain('can be read before SFT');
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the full-group restart example, straggler calculation and original scheduling plots', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/fault-tolerance-orchestration/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['run_recovery', 'pack_state', 'INJECTED_EXIT', 'TORCH_DISABLE_SHARE_RDZV_TCP_STORE',
      'LOADED', 'VERIFIED', '900', '19 из 64 GPU'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    for (const name of ['harvard-checkpoint-tradeoff.png', 'harvard-cluster-fragmentation.png',
      'failure-domains.svg', 'failure-types.svg', 'gang-deadlock.svg', 'topology-placement.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(young-daly-optimization|checkpoint-recovery-timeline)\.svg/);
    expect(html).not.toContain('katex-error');
  });
  it('renders executable DCP/DTensor/FSDP restoration and distinguishes DeepSpeed UCP', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/network-storage-checkpoints/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['reshard_vector', 'verify_resume', 'prepare_fsdp_training',
      'DeepSpeed Universal Checkpointing', '716.1 TB'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    for (const name of ['jax-superpod-diagram.png', 'deepspeed-ucp-flowchart.png',
      'openhackathons-gpudirect-rdma.png', 'pytorch-async-checkpoint-flow.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*five-level-model\.svg/);
    expect(html).not.toContain('katex-error');
  });
  it('renders network/checkpoint and recovery subject links in both locale routes', async()=>{
    const specs=[
      {route:'network-storage-checkpoints',terms:36,programs:8},
      {route:'fault-tolerance-orchestration',terms:31,programs:2},
    ];
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-network-fault-subject-additions.json'),'utf8'));
    for(const [i,spec]of specs.entries()){
      const source=await readFile(join(rootDir,journal.files[i].source),'utf8');
      const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
      expect(anchors).toHaveLength(spec.terms);
      for(const locale of ['', 'en/']){
        const route=locale+'textbook/pretraining/'+spec.route;
        const html=await readFile(join(distDir,route+'/index.html'),'utf8');
        const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
        for(const anchor of anchors){
          expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
          expect(index).toContain('href="/'+route+'/#'+anchor+'"');
        }
        expect(html.match(/<pre data-language="python"/g)).toHaveLength(spec.programs);
        expect(html).not.toContain('katex-error');
        if(i===0)expect(html).toContain('stager создало');
        if(locale)expectCanonicalEditionNotice(html);
      }
    }
  });
  it('renders actual EP dispatch code, FSDP motivation and the full source EP illustration', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/expert-hybrid-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['ExchangeRows', 'expert_parallel_linear', 'received_experts', 'grad_rows',
      '31.5 MiB', '21 GB', '57 ms', 'ETP=1, EP=64', '-0.375'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    for (const name of ['hf-ep-layer.svg', 'mcore-token-capacity.png',
      'edls-moe-tensor-parallel.png', 'edls-moe-expert-parallel.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).toContain('id="tp-или-ep"');
    expect(html).not.toContain('katex-error');
  });
  it('resolves expert-parallel subjects to the full canonical explanation in both locales', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-expert-hybrid-subject-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(31);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/expert-hybrid-parallelism/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+locale+'textbook/pretraining/expert-hybrid-parallelism/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
      expect(html.match(/class="katex-display"/g)).toHaveLength(11);
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders FSDP ownership, actual hybrid reductions and six complete source illustrations', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/zero-fsdp2-devicemesh-dtensor/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['owned_adam_step', 'correct_relu', 'wrong_relu', 'row_sum_shard',
      'reduce_scatter_single', '29.3125 GB', '15.53125 GB', '7 GB', '5.25 GB'])
      expect(text).toContain(value);
    for (const name of ['hf-zero-memory.png', 'hf-fsdp-forward.png', 'hf-fsdp-backward.png',
      'pytorch-fsdp-figure1-overview.png', 'pytorch-fsdp-figure4-hybrid-groups.png', 'edls-dtensor-layouts.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-distributed-fsdp"');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/pipeline-parallelism/"');
  });
  it('renders FSDP subjects and two-level HSDP accumulation in both locale routes', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-fsdp-hsdp-accumulation-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(26);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/zero-fsdp2-devicemesh-dtensor/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+locale+'textbook/pretraining/zero-fsdp2-devicemesh-dtensor/#'+anchor+'"');
      }
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const value of ['hsdp_accumulated_step','model.set_requires_gradient_sync(True)',
        'model.set_requires_all_reduce(index == len(window) - 1)',
        'No valid targets in the global window'])expect(text).toContain(value);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
      expect(html.match(/class="katex-display"/g)).toHaveLength(4);
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders pipeline autograd and nine original figures including complete Zero Bubble schedules', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/pipeline-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['pipeline_step', 'partition_costs', 'saved_x.T @ saved_dy',
      '384 MiB', '896 MiB', '12 GiB', '28 GiB', '4/8/6', 'flex_pp', 'post-validation', 'BatchNorm'])
      expect(text).toContain(value);
    for (const name of ['hf-naive-pipeline.svg', 'hf-pipeline-gpipe.svg',
      'hf-pipeline-1f1b.svg', 'hf-interleaved-1f1b.svg', 'dependency.png',
      'zb1p.png', 'zb2p.png', 'zbv.png', 'optimizer.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(7);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-distributed-pipeline"');
    expect(html).toContain('href="/licenses/zero-bubble-Apache-2.0.txt"');
    expect(text).toContain('24 такта работы');
    expect(text).toContain('семь наборов активаций');
    expect(html).toContain('href="/textbook/pretraining/tensor-sequence-parallelism/"');
    expect(html).toContain('href="/textbook/pretraining/zero-fsdp2-devicemesh-dtensor/"');
  });
  it('renders vocabulary-parallel loss, TP arithmetic, FSDP shapes and animated context-parallel sources', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/tensor-sequence-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['vocab_cross_entropy', '0.440190', 'mlp_backward',
      '179.2e12', '4.027 ms', '2.600 ms', 'loss_parallel', '84', '28 MiB', 'FSDP×TP'])
      expect(text).toContain(value);
    for (const name of ['tensor-parallel-split.svg', 'hf-attention-tp.png', 'hf-tp-sp-regions.png',
      'hf-causal-zigzag.svg', 'ulysses-design.png', 'hf-ring-attention.gif'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/gradient-checkpointing-offload/"');
    expect(html).toContain('href="/textbook/pretraining/pipeline-parallelism/"');
  });
  it('renders all pipeline subjects in both locale routes', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-pipeline-subject-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(29);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/pipeline-parallelism/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+locale+'textbook/pretraining/pipeline-parallelism/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
      expect(html.match(/class="katex-display"/g)).toHaveLength(11);
      expect(html.match(/<img[^>]+src="[^\"]+\.(?:svg|png)"/g)?.filter(s=>s.includes('/curated/'))).toHaveLength(9);
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders all TP subjects and the unchanged EDLS classes in both locale routes', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-tensor-parallel-source-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(31);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/tensor-sequence-parallelism/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+locale+'textbook/pretraining/tensor-sequence-parallelism/#'+anchor+'"');
      }
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const value of ['class LlamaMLP','class ComputeWithAllReduce','class AllReduceModule',
        'input.detach().requires_grad_(input.requires_grad)','не сохраняет состояние RNG',
        'Ещё одна TP-сумма'])expect(text).toContain(value);
      expect(html).toContain('href="/licenses/edls-e632aa89-MIT.txt"');
      expect(html).not.toContain('katex-error');
      if(locale)expect(text).toContain('no separate English edition');
    }
  });
  it('renders checkpoint storage aliasing, budget selection, originals and complete code', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/gradient-checkpointing-offload/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['view = base[:1]', '4 194 304', 'choose_saved',
      'torch._functorch.config', '7 ms', '1.4317 с', '2.75', '3.75'])
      expect(text).toContain(value);
    for (const name of ['pytorch-eager-saved-tensors.png', 'pytorch-memory-budget-pareto.png',
      'pytorch-checkpoint.png', 'pytorch-selective-checkpoint.png', 'deepspeed-optimizer-offload.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(9);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/processes-collectives-ddp/"');
    expect(html).toContain('href="/textbook/pretraining/tensor-sequence-parallelism/"');
  });
  it('renders DDP flattening, explicit ring rounds, DP capacity and source compression diagrams', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['flat = torch.cat', '[2,4,6,8,10,12]',
      'alternate_ring_all_reduce', '5,285 ms', '4,624 ms', '5,395 ms',
      'до 13 ms', 'до 11 ms', 'Algorithm 1', 'divide_by_initial_world_size'])
      expect(text).toContain(value);
    for (const name of ['hf-ddp-sequential.svg', 'hf-ddp-overlap.svg', 'edls-error-feedback.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(15);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/edls-e632aa89-MIT.txt"');
    expect(html).toContain('href="/textbook/pretraining/gradient-checkpointing-offload/"');
  });
  it('renders scaling-law fitting steps, original ablation plots and task calibration', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/scaling-laws/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['three_point_minimum', 'N=6.236B, D=80.185B', 'L-BFGS',
      'LSE', 'However, this led to poor results at smaller batch sizes',
      'the attention logits in a Transformer', 'MuAdam', 'ARC Challenge',
      'липшицева', 'n^{-2/(d+2)}'])
      expect(text).toContain(value);
    for (const name of ['porian-figure1.svg', 'sp-coordinates.png', 'mup-coordinates.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(2);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/mup-19814971-MIT.txt"');
    expect(html).toContain('href="/textbook/pretraining/next-token-objective/"');
    expect(html).toContain('href="/textbook/pretraining/distributed-systems/"');
  });
  it('shows how to choose among equal-loss models and where the long-run fit fails', async () => {
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/pretraining/scaling-laws/index.html'),'utf8');
      const card=await readFile(join(distDir,locale+'sources/papers/beyond-chinchilla-optimal/index.html'),'utf8');
      const s=html.match(/<a id="scaling-iso-loss-lifetime"><\/a>[\s\S]*?(?=<h2\b)/)?.[0] ?? '';
      for(const value of ['159,06','12,80','14,04','246 млрд','sardana24a.pdf#page=6'])
        expect(s).toContain(value);
      const figure=s.match(/<img[^>]+src="([^"]*\/sardana24a-figure5\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      const bytes=await readFile(join(distDir,decodeURIComponent(figure!).replace(/^\//,'')));
      expect(createHash('sha256').update(bytes).digest('hex')).toBe('2eeafe50cf15461f81656df20cfbe90302b4ec97bdb09f7a1b7834892038094c');
      expect(s).toContain('href="/'+locale+'sources/papers/beyond-chinchilla-optimal/"');
      expect(card).toContain('href="/textbook/pretraining/scaling-laws/#scaling-iso-loss-lifetime"');
      expect(s).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('keeps both old speculative decoding bookmarks beside the renamed algorithm', async () => {
    const route = 'textbook/inference/speculative-decoding/index.html';
    for (const [prefix, current] of [
      ['', 'алгоритм-2-черновик-проверка-и-замена'],
      ['en/', 'algorithm-2-drafting-verification-and-correction'],
    ]) {
      const html = await readFile(join(distDir, prefix, route), 'utf8');
      const legacy = ['полный-алгоритм-на-исходном-слайде'];
      if (prefix) legacy.push('the-full-algorithm-in-the-original-slide');
      for (const id of legacy) {
        const marker = 'id="' + id + '"';
        expect(html.split(marker)).toHaveLength(2);
        const offset = html.indexOf(marker);
        expect(html.slice(offset, offset + 700)).toContain('id="' + current + '"');
      }
      expect(html.split('id="' + current + '"')).toHaveLength(2);
    }
  });
  it('renders EAGLE-3 training targets and the unchanged original chart in both editions', async () => {
    const route = 'textbook/inference/speculative-decoding/';
    const note = 'sources/papers/eagle-3/';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      expect(html).toContain('id="eagle3-training-targets"');
      expect(html).toContain('id="eagle3-benchmark-reading"');
      expect(html).toContain('href="/licenses/eagle-cb7e0841-Apache-2.0.txt"');
      expect(html).not.toContain('katex-error');
      const src = [...html.matchAll(/<img[^>]* src="([^"]+)"/g)]
        .map(match => match[1]).find(url => url.endsWith('/eagle3r.jpg'));
      expect(src).toBeDefined();
      const bytes = await readFile(join(distDir, decodeURIComponent(src!)));
      expect(createHash('sha256').update(bytes).digest('hex')).toBe('5e404ac75809d8125e1c360c36054c2f90ebd72453a46a352609e538d91f6ba1');
      for (const [anchor, file, figure, digest] of [
        ['eagle3-inference-figure', 'eagle3-inference-figure5.png', 'S3.F5', '1fc5aea1023b11258bcc47d75f528b4e0224364089c2bbdba20a0c0aeb0f86c1'],
        ['eagle3-training-mask', 'eagle3-training-mask-figure6.svg', 'S3.F6', '91c835f7a9949fdd466d453204670652cab8174c088326f03d46ad040c3ee656'],
      ]) {
        const start = html.indexOf('id="' + anchor + '"');
        const figureHtml = html.slice(start, html.indexOf('</figcaption>', start));
        expect(start).toBeGreaterThan(0);
        expect(figureHtml).toContain('https://arxiv.org/html/2503.01840v3#' + figure);
        expect(figureHtml).toContain('https://openreview.net/forum?id=4exx1hUffq');
        expect(figureHtml).toContain('https://creativecommons.org/licenses/by/4.0/');
        const url = [...figureHtml.matchAll(/<img[^>]* src="([^"]+)"[^>]* alt="[^"]+"/g)]
          .map(m => m[1]).find(value => value.endsWith('/' + file));
        expect(url).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(url!)))).digest('hex'))
          .toBe(digest);
      }
      expect(html).toContain('href="/' + locale + note + '"');
      const source = await readFile(join(distDir, locale, note, 'index.html'), 'utf8');
      for (const anchor of ['eagle3-inference-figure', 'eagle3-training-mask', 'eagle3-training-targets', 'eagle3-benchmark-reading'])
        expect(source).toContain('href="/' + locale + route + '#' + anchor + '"');
    }
  });
  it('renders the original IsoFLOP figures, worked example and source-note round trip', async () => {
    const route = 'textbook/pretraining/scaling-laws/';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      for (const [name, digest, paper] of [["happy_path.svg","0ac01e9299c3c92fc3705dc0a2482e1979e9d721dcff70935adf2efdb45df8ad","https://arxiv.org/html/2603.22339v1#S4.F3"],["asymmetric.svg","6df85f0b6ddf5d8be2486e2ea614d6aa37d60076e0f30072ab53b14c34fa8ebc","https://arxiv.org/html/2603.22339v1#S5.F4"],["training_flop_vs_model_size.png","3fbaefa50fa091f9dc58136ae9772a1d27123bf9399751bc8be0c919a2be7841","https://arxiv.org/html/2404.10102v2#S2.F2"],["distributions.svg","98769023cbf1cd7193d454cab36051328d71182d12bb0a9895c2e4d8df6bd1e5","https://arxiv.org/html/2404.10102v2#S3.F3"],["tokens_to_params_ratio_plot.svg","c550efc1565c868ac3ebb095a2f46a14f8fdd0ceb16246dcbab7d67028461cd6","https://arxiv.org/html/2404.10102v2#S3.F5"]]) {
        const src = [...html.matchAll(/<img[^>]* src="([^"]+)"/g)]
          .map(match => match[1]).find(url => url.endsWith('/' + name));
        expect(src).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex')).toBe(digest);
        expect(html).toContain(paper);
      }
      for (const old of ['lower-envelope', 'isoflops', 'joint-fit', 'residuals'])
        expect(html).not.toContain('chinchilla-' + old + '-chart.png');
      for (const anchor of ['S3.F2', 'S3.F3', 'S3.F4'])
        expect(html).toContain('https://arxiv.org/html/2203.15556v1#' + anchor);
      expect(html).toContain('id="isoflop-parabola-bias"');
      expect(html).toContain('1,3285');
      expect(html).not.toContain('katex-error');
      const note = 'sources/papers/isoflop-parabola-bias/';
      expect(html).toContain('href="/' + locale + note + '"');
      const source = await readFile(join(distDir, locale, note, 'index.html'), 'utf8');
      expect(source).toContain('href="/' + locale + route + '#isoflop-parabola-bias"');
      expect(source).toContain('Textbook (in Russian): a three-point fit with a known minimum');
    }
  });
  it('renders original scaling metric figures and the links to their explanations in both locale routes', async () => {
    const route = 'textbook/pretraining/scaling-laws/';
    const anchor = 'scaling-metric-emergence';
    const figures = [
      ['wei-emergence-figure2.png', 'd905d44f12c6861f6e1febd0af3e7ab446f1ccf1531a1702c43dd4b977dd453b'],
      ['schaeffer-multiplication-exact-match.svg', '19c9799140b911f3d7d6eb8a8f0194f154b12800ca89012c6336aac166d77054'],
      ['schaeffer-multiplication-token-edits.svg', '19ed0bcfe5a86739ae0c656e516fdc28c242156c5d23b60ca2aed548c40f9aab'],
    ];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const section = html.split('<a id="' + anchor + '"></a>')[1]?.split('<a id="train-inference-tradeoff">')[0];
      expect(section).toBeDefined();
      expect(section).not.toContain('katex-error');
      for (const phrase of ['0,080%', '12,158%', '92,3%', 'BLEU, ROUGE или BLEURT'])
        expect(section).toContain(phrase);
      expect(section).toContain('https://arxiv.org/html/2206.07682v2#S3.F2');
      expect(section).toContain('https://arxiv.org/html/2304.15004v2#S3.F3');
      for (const [name, digest] of figures) {
        const src = [...section!.matchAll(/<img[^>]* src="([^"]+)"/g)]
          .map(match => match[1]).find(url => url.endsWith('/' + name));
        expect(src).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex')).toBe(digest);
      }
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    const manifest = loadManifest(join(rootDir, 'publishing/navigation.yml'));
    for (const source of ["05 Источники/Papers/Emergent Abilities of Large Language Models.md","05 Источники/Papers/Are Emergent Abilities of Large Language Models a Mirage.md","05 Источники/Courses/Stanford CS336 Spring 2026/_index.md"]) {
      const page = manifest.sections.flatMap(section => section.pages).find(item => item.source === source);
      expect(page).toBeDefined();
      const html = await readFile(join(distDir, page!.route.replace(/\/index$/, ''), 'index.html'), 'utf8');
      expect(html).toContain('href="/' + route + '#' + anchor + '"');
    }
  });
  it('renders the distributed overview with visible TP gradients, overlap and published group configurations', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/distributed-systems/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['split_loss.backward()', '[[60.0, 12.0]]',
      '[9.0, 33.0] 42.0 2.0', 'ddp.no_sync()', '4–7', '12–15 ms',
      'fsdp_tp_calcs', '49 устройств', '16 устройств',
      'TP=8, CP=16, PP=16, DP=8', '2048 H800']) expect(text).toContain(value);
    for (const name of ['hf-ddp-sequential.svg', 'hf-ddp-overlap.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^\"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(3);
    expect(html).toContain('device-placement-dimensions.svg');
    for (const fact of ['placement_groups', '[0, 8, 16, 24]', '[7, 15, 23, 31]',
      '32 devices, not eight', 'wrong batch rule']) expect(text).toContain(fact);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/scaling-laws/"');
  });
  it('renders the complete placement SVG, source caption and group example in both locale routes',async()=>{
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/pretraining/distributed-systems/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="three-coordinates-for-one-gpu"');
      expect(html).toContain('device-placement-dimensions.svg');
      for(const fact of ['Device Placement Dimensions','placement_groups','32 devices, not eight',
        '30 sequences','480','not a network topology or pipeline schedule']) expect(text).toContain(fact);
      expect(html).not.toContain('katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders finite-context PPL examples, original HF animations and sequential MTP explanation', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/next-token-objective/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['scoring_windows', 'num_valid_tokens - batch_size', '3, 1 и 0',
      '3, 2 и 1', '3{,}1748', '2{,}3784', 'RMSNorm', 'матрица', '16/9'])
      expect(text).toContain(value);
    for (const name of ['d2l-language-model-data.png', 'd2l-gpt-decoder.png', 'mtp-overview.png',
      'ppl-full.gif', 'ppl-chunked.gif', 'ppl-sliding.gif'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/transformers-e7b03417-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/scaling-laws/"');
    expect(html).toContain('href="/textbook/pretraining/dedup-pii-corpus-quality/"');
  });
  it('keeps the packed-document goal separate from the original attention mask figure', async () => {
    for (const base of ['', 'en/']) {
      const html=await readFile(join(distDir,base+'textbook/pretraining/next-token-objective/index.html'),'utf8');
      const card=await readFile(join(distDir,base+'sources/papers/barbero-attention-sinks/index.html'),'utf8');
      const image=html.match(/<img[^>]+src="([^"]*\/barbero-figure8-data-packing\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(image).toBeDefined();
      const bytes=await readFile(join(distDir,decodeURIComponent(image!).replace(/^\//,'')));
      expect(createHash('sha256').update(bytes).digest('hex')).toBe('f513d40145efbd10156eae38094ba7100fb140eca064478d0250b18910430978');
      for(const value of ['id="ntp-packed-document-boundary"','A1.F8','eod_mask_loss','packed_targets','S3.SS2'])
        expect(html).toContain(value);
      expect(html).toContain('href="/'+base+'textbook/ml-systems/data-pipeline-padding-packing/#data-document-isolation"');
      expect(html).toContain('href="/'+base+'sources/papers/barbero-attention-sinks/"');
      expect(card).toContain('href="/textbook/pretraining/next-token-objective/#ntp-packed-document-boundary"');
      if(base)expectCanonicalEditionNotice(html);
    }
  });
  it('renders the original sequential DeepSeek figure next to its readout and notice in both shells', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + 'textbook/pretraining/next-token-objective/index.html'), 'utf8');
      const section = html.match(/<a id="deepseek-sequential-mtp"><\/a>[\s\S]*?(?=<h2\b)/)?.[0];
      expect(section).toBeDefined();
      expect(section).toMatch(/<img[^>]+src="[^"]*\/deepseek-v3-mtp-figure3\.png"[^>]+alt="[^"]*t₂/);
      expect(section).toContain('DeepSeek_V3.pdf#page=10');
      expect(section).toContain('href="/licenses/deepseek-v3-d5c08b3-MIT.txt"');
      expect(section).toContain('1,8 раза');
      expect(section).toContain('53,7');
      expect(section).toContain('66,6');
      expect(section).not.toContain('katex-error');
      if (prefix) expectCanonicalEditionNotice(html);
    }
  });
  it('renders text and image deduplication with seven original figures and five Python examples', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/dedup-pii-corpus-quality/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['banana', 'longest common prefix', 'Results are based on minhash segment collision',
      '450 полос по 20 значений', 'EditDistance', '0,9946', '11,2%', 'print(verified)']) expect(text).toContain(value);
    for (const name of ['presidio-analyze-anonymize.png', 'lsh-characteristics.png',
      'lee-dedup-cluster-sizes-figure1.png', 'lee-dedup-perplexity-figure2.png', 'semdedup-intro.png',
      'dolma-figure31-c4.svg', 'c4-figure5-blocklist-pmi.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(6); // Five Python examples and the JSON audit record.
    for (const value of ['difference_bits', 'perceptual_bits', 'remove_any_earlier',
      'Colour collision', 'caption diversity']) expect(text).toContain(value);
    expect(html).toContain('href="/licenses/imagehash-BSD-2-Clause.txt"');
    expect(html).toContain('href="/licenses/semdedup-CC-BY-NC-4.0.txt"');
    for (const id of ['pii', 'harm-policy', 'exact-dedup', 'minhash-lsh', 'image-deduplication', 'contamination', 'audit-log'])
      expect(html).toContain('id="' + id + '"');
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/data-pipeline/"');
    expect(html).toContain('href="/textbook/pretraining/next-token-objective/"');
  });
  it('publishes the source-based Dolma and C4 audits without a second textbook chapter', async () => {
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/dedup-pii-corpus-quality/index.html'),'utf8');
      for(const [source,anchor] of [
        ['dolma-corpus-curation','dolma-pii-policy'],['c4-corpus-audit','c4-blocklist-audit']
      ]){
        const card=await readFile(join(distDir,locale+'sources/papers/'+source+'/index.html'),'utf8');
        expect(html).toContain('href="/'+locale+'sources/papers/'+source+'/"');
        expect(html).toContain('id="'+anchor+'"');
        expect(card).toContain('href="/textbook/pretraining/dedup-pii-corpus-quality/#'+anchor+'"');
      }
      expect(html).toContain('href="/'+locale+'textbook/pretraining/data-pipeline/#fineweb-deduplication-scope"');
      expect(html).toContain('53,6%');
      expect(html).toContain('1,8%');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders Harvard lab strings with intact TeX and separates runtime templates from original explanations', async () => {
    const records = JSON.parse(await readFile(join(rootDir,
      '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'), 'utf8'));
    const labs = records.filter((row: any) => row.kind === 'lab');
    expect(labs).toHaveLength(34);
    for (const row of labs) {
      const html = await readFile(join(distDir, row.route, 'index.html'), 'utf8');
      expect(html, row.route).not.toContain('class="katex-error');
      expect(html).toContain('Static text, not a running lab');
      expect(html).toContain('Complete original Marimo source');
      expect(html).toContain('/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md');
      const math = [...html.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)]
        .map(match => match[1]);
      expect(math.every((value: string) => !value.includes('`…`')), row.route).toBe(true);
    }
    const edge = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/labs/vol2/lab-11-edge-intelligence/index.html'), 'utf8');
    expect(edge).toContain('M_{\\text{train}} = M_{\\text{weights}}');
    expect(edge).toContain('UI text and runtime templates (not executed)');
    expect(edge).toContain('<details>');
    const chapter = await readFile(join(distDir, 'textbook/deployment/edge-federated/index.html'), 'utf8');
    expect(chapter).toContain('id="harvard-mixed-state-memory"');
    expect(chapter).toContain('120 MB');
    expect(chapter).toContain('152 MB');
    expect(chapter).toContain('192 MB');
    expect(chapter).toContain('issue in the source');
    expect(chapter).not.toContain('class="katex-error');
  });
  it('publishes finite-product examples with original JPEGs, English source prose and convergence conditions', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/formal-math/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="finite-products-and-missing-assumptions"');
    for (const name of ['prime-factorization.jpg', 'prime-geometric-factors.jpg'])
      expect(html.match(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"', 'g'))).toHaveLength(1);
    for (const value of ['remainders == [17, 17, 100]', 'F(91, 36)',
      'An intermediate step is', 'norm-summable', 'finite partial sums and bounds',
      'non-integer', 'symmetric limit', 'do not establish those theorems']) expect(text).toContain(value);
    expect(html).toContain('href="/sources/papers/mathlib-euler-products/"');
    expect(html).not.toContain('katex-error');
    const card = await readFile(join(distDir, 'sources/papers/mathlib-euler-products/index.html'), 'utf8');
    expect(card).toContain('not an LLM benchmark paper');
    for (const notice of ['formal-zeta-NOTICE.txt', 'mathlib-c44e0c8-Apache-2.0.txt'])
      expect(await readFile(join(distDir, 'licenses', notice), 'utf8')).toBe(
        await readFile(join(rootDir, 'site/public/licenses', notice), 'utf8'));
  });
  it('publishes the OLMo recipe, complete schedule slide and corrected mixture attribution', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/data-and-pretraining/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="olmo-two-stage-recipe"');
    expect(html.match(/<img[^>]+src="[^"]*m04-p147-two-stage-training\.png"/g)).toHaveLength(1);
    expect(html).toContain('alt="OLMo: warmup, длинная косинусная стадия и короткое линейное снижение до нуля"');
    for (const value of ['truncated normal distribution', 'three separate times for 50B tokens',
      '8,388,608,000', '16.78B', '150B-token sequential continuation',
      '600B tokens', '60.5 to 43.0', 'MMLU-Pro', '43.4 percentage points'])
      expect(text).toContain(value);
    expect(html).toContain('href="https://arxiv.org/html/2501.00656v1#bib.bib123"');
    expect(html).toContain('href="/sources/papers/olmo2-dolmino/"');
    expect(html).not.toContain('katex-error');
    const collection = await readFile(join(distDir, 'textbook/pretraining/data-pipeline/index.html'), 'utf8');
    expect(collection).toContain('William Held');
    expect(collection).not.toContain('Shen и соавторы');
    const card = await readFile(join(distDir, 'sources/papers/olmo2-dolmino/index.html'), 'utf8');
    expect(card.match(/<table[\s>]/g)).toHaveLength(3);
    for (const value of ['21.32T', '2.95B', '832.6B', '739.8B', 'Dolmino SynthMath',
      '35.1M', '7.27K', '21.37M', 'not GPU memory requirements']) expect(card).toContain(value);
    const notice = 'olmo-pretraining-NOTICE.txt';
    expect(await readFile(join(distDir, 'licenses', notice), 'utf8')).toBe(
      await readFile(join(rootDir, 'site/public/licenses', notice), 'utf8'));
  });
  it('renders microanneals, probabilistic filtering and the complete OpenWebMath pipeline', async () => {
    const intro = await readFile(join(distDir, 'textbook/pretraining/data-and-pretraining/index.html'), 'utf8');
    const html = await readFile(join(distDir, 'textbook/pretraining/data-pipeline/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['gpt3_keep_probability', 'Pareto', 'MathScore', '4968',
      '30 000', '15 000', 'contains_math', 'Prefiltering HTML Documents']) expect(text).toContain(value);
    expect(html).toMatch(/<img[^>]+src="[^"]*corpus-collection-2026-10-23\/openwebmath-pipeline\.png"/);
    // The HTML example also contains a literal <pre> inside the copy-button attribute.
    expect(html.match(/<pre data-language=/g)).toHaveLength(5);
    const introText = intro.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['microanneals', '130 млрд', '19 microanneals',
      '200 вопросов', 'Короткие продолжения одного чекпойнта']) expect(introText).toContain(value);
    expect(intro).not.toContain('katex-error');
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/dedup-pii-corpus-quality/"');
    expect(intro).toContain('href="/textbook/ml-systems/gpu-kernels-triton/"');
  });
  it('renders the corpus mix, OLMo schedule and benchmark overlap comparison in both interfaces', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale+'textbook/pretraining/data-and-pretraining/index.html'), 'utf8');
      for (const anchor of ['dolmino-source-vs-mixture-share', 'olmo-early-loss-vs-anneal', 'gpt3-overlap-vs-memorization'])
        expect(html).toContain('id="'+anchor+'"');
      for (const file of ['olmo2-learning-rates-figure11.svg','olmo2-learning-rate-anneals-figure11.svg',
        'lee-dedup-perplexity-figure2.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+file.replaceAll('.','\\.')+'"'));
      expect(html).toContain('href="/'+locale+'textbook/pretraining/dedup-pii-corpus-quality/#contamination"');
      expect(html).toContain('href="/'+locale+'textbook/modern-llm/transformer-shapes-resources/#logits-to-next-token-loss"');
      expect(html).toContain('4095 её целей');
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the corpus-release distinctions and complete ROOTS figure in both locales', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/pretraining/data-pipeline/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const value of ['20.8 billion documents','30.4 trillion tokens','TB of text',
        'sum to 47%','remove 23.1%','not training-token weights']) expect(text).toContain(value);
      for(const anchor of ['redpajama-releases','roots-multilingual-corpus'])
        expect(html).toContain('id="'+anchor+'"');
      expect(html.match(/<img[^>]+src="[^"]*corpus-releases-2026-10-29\/roots-oscar-filtered\.png"/g))
        .toHaveLength(1);
      expect(html).toContain('aa033f6b6480c445557e9118283993403f069f6a/README.md');
      expect(html).toContain('https://arxiv.org/html/2303.03915v1#S3.F3');
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders the WARC range, both original FineWeb graphics and their linked source note', async () => {
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/pretraining/data-pipeline/index.html'),'utf8');
      const card=await readFile(join(distDir,locale+'sources/papers/fineweb/index.html'),'utf8');
      for(const a of ['common-crawl-record-range','fineweb-deduplication-scope'])
        expect(html).toContain('id="'+a+'"');
      for(const f of ['fineweb-global-dedup-old-snapshot.svg','fineweb-per-snapshot-dedup.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+f.replaceAll('.','\\.')+'"'));
      expect(html).toContain('160581106-160593800');
      expect(html).toContain('206 Partial Content');
      expect(html).toContain('href="/'+locale+'sources/papers/fineweb/"');
      expect(card).toContain('href="/textbook/pretraining/data-pipeline/#fineweb-deduplication-scope"');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders Triton derivations, executable chunked CE and nine original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['correct_block', 'wrong_block', '0.625', '435982',
      'chunked_linear_ce', 'full_dh', 'full_dw', '125 МиБ', '7.8125 ГиБ',
      '1.357707', '1.723299', 'weight.T']) expect(text).toContain(value);
    for (const name of ['l05-p31.png', 'l05-p32.png', 'l05-p33.png',
      'triton-softmax.png', 'triton-row-sum.png', 'l05-p42.png', 'l05-p43.png',
      'triton-grouped-vs-row-major.png', 'liger-flce-original.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const id of ['cs336-systems-triton', 'контракт-собственного-kernel',
      'что-именно-задаёт-triton-program', 'первый-полный-пример-softmax-одной-строки',
      'reduction-которая-не-помещается-в-один-tile',
      'tiled-gemm-reuse-вместо-идеализированной-загрузки-всей-матрицы',
      'autotuning--поиск-по-корректным-кандидатам', 'как-читать-профиль-triton-kernel'])
      expect(html).toContain('id="' + id + '"');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(10);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/ml-systems/profiling/"');
    expect(html).toContain('href="/textbook/pretraining/data-and-pretraining/"');
  });
  it('renders profiler interpretation, allocator examples and four original screenshots', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/profiling/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['step-5.json', '22 мс', '4001-м', '156 МиБ',
      'record_pinned_host_memory=True', 'host_segments', 'forward_and_loss']) expect(text).toContain(value);
    for (const name of ['pytorch-profiler-trace.png', 'a2-nsight-trace.png',
      'pytorch-active-memory-timeline.png', 'pytorch-allocator-state-history.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const id of ['cs336-systems-profiling', 'полный-диагностический-маршрут',
      'диагноз-по-уровню-стека', 'pytorch-profiler', 'cpu-py-spy', 'nsight-systems',
      'от-python-функции-до-kernel-где-находится-framework', 'capture-не-равен-optimization',
      'interoperability-graph-как-контракт-с-границами', 'memory-snapshot'])
      expect(html).toContain('id="' + id + '"');
    expect(html.match(/<pre/g)).toHaveLength(13);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/ml-systems/data-pipeline-padding-packing/"');
    expect(html).toContain('href="/textbook/ml-systems/gpu-kernels-triton/"');
  });
  it('publishes the full Harvard step and the separate CPU gradient/event walkthrough in both routes',async()=>{
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/ml-systems/profiling/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const fact of ['A complete training step','not a standalone runnable program',
        'observed_step','fixed_input_products','trainable_input_products',
        'expected_gradients','13,008,896','13,172,736','not five CUDA kernel launches'])
        expect(text).toContain(fact);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
      // The viewer also contains an empty image element without src.
      expect(Array.from(html.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*>/g))).toHaveLength(6);
      expect(html).toContain('cs231n-two-layer.jpeg');
      expect(html).toContain('href="https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/');
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('publishes24 profiling subjects and the complete source flame graph in both interfaces',async()=>{
    const anchors=["profile-dependencies","profile-tools","profile-experiment","pytorch-profiler","profile-aggregates","cpu-py-spy","profile-flamegraph","nsight-systems","profile-nvtx","profile-patterns","profile-gelu","profile-runtime","profile-training-step","profile-step-gradients","profile-step-products","profile-capture","profile-fusion","profile-export","memory-snapshot","profile-memory-history","profile-pinned-history","profile-nsight-compute","profile-end-to-end","profile-reproducibility"];
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/profiling/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      for(const fact of ['Reading a sampled flame graph','2893','1490','1371',
        '814','813','51.50%','47.39%','not chronology','same 2893 samples'])
        expect(text).toContain(fact);
      expect(html).toContain('py-spy-flamegraph.svg');
      expect(html).toContain('/licenses/py-spy-7a7ac96e-MIT.txt');
      expect(html).toContain('7a7ac96e96e0f9b4ba3eee80de3f4d255698170b');
      expect(html.match(/<pre/g)).toHaveLength(13);
      expect(html).not.toContain('katex-error');
      if(prefix)expectCanonicalEditionNotice(html);
    }
  });
  it('renders data-pipeline reasoning, executable packing, caching and seven source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/data-pipeline-padding-packing/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['104 мс', '144', '26,5%', '555 806', 'in_order=False',
      'causal_only', 'changed_v', '10, 15, 20, 25', 'Нормировка при усреднении градиентов DDP',
      'содержимое shuffle-буфера теряется', 'Декодирование изображений и DALI',
      '37.5 МиБ', '4.59 МиБ', '18.38 МиБ', 'fn.readers.caffe']) expect(text).toContain(value);
    for (const name of ['pinmem.png', 'same-stream.png', 'separate-stream.png',
      'optimal-sequence-processing.png', 'padding-packing.png', 'cached-dataset.svg', 'two_readers.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre/g)).toHaveLength(10);
    expect(text).toContain('Caching across epochs');
    expect(text).toContain('Persistent cache identity and memory');
    expect(text).toContain('143.55 MiB');
    expect(text).toContain('574.22 MiB');
    expect(html).toContain('id="cache-placement-and-epoch-state"');
    expect(html).toContain('/licenses/tensorflow-docs-e4210b94-Apache-2.0.txt');
    expect(html).not.toContain('katex-error');
    for (const id of ['storage-и-dataloader-как-система', 'packing-без-leakage',
      'базовый-путь-загрузки-без-доказанного-перекрытия', 'проверка-throughput'])
      expect(html).toContain('id="' + id + '"');
    expect(html).toContain('href="/textbook/ml-systems/numerical-formats-mixed-precision/"');
    expect(html).toContain('href="/textbook/ml-systems/profiling/"');
  });
  it('retains the cache examples and full vector in root and explicit English fallback', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/ml-systems/data-pipeline-padding-packing/index.html'),'utf8');
      expect(html.match(/<pre\b/g)).toHaveLength(10);
      expect(Array.from(html.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*>/g))).toHaveLength(7);
      for(const value of ['cache-placement-and-epoch-state','decode_calls','epoch_dependent',
        'reshuffle_each_iteration','records-v1','cached-dataset.svg']) expect(html).toContain(value);
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('publishes the full original DALI graph and21 unique subject anchors in both interfaces',async()=>{
    const anchors=["data-storage","data-decoding","data-dali-graph","data-loader","cache-placement-and-epoch-state","data-cache-counts","data-cache-randomness","data-cache-completion","data-cache-identity","data-h2d","data-padding","data-pad-eos","data-bucketing","data-packing","data-document-isolation","data-sdpa-isolation","data-loss-normalization","data-ddp-weighting","data-loader-loop","data-sharding-resume","data-throughput"];
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/data-pipeline-padding-packing/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      for(const fact of ['Operators, data nodes, and pipeline outputs','fn.readers.file',
        'syntax-checked, not executed on a GPU','CPU coin flip','labels.gpu()'])
        expect(text).toContain(fact);
      expect(html).toContain('two_readers.svg');
      expect(html).toContain('/licenses/nvidia-dali-5b33fee1-Apache-2.0.txt');
      expect(html).toContain('5b33fee1968ba069a9ed04965fd2af5f0025de7e');
      if(prefix)expectCanonicalEditionNotice(html);
    }
  });
  it('renders precision experiments, scaling examples and six source diagrams', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/numerical-formats-mixed-precision/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['repeated_sum', '10.000133514404297', '9.953125',
      '10.00213623046875', 'Current и delayed scaling', '31 единицу', 'reduction="sum"'])
      expect(text).toContain(value);
    for (const name of ['te-float-bits.png', 'hf-format-ranges.png', 'hf-format-resolution.png',
      'te-master-weights.svg', 'te-current-delayed-scaling.svg', 'te-mxfp8-blocks.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-precision"');
    expect(html).toContain('href="/textbook/ml-systems/transformer-moe-arithmetic/"');
    expect(html).toContain('href="/textbook/ml-systems/data-pipeline-padding-packing/"');
  });
  it('renders21 precision subjects, complete MXFP8 source geometry and worked programs in both interfaces',async()=>{
    const anchors=["precision-bits","precision-range","precision-subnormals","precision-sums","precision-input-error","precision-tf32","precision-autocast","precision-master","precision-loss-scaling","precision-amp-step","precision-accumulation","precision-accumulation-order","precision-clipping-order","precision-memory","precision-bandwidth","precision-fp8","precision-scaling-history","precision-mxfp8","precision-mxfp8-orientation","precision-validation","precision-mfu"];
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/numerical-formats-mixed-precision/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\\s+/g,' ');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      for(const fact of ['960.5','1024','hardware-packed','rowwise_roundtrip','0.075,0.1'])
        expect(text).toContain(fact);
      expect(html.match(/<pre/g)).toHaveLength(4);
      expect(html).toContain('te-mxfp8-row-column.svg');
      expect(html).toContain('pytorch-v2.14.1-LICENSE.txt');
      expect(html).not.toContain('katex-error');
      if(prefix)expectCanonicalEditionNotice(html);
    }
  });
  it('renders transformer arithmetic, the numeric backward and scaling limits', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/transformer-moe-arithmetic/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['finite_difference', '13.0 5.0', 'MFU и HFU', '66.7%',
      '60.0%', '80.0%', 'Сколько устройств', '2.048', '4.128', '1.215', 'общая сеть'])
      expect(text).toContain(value);
    for (const name of ['jax-transformer.png','hf-training-memory.svg','nemo-checkpointing-detail.png',
      'hf-fsdp-forward.svg','hf-fsdp-backward.svg','edls-tp.png','edls-ep.png',
      'hf-pipeline-gpipe.svg','hf-pipeline-1f1b.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-arithmetic"');
    expect(html).toContain('href="/textbook/ml-systems/performance-measurement-roofline/"');
    expect(html).toContain('href="/textbook/ml-systems/numerical-formats-mixed-precision/"');
  });
  it('renders31 arithmetic subjects and descriptive source figures in both interfaces',async()=>{
    const anchors=["arith-shapes","arith-attention","arith-swiglu","arith-llama","arith-linear-gradients","arith-sixp","arith-attention-length","arith-states","arith-activations","arith-checkpointing","arith-mfu-hfu","arith-kv","arith-collectives","arith-ddp","arith-zero","arith-fsdp","arith-fsdp-unit","arith-tp","arith-fsdp-tp","arith-strong-scaling","arith-shared-network","arith-moe","arith-tp-ep-work","arith-tp-ep-traffic","arith-qwen3","arith-deepseek","arith-pipeline","arith-microbatches","arith-zero-bubble","arith-dualpipe","arith-budget"];
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/transformer-moe-arithmetic/index.html'),'utf8');
      for(const anchor of anchors)expect(html.split('id="'+anchor+'"')).toHaveLength(2);
      expect(html.match(/<pre/g)).toHaveLength(1);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('alt="TP: на каждой GPU хранятся части всех экспертов"');
      expect(html).toContain('alt="EP: на каждой GPU хранятся целые эксперты, токены доставляются Dispatch"');
      expect(html).toContain('cs336-systems-arithmetic');
      if(prefix)expectCanonicalEditionNotice(html);
    }
  });
  it('shows the controlled OPT FLOP counts and original Qwen prefill figure in both reader routes', async () => {
    const route = 'textbook/ml-systems/transformer-moe-arithmetic/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const section = html.split('id="arith-attention-share"')[1]?.split(/<h2\b[^>]*id="3-почему-памяти/)[0];
      expect(section).toBeDefined();
      expect(section).not.toContain('katex-error');
      for (const term of ['OPT-350M, 2048', 'OPT-1.3B, 2048', 'OPT-350M, 1024',
        '14,29%', '7,70%', 'расчётный выигрыш', 'CC BY 4.0', 'https://arxiv.org/html/2504.17768v2#A2.F12'])
        expect(section).toContain(term);
      const src = section!.match(/<img[^>]* src="([^"]+\/prefilling_seq_length_impact\.svg)"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex')).toBe('e309c11191da88331f9302fdb3983f75c68e6c8afe119b1721f98c3d11cfeef6');
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      expect(reference).toContain('href="/' + prefix + route + '#arith-attention-share"');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    const note = await readFile(join(distDir, 'sources/papers/the-sparse-frontier/index.html'), 'utf8');
    const course = await readFile(join(distDir, 'sources/courses/stanford-cs336-spring-2026/index.html'), 'utf8');
    for (const origin of [note, course]) expect(origin).toContain('href="/' + route + '#arith-attention-share"');
  });
  it('renders the performance capstone with source trace, complete examples and evidence limits', async () => {
    const html = await readFile(join(distDir, 'practice/13-optimizirovat-odin-transformer-step/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['Profile, compare, size, and choose', 'Find the interval that delays the result',
      'Separate an observed difference', 'Plan sample size before seeing the result',
      'Apply constraints before ranking candidates', 'ttest_ind', 'equal_var=False',
      'planned_shift', 'required_units', 'hold: no feasible option', 'not measurements']) expect(text).toContain(value);
    expect(html).toMatch(/<img[^>]+src="[^"]*\/a2-nsight-trace\.png"/);
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).toContain('id="performance-evidence-profiling-regression-capacity"');
    expect(html).toContain('href="/licenses/scipy-1.18.0-LICENSE.txt"');
    expect(html).not.toContain('katex-error');
    expect(text).toContain('TinyTorch');
  });
  it('renders per-level and sequential roofline calculations with nine source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/performance-measurement-roofline/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const name of ['b2b9aaeaa637a9a665f0f4e9229daeb5bc5a4f2d.svg',
      '682e143e450db0d85ad5c608e3a125ac5176e0ba.svg', 'd2l-asyncgraph.svg',
      'd2l-threading.svg', 'jax-roofline.png', 'l05-p47.png', 'a2-nsight-trace.png',
      'megatron-weak-scaling.png', 'megatron-strong-scaling.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const value of ['4.5 TB/s', '90', 'Последовательность операций:',
      'lower_bound_ms', 'sequential_ms', 'pooled_ms', '3.0', '4.8']) expect(text).toContain(value);
    expect(html.match(/<pre/g)).toHaveLength(7);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-roofline"');
    expect(html).toContain('id="fleet-mfu-scaling-and-preserved-progress"');
    for (const value of ['MFU and HFU count different work', '35446', '26.15%',
      'Preserved progress, checkpointing and recovery', 'Preserved-work utilization:',
      'did not train these models to convergence']) expect(text).toContain(value);
    expect(html).toContain('href="/textbook/ml-systems/gpu-cuda-memory-hierarchy/"');
    expect(html).toContain('href="/textbook/ml-systems/transformer-moe-arithmetic/"');
  });
  it('renders blocked observations and all24 subject anchors in both locale surfaces',async()=>{
    const source=await readFile(join(rootDir,'00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md'),'utf8');
    const meta=YAML.parse(source.split('---')[1]);
    expect(meta.reference_terms).toHaveLength(24);
    for(const prefix of ['','en/']){
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/performance-measurement-roofline/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html.match(/<pre/g)).toHaveLength(7);
      for(const term of meta.reference_terms)
        expect(html.match(new RegExp('id="'+term.anchor+'"','g'))).toHaveLength(1);
      for(const fact of ['Blocked measurements and individual-call latency','2012 microseconds',
        'raw_times','number_per_run','group_measurement','451','20 of the 1000 calls',
        'does not run a timed operation','TaskSpec','adaptive_autorange'])
        expect(text).toContain(fact);
      expect(html).toContain('/licenses/pytorch-tutorials-5672cd98-BSD-3-Clause.txt');
      expect(html).not.toContain('katex-error');
      if(prefix)expectCanonicalEditionNotice(html);
    }
  });

  it('renders GPU memory examples, fifteen originals and adjacent navigation', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-cuda-memory-hierarchy/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    const figures = ['cuda-grid-cta.png', 'gpu-hardware.png', 'nvidia-coalescing-4.png',
      'nvidia-coalescing-5.png', 'banks-stride-1.svg', 'banks-stride-2.svg', 'banks-stride-4.svg',
      'nvidia-gemm-step-6.png', 'nvidia-gemm-step-7.png', 'nvidia-gemm-step-8.png',
      'nvidia-gemm-step-9.png', 'triton-grouped-vs-row-major.png', 'nvidia-fusion.svg',
      'nvidia-same-stream.png', 'nvidia-two-streams.png'];
    for (const name of figures)
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const value of ['Swizzling:', 'Неполная плитка:', 'tiled_matmul', '139.', '154.',
      'Tensor Memory', 'sm_100a', '16 KiB']) expect(text).toContain(value);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-gpu"');
    expect(html).toContain('href="/textbook/ml-systems/model-as-part-of-system/"');
    expect(html).toContain('href="/textbook/ml-systems/performance-measurement-roofline/"');
  });
  it('renders end-to-end latency, queueing example and six source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/model-as-part-of-system/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const name of ['ml-lifecycle.png', 'mit-amdahl.png', 'mit-pipeline.png',
      '2f10e92057738ef1e4a50e524fddf5389d61d795.svg',
      'c491ee41569a92f002bf915150731598e0d64e26.svg',
      '3b022a3ffa6f028fe2c2c3165c292a16a3a9e0ff.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const s of ['Почему средней скорости недостаточно', '69 мс', '103 мс',
      '35, 54 и 73 мс', '0.94', 'не результат измерения конкретной камеры или GPU'])
      expect(text).toContain(s);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/modern-llm/sequence-model-alternatives/"');
    expect(html).toContain('href="/textbook/ml-systems/gpu-cuda-memory-hierarchy/"');
  });
  it('renders sequence-model calculations, complete Mamba-2 figures and navigation',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/sequence-model-alternatives/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['mamba2-ssd-algorithm.png','mamba2-block-complete.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/sequence-alternatives-2026-10-23/'+name.replaceAll('.','\\.')+'"'));
    for(const s of ['diagonal_recurrence','selective_ssm','ssd_chunked','normalized_linear',
      'gated_delta','ttt_linear','9.20625','6.28125','17.5'])
      expect(text).toContain(s);
    expect(html).not.toContain('mamba2-block-figure.png');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/moe\/mixture-of-experts\/"/);
    expect(html).toMatch(/К модулям:[\s\S]{0,300}href="\/textbook\/#/);
  });
  it('uses source originals in the sequence-model reader and excludes the retired course frames', async () => {
    const route = 'textbook/modern-llm/sequence-model-alternatives';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      for (const name of ["unfolded-rnn.svg","d2l-lstm-2.svg","d2l-seq2seq-state.svg","seq2seq-attention.svg","d2l-gpt-decoder-original.svg","mamba-v2-figure1.svg","jamba-architecture.svg","katharopoulos-figure1.png"]) {
        const path = html.match(new RegExp('<img[^>]+src="([^"]*/' + name.replaceAll('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(path, name).toBeDefined();
        if (name === 'katharopoulos-figure1.png')
          expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(path!)))).digest('hex')).toBe('b42e88dcb5c22becc7b26fa197512d49c89886963570e89fc5c53f6e34f78d22');
      }
      for (const name of ["berkeley-bahdanau-attention.png","berkeley-qwen3-next-gdn.png","berkeley-seq2seq-bottleneck.png","berkeley-transformer-language-model.png","cmu-hybrid-models.png","cmu-linear-attention.png","mamba-selective-scan-course.png","stanford-cs230-lstm.png","stanford-cs230-rnn-unrolled.png","qwen-next-hybrid-architecture.jpg"]) expect(html).not.toContain(name);
      for (const url of [
        'https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf#page=6',
        'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct#model-overview',
        'https://www.cs.cmu.edu/~mgormley/courses/10423/slides/lecture21-ssm.pdf#page=34',
      ]) expect(html).toContain('href="' + url + '"');
      expect(html).toContain('768 МиБ');
      expect(html).not.toContain('katex-error');
    }
    for (const file of ["00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/berkeley-bahdanau-attention.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/berkeley-qwen3-next-gdn.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/berkeley-seq2seq-bottleneck.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/berkeley-transformer-language-model.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/cmu-hybrid-models.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/cmu-linear-attention.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/mamba-selective-scan-course.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/stanford-cs230-lstm.png","00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/stanford-cs230-rnn-unrolled.png","00 Учебник/Assets/Figures/curated/stanford-cs336-2026/architectures/qwen-next-hybrid-architecture.jpg"])
      await expect(readFile(join(distDir, 'assets', file))).rejects.toMatchObject({ code: 'ENOENT' });
  });
  it('renders licensed recall plots and leaves RetNet and Waleffe originals as links', async () => {
    const review = JSON.parse(await readFile(join(rootDir,
      'docs/audit/2026-11-11-sequence-recall-licensed-figures-additions.json'), 'utf8')) as {
      figures: { asset: string; sha256: string; figure_url: string }[];
      retired: { asset: string; figure_url: string }[];
    };
    expect(review.figures).toHaveLength(2);
    expect(review.retired).toHaveLength(4);
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale,
        'textbook/modern-llm/sequence-model-alternatives/index.html'), 'utf8');
      for (const figure of review.figures) {
        const img = [...html.matchAll(/<img[^>]+src="([^"]+)"[^>]+alt="[^"]+"/g)]
          .map(match => decodeURIComponent(match[1])).find(src => src.endsWith('/' + figure.asset));
        expect(img, figure.asset).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, img!))).digest('hex')).toBe(figure.sha256);
        expect(html).toContain(figure.figure_url);
      }
      for (const old of review.retired) {
        expect(html).not.toContain(old.asset.split('/').at(-1));
        expect(html).toContain(old.figure_url);
      }
      expect(html).toContain('id="seq-state-update-diagram"');
      expect(html).toContain('id="seq-hybrid-recall"');
      for (const value of ['10.5', '5500', '87,5%', '55,9%', '29,3%', '33,2%'])
        expect(html).toContain(value);
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
      const atlas = await readFile(join(distDir, locale, 'models/semeystva/retnet/index.html'), 'utf8');
      const sources = await readFile(join(distDir, locale, 'sources/papers/recall-efficient-sequence-models/index.html'), 'utf8');
      expect(atlas).toContain(review.retired[0].figure_url);
      expect(atlas).not.toContain('retnet-dual-form.png');
      for (const old of review.retired.slice(1, 3)) {
        expect(sources).toContain(old.figure_url);
        expect(sources).not.toContain(old.asset.split('/').at(-1));
      }
    }
    for (const old of review.retired)
      for (const file of [old.asset, old.asset.replace(/^00 Учебник\/Assets\//, '')])
        await expect(readFile(join(distDir, 'assets', file))).rejects.toMatchObject({ code: 'ENOENT' });
  });
  it('renders complete MoE source graphs, worked examples and neighboring navigation',async()=>{
    const html=await readFile(join(distDir,'textbook/moe/mixture-of-experts/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['switch-scaling','switch-capacity','switch-parallelism','olmoe-shared',
      'olmoe-granularity','olmoe-lbl','olmoe-lbltoks','olmoe-zloss'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/moe-2026-10-23/'+name+'\\.png"'));
    for(const s of ['dispatch_combine','capacity_mask','router_losses','node_limited_route',
      '0.99024','route_scale','32 байта'])expect(text).toContain(s);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/long-context\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/sequence-model-alternatives\/"/);
  });
  it('renders the complete FFN examples, original architecture and vocabulary figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/dense-ffn/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['transformer-block-ffn.png','gpt2-mlp-expand.gif','gpt2-mlp-project.gif',
      'cs336-silu-vs-relu.png','geva-ffn-memory.svg','tensor-parallel-split.svg',
      'mixtral-smoe-layer.png','geva-composition.svg','tajima-figure1-ffn.svg',
      'geva-ffn-vocabulary.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['swiglu_backward','fused_projection','split_y','1.633','4000 префиксах',
      '135 266 304','172 MiB'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="ffn-vocabulary-projection"');
    expect(html).toContain('https://arxiv.org/html/2203.14680v1#S1.F1');
    expect(html).toContain('https://arxiv.org/html/2506.23225v1#S1.F1');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/rope\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/transformer-shapes-resources\/"/);
  });
  it('renders long-context mechanisms, CSA/HCA originals and adjacent reading links',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/long-context/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['position-interpolation.png','yarn-method-summary.png','mistral-sliding-window.png',
      'mistral-rolling-step-0.png','mistral-rolling-step-1.png','mistral-rolling-step-2.png',
      'cla-diagram.png','deepseek-dsa-architecture.png','lost-in-the-middle.png',
      'mistral-chunk-mask.png','cla-cache-groups.png','ruler-aggregation.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['window_attention','h2_shared','index_scores','13.25','1.375 GiB',
      'пересчитать затронутый префикс','SQuAD и HotpotQA','500 примеров'])expect(text).toContain(fact);
    for (const figure of ['deepseek-v4-csa.png', 'deepseek-v4-hca.png']) expect(html).toContain(figure);
    expect(html).toContain('compressed-sparse-attention-csa-hca');
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mla-kv-compression\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/moe\/mixture-of-experts\/"/);
  });
  it('renders normalized MLA, positional obstruction and original source panel',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/mla-kv-compression/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['v2-figure3-attention.png','mha-mla-projections.svg',
      'deepseek-mla-panel.png','v2-table8-dense-ablation.png','v2-table9-mla-ablation.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const s of ['TinyMLA','same_latents','q_gain','kv_gain','11110','11111',
      '80 чисел','20 чисел','13 254 000 640','8 H800','420 млрд','860,2 до 34,6',
      '4,0 процентного пункта','50,9%, у MHA 51,6%'])expect(text).toContain(s);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).not.toMatch(/<img[^>]+src="[^"]*deepseek-v2-figure2-hq/);
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mha-mqa-gqa\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/long-context\/"/);
  });
  it('renders the original MHA–MLA comparison with attribution wherever the worked cases use it', async () => {
    const routes = ['textbook/modern-llm/mla-kv-compression', 'textbook/inference/kv-cache-compression-offload',
      'models/sravneniya/karta-arhitekturnyh-liniy', 'research/effektivnyy-attention',
      'models/sravneniya/sravnenie-semeystv'];
    for (const locale of ['', 'en']) {
      for (const route of routes) {
        const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
        expect(html).toMatch(/<img[^>]*src="[^"]+\/v2-figure3-attention\.png"[^>]*alt="[^"]+"/);
        expect(html).toContain('deepseek-v2-tech-report.pdf#page=7');
        expect(html).toContain('/licenses/deepseek-v2-ec98ee3c-MIT.txt');
        expect(html).not.toMatch(/<img[^>]*src="[^"]+\/(?:cs336-mha-gqa-mqa-mla|deepseek-v2-mha-gqa-mqa-ablation|deepseek-v2-mla-ablation)\.png"/);
      }
    }
  });
  it('connects the fixed-prefix family comparison to the decoder and architecture map', async () => {
    const route = 'models/sravneniya/sravnenie-semeystv/';
    const id = 'llama-qwen-deepseek-worked-comparison';
    for (const locale of ['', 'en/']) {
      const page = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = page.match(/<a id="llama-qwen-deepseek-worked-comparison"><\/a>[\s\S]*?(?=<h2[^>]+id="другие-семейства-и-поколения")/)?.[0];
      expect(section).toBeDefined();
      for (const fact of ['70 272', '1,07 ГиБ', '37 748 736', '396 361 728', '11 318 329 344',
        'deepseek-v2-tech-report.pdf#page=7', 'DeepSeek_V3.pdf#page=7',
        '/licenses/deepseek-v2-ec98ee3c-MIT.txt', '/licenses/deepseek-v3-d5c08b3-MIT.txt'])
        expect(section).toContain(fact);
      for (const [name, digest] of [
        ['v2-figure3-attention.png', 'f1457699df16b148d9a69a47d2884d6628912dba292376fc1bb73c0ffc57fa42'],
        ['v3-figure2-block.png', 'ad55ce4f20f25e0f04fa439b3fb733557d254062df7d9f97d406f52915a4e809'],
      ]) {
        const asset = section!.match(new RegExp('<img[^>]+src="([^"]*/'+name.replace('.', '\\.')+')"[^>]+alt="[^"]+"'))?.[1];
        expect(asset).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(asset!)))).digest('hex')).toBe(digest);
      }
      for (const source of ['textbook/modern-llm/decoder-block/', 'models/sravneniya/karta-arhitekturnyh-liniy/']) {
        const parent = await readFile(join(distDir, locale, source, 'index.html'), 'utf8');
        // The textbook localizes internal links; the canonical atlas edition
        // continues to link to the original-language atlas route.
        const targetLocale = source.startsWith('textbook/') ? locale : '';
        expect(parent).toContain(`href="/${targetLocale}${route}#${id}"`);
      }
      expect(section).not.toContain('katex-error');
    }
  });
  it('renders compact GQA examples, complete original results and adjacent chapter links',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/mha-mqa-gqa/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['prefill-decode.png','cached-first-token.png','next-token-attends-cache.png',
      'mha-gqa-mqa.png','uptrain-recycling.png','task-results.png','conversion-methods.png',
      'uptraining-proportion.png','groups-speed.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['grouped_attention','query_pos','2.5379','3.4621','37.5%',
      '31.88','time per sample per TPUv4 chip','трём запускам'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('/source-first-35-36/gqa-');
    expect(html).toMatch(/К модулям:[\s\S]{0,300}href="\/textbook\/#/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mla-kv-compression\/"/);
  });
  it('renders complete resource examples, original plots and uncropped optional course pages',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/transformer-shapes-resources/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['a1-figures1-2-model-block.png','gpt2-output-projection-2.png','tpu-load-weights.gif',
      'jax-scaling-roofline.webp','jax-roofline-batch.png',...Array.from({length:10},(_,i)=>'a1-page-'+(i+18)+'.png')])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['parameter_layout','17576448','27795652608','accumulated_gradient',
      '136.1','226.3','20480','27.80%','3.47%','1.3266','3.3490','61.7%'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(7);
    expect(html).toContain('id="logits-to-next-token-loss"');
    expect(html).toContain('id="gpt2-sized-not-gpt2"');
    expect(html).not.toMatch(/<img[^>]+src="[^"]*\/compute-memory\.png"/);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('/a1-sequence-');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/dense-ffn\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mha-mqa-gqa\/"/);
  });
  it('renders RoPE derivation, scaling/cache examples and all eight original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/rope/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['roformer-figure1.png','roformer-decay.png','pi-extrapolation.png',
      'position-interpolation.png','yarn-method-summary.png','barbero-llama31-frequency-figure10.png',
      'barbero-previous-token-construction-figure1.png','barbero-gemma-position-heads-figure5.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['3.701805','0.00232628','63496.04','1.29648','stale_kv',
      'Причинная маска уже задаёт порядок','не измерение attention logits','0.37112','0.86540'])
      expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).toContain('id="rope-coordinate-layout"');
    expect(html).toContain('id="rope-previous-token-head"');
    expect(html).toContain('id="rope-llama31-scaling"');
    expect(html).not.toContain('katex-error');
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(rope-implementation|rope-decay)\.png/);
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/norm-gating-residual\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/dense-ffn\/"/);
  });
  it('renders normalization examples, source figures and the stability explanations',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/norm-gating-residual/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['xiong-pre-post-norm.png','xiong-gradient-norms.png',
      'cs336-silu-vs-relu.png','chameleon-norm-qk-dropout.png','chameleon-training-ablations.png',
      'gemma-scope-figure1.svg','xiong-warmup.png','tajima-figure1-ffn.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['0.03162','1.66667','1.0204','0.1230','4000 шагами warm-up',
      'naive = large','logsumexp','10.0063','порядок округления различается',
      'QK normalization','Z-loss','Logit soft-capping'])expect(text).toContain(fact);
    expect(html).not.toMatch(/<img[^>]+src="[^"]*rmsnorm-paper-results/);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/decoder-block\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/rope\/"/);
  });
  it('renders the warm-up experiment and its intact four-panel figure in both normalization routes', async () => {
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'textbook/modern-llm/norm-gating-residual/index.html'),'utf8');
      const warm=html.match(/id="что-показали-эксперименты-с-warm-up"[\s\S]*?(?=<a id="rmsnorm-validation")/)?.[0];
      expect(warm).toBeDefined();
      expect(warm).toContain('https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=4');
      expect(warm).toContain('На всех четырёх панелях архитектура post-LN');
      const src=warm?.match(/<img[^>]+src="([^"]*\/xiong-warmup\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir,decodeURIComponent(src!)))).digest('hex'))
        .toBe('cc217f6c59fab276798f9f72d001dd82a13382ef44ef44990d1cbf8ff74f62ef');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('reaches the Gemma 2 normalization example from the family and reference pages in both locales', async () => {
    const route='textbook/modern-llm/norm-gating-residual/';
    const anchor='gemma2-sandwich-norm';
    for(const locale of ['', 'en/']){
      const get=(path:string)=>readFile(join(distDir,locale,path,'index.html'),'utf8');
      const [chapter,reference,atlas]=await Promise.all([get(route),get('reference'),get('models/semeystva/gemma')]);
      const section=chapter.match(/id="gemma2-sandwich-norm"[\s\S]*?(?=<a id="normalization-evidence")/)?.[0];
      expect(section).toBeDefined();
      const src=section?.match(/<img[^>]+src="([^"]*\/gemma-scope-figure1\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir,decodeURIComponent(src!)))).digest('hex'))
        .toBe('a89a4f331b2aefab50191633add37e4e80691acf63cd325f3ca309653c12b424');
      expect(section).toContain('https://arxiv.org/html/2408.05147v2#S3.F1');
      expect(section).not.toContain('katex-error');
      expect(reference).toContain('href="/'+locale+route+'#'+anchor+'"');
      // The untranslated atlas page links to the canonical chapter.
      expect(atlas).toContain('href="/'+route+'#'+anchor+'"');
      if(locale) expectCanonicalEditionNotice(chapter);
    }
  });
  it('renders the worked decoder stack, SwiGLU intermediates and original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/decoder-block/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['cs336-prenorm-block.png','mha-gqa-mqa.png',
      'gpt2-transformer-block-vectors-2.png','mobilellm-design-figure3.png','mistral-chunk-mask.png',
      'mobilellm-depth-figure4.png','han-weight-decay-figure3.png','gemma-3-kv-cache.svg','hf-kv-cache.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const name of ['ffn-ratio-evidence-slide.png','aspect-ratio-evidence-slide.png','weight-decay-evidence-slide.png'])
      expect(html).not.toContain(name);
    for(const fact of ['Два блока и два кэша','Почему кэш для генерации не заменяет граф обучения',
      '−1.5672','1.8936','−0.5123','1.2456','192 байта','copy.deepcopy(model)',
      'строке станет нулевым','0.3164','0.2356','start_pos=1'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/llama-architecture\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/norm-gating-residual\/"/);
  });
  it('renders the first-token calculation and Hugging Face source figure in both decoder routes', async () => {
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'textbook/modern-llm/decoder-block/index.html'),'utf8');
      const section=html.match(/<a id="decoder-first-token"><\/a>[\s\S]*?(?=<h3\b[^>]*id="почему-кэш)/)?.[0];
      expect(section).toBeDefined();
      expect(section).toMatch(/<img[^>]+src="[^"]*\/hf-kv-cache\.png"[^>]+alt="[^"]+"/);
      expect(section).toContain('https://huggingface.co/blog/continuous_batching#kv-cache');
      expect(section).toContain('generation.py#L160-L186');
      expect(section).toMatch(/0\.1996, 1\.1409,<br\s*\/?>1\.6305, 0\.0027/);
      expect(section).toMatch(/1\.0877, 0\.8951,<br\s*\/?>0\.2321, 1\.4006/);
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('connects the cached SDPA walkthrough and the original Mistral mask to both topic indexes',async()=>{
    for (const locale of ['', 'en/']) {
      const route='textbook/modern-llm/decoder-block/';
      const html=await readFile(join(distDir,locale,route,'index.html'),'utf8');
      const section=html.match(/<a id="decoder-rectangular-mask"><\/a>[\s\S]*?(?=<h3\b[^>]*id="два-блока-и-два-кэша")/)?.[0];
      expect(section).toBeDefined();
      expect(section).toContain('causal_lower_right');
      expect(section).toContain('0.298538');
      expect(section).toContain('https://arxiv.org/pdf/2310.06825v1#page=3');
      expect(section).toMatch(/<img[^>]+src="[^"]*mistral-chunk-mask\.png"[^>]+alt="[^"]+"/);
      expect(section).toContain(`href="/${locale}textbook/modern-llm/long-context/#`);
      const ref=await readFile(join(distDir,locale,'reference/index.html'),'utf8');
      expect(ref).toContain(`href="/${locale}${route}#decoder-rectangular-mask"`);
      const long=await readFile(join(distDir,locale,'textbook/modern-llm/long-context/index.html'),'utf8');
      expect(long).toContain('разрешены четыре позиции, включая текущую');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('publishes the Qwen projection example and the original Gemma cache graph in both locales',async()=>{
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'textbook/modern-llm/decoder-block/index.html'),'utf8');
      const gemma=html.match(/id="gemma3-mixed-kv-cache"[\s\S]*?(?=<h2\b)/)?.[0];
      const qwen=html.match(/id="qwen3-checkpoint-shapes"[\s\S]*?(?=<h3\b)/)?.[0];
      expect(gemma).toMatch(/<img[^>]+src="[^"]*\/gemma-3-kv-cache\.svg"[^>]*alt="[^"]+"/);
      expect(gemma).toContain('https://arxiv.org/html/2503.19786v1#S5.F6');
      expect(gemma).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(gemma).toContain('19,3%');
      for(const detail of ['[2048,1024]','[1024,2048]','596 049 920','448 МиБ','576 МиБ'])
        expect(qwen).toContain(detail);
      expect(qwen).toContain(`href="/${locale}textbook/modern-llm/norm-gating-residual/#qk-normalization"`);
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('opens the four-model decoder route from both home pages', async () => {
    for (const locale of ['', 'en']) {
      const home = await readFile(join(distDir, locale, 'index.html'), 'utf8');
      const link = home.match(/<a href="([^"]*#gpt-decoder-route)">От GPT-1 к LLaMA<\/a>/)?.[1];
      expect(link).toBeDefined();
      const origin = 'https://example.test';
      const target = new URL(link!, new URL(locale ? '/en/' : '/', origin));
      expect(target.origin).toBe(origin);
      expect(target.pathname).toBe(locale ? '/en/textbook/' : '/textbook/');
      const contents = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      expect(contents).toContain(`id="${target.hash.slice(1)}"`);
    }
  });
  it('shows the decoder history beside each model title in both languages', async () => {
    const steps = ['models/gpt-1', 'models/gpt-2', 'models/gpt-3', 'modern-llm/llama-architecture'];
    const names = ['GPT-1', 'GPT-2', 'GPT-3', 'LLaMA v1'];
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const locale of ['', 'en/']) {
      const textbook = `${locale}textbook/`;
      for (const [index, step] of steps.entries()) {
        const html = await readFile(join(distDir, textbook, step, 'index.html'), 'utf8');
        const nav = html.match(/<nav class="decoder-history(?: [^"]*)?"[^>]*>[\s\S]*?<\/nav>/)?.[0];
        expect(nav).toBeDefined();
        expect(nav).toContain(locale ? 'aria-label="From GPT-1 to LLaMA"' : 'aria-label="От GPT-1 к LLaMA"');
        expect(nav).toMatch(new RegExp(`aria-current="page"[^>]*>${names[index]}<`));
        expect(nav?.match(/aria-current="page"/g)).toHaveLength(1);
        const links = [...nav!.matchAll(/href="([^"]+)"/g)].map(match => match[1]);
        expect(links).toEqual([
          ...steps.filter(other => other !== step).map(other => `${base}/${textbook}${other}/`),
          `${base}/${textbook}#gpt-decoder-route-comparison`,
        ]);
      }
      const unrelated = await readFile(join(distDir, textbook, 'models/t5/index.html'), 'utf8');
      expect(unrelated).not.toMatch(/<nav class="decoder-history(?: [^"]*)?"/);
    }
  });
  it('compares next-token loss in a common unit using the original Pile plot in both locales', async () => {
    for (const locale of ['', 'en/']) {
      const index = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      expect(index).toContain('href="#gpt-decoder-route-bpb"');
      const section = index.match(/id="gpt-decoder-route-bpb"[\s\S]*?(?=<a id="gpt-decoder-route-prompt-order")/)?.[0];
      expect(section).toBeDefined();
      for (const part of ['cats', '2.83', '1.68', '0.75', ...(locale ? ['1.2253', '0.7177'] : ['1,2253', '0,7177']), '508',
        'class="katex-display"', 'https://arxiv.org/html/2101.00027v1#S3.F2',
        'https://arxiv.org/html/2101.00027v1#S3.T2', 'https://creativecommons.org/licenses/by/4.0/']) {
        expect(section).toContain(part);
      }
      expect(section).toContain(`href="/${locale}textbook/modern-llm/llama-architecture/#gpt-llama-tokenization"`);
      expect(section).not.toContain('katex-error');
      const plot = section!.match(/<img[^>]+src="([^"]*\/pile-gpt2-gpt3-figure2\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(plot).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(plot!)))).digest('hex'))
        .toBe('ba6562096ae1648e1f57a66ce73e765254fdd35ede23bc27746e4aec687b2753');
    }
  });
  it('links the four causal decoder chapters in sequence after T5',async()=>{
    const modelRoot='/textbook/models/';
    const llama='/textbook/modern-llm/llama-architecture/';
    const index=await readFile(join(distDir,'textbook/index.html'),'utf8');
    const intro=index.match(/<h2[^>]*id="от-gpt-1-к-llama"[^>]*>[\s\S]*?(?=<h2\b)/)?.[0];
    expect(intro).toBeDefined();
    const paths=['gpt-1/','gpt-2/','gpt-3/'].map(model=>modelRoot+model);
    const positions=[...paths,llama+'#gpt3-to-llama1'].map(path=>intro!.indexOf(`href="${path}"`));
    expect(positions.every((position,i)=>position>=0&&(i===0||position>positions[i-1]))).toBe(true);
    const chapter=await readFile(join(distDir,llama.slice(1),'index.html'),'utf8');
    expect(chapter).toContain('id="gpt-llama-comparison"');
    const comparison=chapter.match(/<a id="gpt-llama-comparison"><\/a>[\s\S]*?(?=<h2\b[^>]*id="от-токенов-к-вероятности-продолжения")/)?.[0];
    expect(comparison).toBeDefined();
    const generations=comparison!.match(/<table>[\s\S]*?<\/table>/)?.[0];
    expect(generations).toBeDefined();
    expect(generations).toContain('Обучение и задачи');
    for(const [model,recipe] of [['GPT-1 (2018)','BooksCorpus'],['GPT-2 (2019)','WebText'],
      ['GPT-3 (2020)','300 млрд токенов'],['LLaMA v1 (2023)','1,4 трлн токенов']]){
      const row=generations!.match(/<tr>[\s\S]*?<\/tr>/g)?.find(candidate=>candidate.includes(model));
      expect(row).toContain(recipe);
    }
    for(const value of ['512 токенов','1024 токена','2048 токенов','BooksCorpus',
      'RMSNorm','SwiGLU'])
      expect(comparison).toContain(value);
    for(const figure of ['xiong-pre-post-norm.png','mistral-sliding-window.png'])
      expect(comparison).toMatch(new RegExp('<img[^>]+src="[^"]*/'+figure.replaceAll('.','\\.')+'"'));
    expect(comparison).toContain('href="https://arxiv.org/html/2310.06825v1#S2.F1"');
    expect(comparison).toContain('Ширину полосы GPT-3 авторы не приводят.');
    expect(comparison).toContain(`href="${paths[2]}#gpt3-full-local-attention"`);
    const applications=chapter.match(/id="gpt-llama-beyond-block"[\s\S]*?(?=<h2\b[^>]*id="источники-и-дальнейшее-чтение")/)?.[0];
    expect(applications).toContain('id="gpt3-demo-order"');
    expect(applications).toContain('href="https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=3"');
    for(const value of ['WebText2','Premise: The door is open.','Relation: contradiction','zhao21c-figure2.png'])
      expect(applications).toContain(value);
    expect(chapter).toContain('id="от-gpt-3-к-первой-llama"');
    const gpt3=await readFile(join(distDir,paths[2].slice(1),'index.html'),'utf8');
    expect(gpt3).toContain(`href="${llama}#`);
    expect(gpt3).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/llama-architecture\/"/);
    for(const prefix of ['','en/']){
      const comparisonPage=await readFile(join(distDir,prefix,llama.slice(1),'index.html'),'utf8');
      const gpt3Page=await readFile(join(distDir,prefix,paths[2].slice(1),'index.html'),'utf8');
      const steps=['/textbook/models/bert-roberta-deberta/','/textbook/models/t5/',...paths,llama];
      const localized=steps.map(path=>'/'+prefix+path.slice(1));
      const contents=await readFile(join(distDir,prefix,'textbook/index.html'),'utf8');
      const curriculum=contents.match(/<h2\b[^>]*id="(?:основной-курс|core-course)"[^>]*>[\s\S]*/)?.[0] ?? '';
      const stops=localized.map(path=>curriculum.indexOf(`href="${path}"`));
      expect(stops.every((position,i)=>position>=0&&(i===0||position>stops[i-1]))).toBe(true);
      for(let i=1;i<localized.length;i++){
        const html=await readFile(join(distDir,localized[i].slice(1),'index.html'),'utf8');
        const prevLabel=prefix?'Previous':'Назад';
        const nextLabel=prefix?'Next':'Далее';
        expect(html).toContain(`<strong>${prevLabel}:</strong> <a href="${localized[i-1]}"`);
        expect(html).toContain(`href="${localized[i-1]}" rel="prev"`);
        if(i+1<localized.length){
          expect(html).toContain(`<strong>${nextLabel}:</strong> <a href="${localized[i+1]}"`);
          expect(html).toContain(`href="${localized[i+1]}" rel="next"`);
        }
      }
      if(prefix){
        expect(comparisonPage).toMatch(/<h1[^>]*>LLaMA as a baseline architecture<\/h1>/);
        expect(comparisonPage).not.toContain('Canonical-language chapter.');
        expect(gpt3Page).toContain('href="/en/textbook/modern-llm/llama-architecture/#gpt-llama-comparison"');
      }
      expect(comparisonPage).toContain(`href="/${prefix}${paths[2].slice(1)}#gpt3-candidate-scores"`);
      const ffn=comparisonPage.match(/<a id="gpt-llama-ffn"><\/a>[\s\S]*?(?=<h3\b)/)?.[0];
      expect(ffn).toMatch(/<img[^>]+src="[^"]*\/tajima-figure1-ffn\.svg"[^>]*alt="[^"]+"/);
      expect(ffn).toContain('https://arxiv.org/html/2506.23225v1#S1.F1');
      expect(ffn).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(ffn).toContain(prefix ? '0.78%' : '0,78%');
      expect(comparisonPage).not.toContain('katex-error');
      expect(gpt3Page).toContain('id="gpt3-candidate-scores"');
      const nextModels=gpt3Page.match(/id="gpt3-next-models"[\s\S]*?(?=<h2\b[^>]*(?:источники-и-дальнейшее-чтение|sources-and-further-reading))/)?.[0];
      expect(nextModels).toBeDefined();
      const legacyHeading=prefix?'two-paths-after-gpt-3':'после-gpt-3-llama-и-t5';
      expect(nextModels).toContain(`<a id="${legacyHeading}"></a>`);
      expect(nextModels).toContain(prefix?'From GPT-3 to the first LLaMA':'От GPT-3 к первой LLaMA');
      expect(nextModels).toContain('https://arxiv.org/abs/1910.10683v1');
      const history=nextModels!.indexOf(`href="/${prefix}${llama.slice(1)}#gpt-llama-comparison"`);
      const course=nextModels!.indexOf(`href="/${prefix}textbook/models/t5/"`);
      expect(history).toBeGreaterThanOrEqual(0);
      expect(course).toBeGreaterThan(history);
      const masks=gpt3Page.match(/id="gpt3-full-local-attention"[\s\S]*?(?=<h3\b[^>]*(?:сколько-места|space-for-demonstrations))/)?.[0];
      expect(masks).toMatch(/<img[^>]+src="[^"]*\/mistral-sliding-window\.png"/);
      expect(masks).toContain('href="https://arxiv.org/html/2310.06825v1#S2.F1"');
    }
  });
  it('explains GPT-3 answer wording with the original COPA diagram in both locales', async () => {
    const note = '/sources/papers/surface-form-competition/';
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, 'textbook/models/gpt-3/index.html'), 'utf8');
      const section = chapter.match(/<a id="gpt3-surface-form-competition"><\/a>[\s\S]*?(?=<span id="архитектура-знакомый-блок-в-новом-масштабе")/)?.[0];
      expect(section).toBeDefined();
      const image = section!.match(/<img[^>]+src="([^"]*\/holtzman-figure3\.png)"[^>]*alt="[^"]+"/)?.[1];
      expect(image).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(image!)))).digest('hex'))
        .toBe('65312423131c60e94a6d32d433d15f1db00ea339a153263fad2613c2dd9edf1e');
      for (const s of ['COPA', 'HellaSwag', '85.2%', '82.8%', '89.2%', '77.2%', '53.5%'])
        expect(section).toContain(locale ? s : s.replace('.', ','));
      expect(section).toContain('https://arxiv.org/pdf/2104.08315v9#page=8');
      expect(section).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(section).toContain(`href="/${locale}${note.slice(1)}"`);
      expect(section).not.toContain('katex-error');
    }
    for (const locale of ['', 'en/']) {
      const source = await readFile(join(distDir, locale, note, 'index.html'), 'utf8');
      expect(source).toContain(`href="/${locale}textbook/models/gpt-3/#gpt3-surface-form-competition"`);
    }
  });
  it('serves original source-library media locally and keeps the interactive notebook links', async () => {
    await verifyCourseLibraryOriginals(rootDir, distDir);
  });
  it('shows source figures and separates GPT training, prompting, and evaluation effects', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const route = html.match(/<a id="gpt-decoder-route-comparison"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-end")/)?.[0];
      expect(route).toBeDefined();
      expect(html).toContain('href="#gpt-decoder-route-evidence"');
      for (const [file, hash, citation] of [
        ['lu22-figure1.png', 'c39af3473f6711d8b99a2f028b92290279296830f060216b8f66c184a6746e68', 'https://arxiv.org/pdf/2104.08786v2#page=1'],
        ['cs336-prenorm-block.png', '156caf48a686acda4c1b7e4584c292205ba908cda444ee7eddaf773d33b5400e', 'https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13'],
      ]) {
        const img = [...route!.matchAll(/<img[^>]+src="([^"]+)"[^>]+alt="[^"]+"/g)]
          .find(match => match[1].endsWith('/' + file))?.[1];
        expect(img, file).toBeDefined();
        expect(route).toContain(citation);
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(img!)))).digest('hex')).toBe(hash);
      }
      const evidence = route!.split('id="gpt-decoder-route-evidence"')[1];
      for (const value of (locale
        ? ['74.7', '59.9', '52.66%', '63.24%', '76.2%', '72.5%', '86.4%', '78.9%', '79.2%', '81.0%', '80.1%']
        : ['74,7', '59,9', '52,66%', '63,24%', '76,2%', '72,5%', '86,4%', '78,9%', '79,2%', '81,0%', '80,1%'])) {
        expect(evidence).toContain(value);
      }
      expect(evidence).toContain('https://arxiv.org/html/2005.14165v4#S3.SS1.SSS2');
      expect(evidence).toContain('https://arxiv.org/html/2302.13971v1#S2.T3');
      expect(evidence).toContain('class="katex"');
      expect(route).not.toContain('katex-error');
    }
  });
  it('shows the original threshold plot and distinguishes label calibration from model accuracy', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      expect(html).toContain('href="#gpt-decoder-route-calibration"');
      const section = html.match(/<a id="gpt-decoder-route-calibration"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-end")/)?.[0];
      expect(section).toBeDefined();
      const figure = section!.match(/<img[^>]+src="([^"]*\/zhao21c-figure5\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('e711ada6fc3f4f0328b017ea100e10b6ba39b800ca72cd646d53fd1e39432032');
      for (const value of ['61.8%', '64.9%', '73.8%', '59.1%', '79.9%', '95.6%', '95.3%'])
        expect(section).toContain(locale ? value : value.replace('.', ','));
      expect(section).toContain(locale ? '94% at 0.68' : '94% при пороге 0,68');
      expect(section).toContain('Review: N/A');
      expect(section).toContain('https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=5');
      expect(section).toContain('https://arxiv.org/pdf/2102.09690v2#page=12');
      expect(section).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(section).toContain('class="katex"');
      expect(section).not.toContain('katex-error');
    }
  });
  it('renders the original quantization figures and the AWQ scale search in both locales', async () => {
    const journal = JSON.parse(await readFile(join(rootDir,
      'docs/audit/2026-11-11-quantization-source-figures-additions.json'), 'utf8')) as {
        figures: { asset: string; sha256: string; figure_url: string }[];
        historical: { asset: string; figure_url: string }[];
      };
    expect(journal.figures).toHaveLength(3);
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/inference/quantization/index.html'), 'utf8');
      for (const fig of journal.figures) {
        const image = [...html.matchAll(/<img[^>]+src="([^"]+)"[^>]+alt="[^"]+"/g)]
          .find(match => decodeURIComponent(match[1]).endsWith('/' + fig.asset))?.[1];
        expect(image, fig.asset).toBeDefined();
        expect(html).toContain(fig.figure_url);
        const bytes = await readFile(join(distDir, decodeURIComponent(image!)));
        expect(createHash('sha256').update(bytes).digest('hex')).toBe(fig.sha256);
      }
      for (const old of journal.historical) {
        expect(html).toContain(old.figure_url);
        expect(html).not.toContain(old.asset.split('/').at(-1)! + '" alt=');
        await expect(access(join(distDir, 'assets', old.asset))).rejects.toThrow();
      }
      const search = html.match(/id="awq-calibration-grid"[\s\S]*?(?=<h3\b)/)?.[0];
      expect(search).toContain(locale ? '0.000593' : '0,000593');
      expect(search).toContain(locale ? '0.067684' : '0,067684');
      expect(html).toContain('/licenses/awq-d6e797a4-MIT.txt');
      expect(html).not.toContain('katex-error');
    }
  });
  it('carries the same entailment prompt from GPT-2 through GPT-3 and the first LLaMA', async () => {
    for (const locale of ['', 'en/']) {
      const base = `/${locale}textbook/`;
      const gpt2 = await readFile(join(distDir, locale, 'textbook/models/gpt-2/index.html'), 'utf8');
      const gpt3 = await readFile(join(distDir, locale, 'textbook/models/gpt-3/index.html'), 'utf8');
      const llama = await readFile(join(distDir, locale, 'textbook/modern-llm/llama-architecture/index.html'), 'utf8');
      const worked = (html: string, id: string) =>
        html.match(new RegExp(`id="${id}"[\\s\\S]*?<pre\\b[^>]*><code>([\\s\\S]*?)</code></pre>`))?.[1]
          ?.replace(/<div class="ec-line">/g, '\n').replace(/<[^>]+>/g, '').trim();
      const single = worked(gpt2, 'gpt2-relation-prefix');
      const shots = worked(gpt3, 'gpt3-relation-shots');
      const target = 'Premise: A cat sleeps.\nHypothesis: An animal rests.\nRelation:';
      expect(single).toContain(target);
      expect(shots).toContain(target);
      expect(shots).toContain('Relation: entailment');
      expect(shots).toContain('Relation: contradiction');
      expect(shots?.slice(shots.indexOf(target))).not.toContain('entailment');
      expect(worked(llama, 'gpt1-to-gpt2')).toBe(target);
      const oneShot = worked(llama, 'gpt2-to-gpt3')?.replace(/\n{3,}/g, '\n\n');
      expect(oneShot).toBe('Premise: The door is open.\nHypothesis: The door is shut.\nRelation: contradiction\n\n' + target);
      const gpt2Transition = llama.match(/id="gpt1-to-gpt2"[\s\S]*?(?=<a id="gpt2-to-gpt3")/)?.[0];
      const projection = gpt2Transition?.match(/<img[^>]+src="([^"]*\/gpt2-output-projection-2\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(projection).toBeDefined();
      expect(gpt2Transition).toContain('href="https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png"');
      expect(gpt2Transition).toContain('href="https://creativecommons.org/licenses/by-nc-sa/4.0/"');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(projection!)))).digest('hex'))
        .toBe('61a9cc4b3dedaa4924de383f45572679280e43cde0f0d4ef810da59bb6846309');
      const anli = llama.match(/id="gpt2-to-gpt3"[\s\S]*?(?=<a id="gpt3-to-llama1")/)?.[0];
      for (const fragment of ['#A8.T1', '#S3.F9', '#A7.F10']) {
        expect(anli).toContain('href="https://arxiv.org/html/2005.14165v4' + fragment + '"');
      }
      expect(anli).toContain(locale ? '35.1% with one and' : '35,1% с одним и');
      expect(anli).toContain(locale ? '40.2%' : '40,2%');
      expect(gpt2).toContain(`href="${base}models/gpt-3/#gpt3-relation-shots"`);
      expect(gpt3).toContain(`href="${base}models/gpt-2/#gpt2-relation-prefix"`);
      expect(llama).toContain(`href="${base}models/gpt-3/#gpt3-relation-shots"`);
      const figure = gpt3.match(/<img[^>]+src="([^"]*\/min-figure2-demonstrations\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(gpt3).toContain('href="https://aclanthology.org/2022.emnlp-main.759.pdf#page=2"');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('0c9388f07b5f6f8bc6c2dda6e4336e837a61250dcf60568a7270b05daf69d587');
      const order = ['id="gpt-llama-comparison"', 'cs336-model-overview.png', 'cs336-prenorm-block.png',
        'id="gpt-llama-beyond-block"', 'llama1-figure3-natural-questions.png', 'sardana24a-figure1.png']
        .map(fragment => llama.indexOf(fragment));
      expect(order.every((position, index) => position >= 0 && (index === 0 || position > order[index - 1]))).toBe(true);
      for (const page of [gpt2, gpt3, llama]) expect(page).not.toContain('katex-error');
    }
  });
  it('connects the same sentiment task and the decoder changes in both locales', async () => {
    const book = '/textbook/';
    const figure = 'd2l-gpt-decoder-original.svg';
    const source = 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#fig-gpt-decoder-only';
    for (const prefix of ['', '/en']) {
      const html = await readFile(join(distDir, prefix, book, 'index.html'), 'utf8');
      const route = html.match(/<a id="gpt-decoder-route-comparison"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-end")/)?.[0];
      expect(route).toBeDefined();
      const sentiment = route!.match(/id="gpt-decoder-route-sentiment"[\s\S]*?(?=<a id="gpt-decoder-route-architecture")/)?.[0];
      expect(sentiment).toBeDefined();
      const generations = ['GPT-1 (2018)', 'GPT-2 (2019)', 'GPT-3 (2020)', 'LLaMA v1 (2023)'];
      let previous = -1;
      for (const label of generations) {
        const index = sentiment!.indexOf(label);
        expect(index).toBeGreaterThan(previous);
        previous = index;
      }
      expect(sentiment).toMatch(/<code\b[^>]*>i liked it<\/code>/);
      const gpt1 = sentiment!.slice(0, sentiment!.indexOf('GPT-2 (2019)'));
      for (const token of ['Start', 'Extract', 'very', 'positive', 'negative']) {
        expect(gpt1).toMatch(new RegExp(`<code\\b[^>]*>${token}</code>`));
      }
      expect(gpt1).toContain('https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8');
      const classifier = gpt1.match(/<img[^>]+src="([^"]*\/cs224n-decoder-classifier\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(classifier).toBeDefined();
      expect(gpt1).toContain('https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=41');
      expect(gpt1).toContain('https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/#logistics');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(classifier!)))).digest('hex'))
        .toBe('c68a1e607ec6d776d8070a1025ed768c60c5cfeeb3b71ace3d230a9a626a196f');
      const vocabulary = gpt1.match(/<img[^>]+src="([^"]*\/cs224n-vocabulary-head\.png)"[^>]+alt="[^"]+"/);
      expect(vocabulary).toBeDefined();
      expect(vocabulary!.index).toBeGreaterThan(gpt1.indexOf(classifier!));
      expect(gpt1).toContain('https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=42');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(vocabulary![1])))).digest('hex'))
        .toBe('f9a668f5be174151f5d2464540ca58f4f85217c01687e62d949ae93862246154');
      const gpt2 = sentiment!.slice(sentiment!.indexOf('GPT-2 (2019)'), sentiment!.indexOf('GPT-3 (2020)'));
      const projection = gpt2.match(/<img[^>]+src="([^"]*\/gpt2-output-projection-2\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(projection).toBeDefined();
      expect(gpt2).toContain('https://jalammar.github.io/illustrated-gpt2/#model-output');
      expect(gpt2).toContain('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(projection!)))).digest('hex'))
        .toBe('61a9cc4b3dedaa4924de383f45572679280e43cde0f0d4ef810da59bb6846309');
      const promptConstruction = sentiment!.match(/<img[^>]+src="([^"]*\/lu22-figure2-prompt-order\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(promptConstruction).toBeDefined();
      expect(sentiment!.indexOf(promptConstruction!)).toBeGreaterThan(sentiment!.indexOf('GPT-3 (2020)'));
      expect(sentiment!.indexOf(promptConstruction!)).toBeLessThan(sentiment!.indexOf('LLaMA v1 (2023)'));
      expect(sentiment).toContain('https://arxiv.org/pdf/2104.08786v2#page=2');
      expect(sentiment).toContain('href="#gpt-decoder-route-prompt-order"');
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(promptConstruction!)))).digest('hex'))
        .toBe('415635674ddf512108848bf3513ba69699fd9527d47889c5ddda399de85e83ef');
      const stopsInExample = ['gpt-route-gpt1', 'gpt-route-gpt2', 'gpt-route-gpt3', 'gpt-route-llama'];
      let lastStop = 0;
      for (const id of stopsInExample) {
        const stop = sentiment!.indexOf(`id="${id}"`);
        expect(stop).toBeGreaterThan(lastStop);
        expect(html).toContain(`href="#${id}"`);
        lastStop = stop;
      }
      const controls = sentiment!.match(/id="gpt-route-demonstration-controls"[\s\S]*?(?=<a id="gpt-route-llama")/)?.[0];
      expect(controls).toBeDefined();
      expect(controls).toContain('https://aclanthology.org/2022.emnlp-main.759.pdf#page=1');
      expect(controls).toContain('https://creativecommons.org/licenses/by/4.0/');
      for (const value of ['41%', '58%', '55%', 'macro-F1']) expect(controls).toContain(value);
      const controlFigure = controls!.match(/<img[^>]+src="([^"]*\/min-figure1-controls\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(controlFigure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(controlFigure!)))).digest('hex'))
        .toBe('582c77d6fde24289fca7a14dc13a5ea2bf69516ccd1b3a7482867ab548bcd1ec');
      expect(sentiment).not.toMatch(/zhao21c-figure4\.png|lu22-figure1\.png/);
      const order = route!.split('<a id="gpt-decoder-route-prompt-order"></a>')[1];
      expect(order).toBeDefined();
      expect(order).toMatch(/<code\b[^>]*>N N P P<\/code>/);
      expect(order).toMatch(/<code\b[^>]*>P P N N<\/code>/);
      expect(order).toContain('lu22-figure1.png');
      expect(order).toContain('https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=4');
      const orderFigure = order.match(/<img[^>]+src="([^"]*\/zhao21c-figure4\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(orderFigure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(orderFigure!)))).digest('hex'))
        .toBe('73663ba1da6fab655de865af6a5326902e8f95aa84f1fb2835ab7bd80092276e');
      expect(sentiment).toContain('https://aclanthology.org/2023.findings-acl.527.pdf#page=1');
      expect(sentiment).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(sentiment).toContain(`href="${prefix}${book}modern-llm/llama-architecture/#gpt-llama-label-mapping"`);
      const promptFigure = sentiment!.match(/<img[^>]+src="([^"]*\/pan-icl-figure1\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(promptFigure).toBeDefined();
      const promptOriginal = await readFile(join(distDir, decodeURIComponent(promptFigure!)));
      expect(createHash('sha256').update(promptOriginal).digest('hex'))
        .toBe('d1c1293d0138dfc5554b2fcb33930faffb61e4d03a79b02f40ab19759222235a');
      const llamaStop = sentiment!.slice(sentiment!.indexOf('id="gpt-route-llama"'));
      const accuracy = llamaStop.match(/<img[^>]+src="([^"]*\/pan-icl-figure2\.png)"[^>]+alt="[^"]+"/);
      expect(accuracy).toBeDefined();
      expect(accuracy!.index).toBeGreaterThan(llamaStop.indexOf(promptFigure!));
      expect(llamaStop).toContain('https://aclanthology.org/2023.findings-acl.527.pdf#page=3');
      expect(llamaStop).toContain('https://creativecommons.org/licenses/by/4.0/');
      for (const value of ['45%', '60%', '48%', '57%']) expect(llamaStop).toContain(value);
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(accuracy![1])))).digest('hex'))
        .toBe('f08610e63e52eec5c66001f6fe3f25254fe82b849455beab0e4d3f526f8846c6');
      const evidence = route!.match(/id="gpt-decoder-route-evidence"[\s\S]*?(?=<a id="gpt-decoder-route-bpb")/)?.[0];
      for (const value of ['SST-2', prefix ? '91.3%' : '91,3%', prefix ? '84.0%' : '84,0%'])
        expect(evidence).toContain(value);
      const architecture = route!.match(/id="gpt-decoder-route-architecture"[\s\S]*?<table>[\s\S]*?<\/table>/)?.[0];
      expect(architecture).toMatch(/512[\s\S]*?1,?024[\s\S]*?2,?048[\s\S]*?2,?048[\s\S]*?RMSNorm[\s\S]*?RoPE[\s\S]*?SwiGLU/);
      const stops = [
        ['models/gpt-1/', 'gpt1-decoder-architecture'],
        ['models/gpt-2/', 'gpt2-pre-norm'],
        ['models/gpt-3/', 'gpt3-full-local-attention'],
        ['modern-llm/llama-architecture/', 'gpt-llama-comparison'],
      ];
      for (const [path, id] of stops) {
        expect(route).toContain(`href="${prefix}${book}${path}#${id}"`);
        const page = await readFile(join(distDir, prefix, book, path, 'index.html'), 'utf8');
        expect(page).toContain(`id="${id}"`);
      }
      const translations = [
        ['models/gpt-2/', prefix ? 'translation-and-answers-through-prefixes' : 'перевод-и-ответ-через-префикс'],
        ['models/gpt-3/', prefix ? 'translation-direction-and-parallel-examples' : 'перевод-направление-и-параллельные-примеры'],
      ];
      const routeLinks = Array.from(route!.matchAll(/href="([^"]+)"/g), match => decodeURIComponent(match[1]));
      for (const [path, id] of translations) {
        expect(routeLinks).toContain(`${prefix}${book}${path}#${id}`);
        const chapter = await readFile(join(distDir, prefix, book, path, 'index.html'), 'utf8');
        const example = chapter.match(new RegExp(`id="${id}"[\\s\\S]*?(?=<h[23]\\b)`))?.[0];
        expect(example).toContain('The bird sings.');
        if (id === 'translation-direction-and-parallel-examples') {
          const prompt = example!.match(/<pre\b[\s\S]*?<\/pre>/)?.[0];
          expect(prompt).toContain('Le chat dort.');
          expect(prompt).toContain('Le chien court.');
          expect(prompt).toContain('The bird sings. =');
          expect(prompt).not.toContain('oiseau chante.');
        }
      }
      const plate = route!.match(/<p><a[^>]+><img[^>]+src="[^"]*\/d2l-gpt-decoder-original\.svg"[^>]+alt="[^"]+"[^>]*><\/a><\/p>\s*<p><em>[\s\S]*?<\/em><\/p>/)?.[0];
      expect(plate).toBeDefined();
      expect(plate).toContain(source);
      expect(plate).toContain('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(route).toContain(`href="${prefix}${book}models/t5/"`);
      expect(sentiment).toContain('https://arxiv.org/html/2005.14165v4#S2.F1');
      expect(route).not.toContain('gpt-3-xshot.svg');
      const emitted = plate!.match(new RegExp('<img[^>]+src="([^"]*' + figure + ')"'))![1];
      const asset = await readFile(join(distDir, decodeURIComponent(emitted)), 'utf8');
      expect(createHash('sha256').update(asset).digest('hex'))
        .toBe('c586784e0c1404f7fe0ffcda1354f02cfe64f708f80eb9c6e5fc1d32248ea964');
      const normalization = route!.match(/<img[^>]+src="([^"]*\/xiong-pre-post-norm\.png)"[^>]+alt="[^"]+"[^>]*><\/a><\/p>\s*<p><em>([\s\S]*?)<\/em><\/p>/);
      expect(normalization?.[2]).toContain('https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2');
      expect(normalization?.[2]).toContain('https://creativecommons.org/licenses/by/4.0/');
      const sourceCrop = await readFile(join(distDir, decodeURIComponent(normalization![1])));
      expect(createHash('sha256').update(sourceCrop).digest('hex'))
        .toBe('fb882ba9d546192eafc0ee81f51244043faa986a1641138e20b50e998e0efdaf');
      const decoder = route!.match(/id="gpt-decoder-route-architecture"[\s\S]*?(?=<a id="gpt-decoder-route-data")/)?.[0];
      const mask = decoder?.match(/<img[^>]+src="([^"]*\/mistral-sliding-window\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(mask).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(mask!)))).digest('hex'))
        .toBe('2c0f0d17c4d9e06d3f3f6c52d3a2e64603a81dfac1399bc8d12996a2bae0c50e');
      for (const fragment of [
        'https://arxiv.org/html/2310.06825v1#S2.F1',
        'https://arxiv.org/html/2310.06825v1/swa.png',
        'https://arxiv.org/html/2005.14165v4#S2.SS1',
        `href="${prefix}${book}modern-llm/llama-architecture/#gpt-llama-ffn"`,
        prefix ? 'does not specify its window width' : 'ширину окна GPT-3 авторы не указали',
      ]) expect(decoder).toContain(fragment);
      const overview = html.match(/<a id="gpt-decoder-route"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-comparison")/)?.[0];
      expect(overview).toBeDefined();
      expect(overview).not.toContain('<img');
      const comparison = overview!.match(/<table>[\s\S]*?<\/table>/)?.[0];
      expect(comparison?.match(/<tbody>[\s\S]*?<\/tbody>/)?.[0].match(/<tr>/g)).toHaveLength(4);
      expect(comparison?.match(/<thead>[\s\S]*?<\/thead>/)?.[0].match(/<th>/g)).toHaveLength(2);
      for (const label of generations) expect(comparison).toContain(label);
      for (const mechanism of ['LayerNorm', 'RMSNorm', 'RoPE', 'SwiGLU', 'BooksCorpus', 'WebText'])
        expect(comparison).toContain(mechanism);
      for (const id of ['sentiment', 'architecture', 'data'])
        expect(overview).toContain(`#gpt-decoder-route-${id}"`);
      const chapterLinks = Array.from(overview!.matchAll(/<a href="([^"]+)">(?:GPT-[123]|LLaMA v1) \(\d{4}\)<\/a>/g), match => match[1]);
      expect(chapterLinks).toEqual(stops.map(([path]) => `${prefix}${book}${path}`));
      const overviewLinks = Array.from(overview!.matchAll(/href="([^"]+)"/g), match => decodeURIComponent(match[1]));
      expect(overviewLinks).toContain('#gpt-decoder-route-comparison');
      const workedAnchors = [
        prefix ? 'textual-entailment' : 'логическое-следование',
        'gpt2-relation-prefix',
        'gpt3-relation-shots',
        'gpt3-to-llama1',
      ];
      for (const [index, [path]] of stops.entries()) {
        const id = workedAnchors[index];
        expect(overviewLinks).toContain(`${prefix}${book}${path}#${id}`);
        const page = await readFile(join(distDir, prefix, book, path, 'index.html'), 'utf8');
        expect(page).toContain(`id="${id}"`);
      }
      expect(html.indexOf('id="gpt-decoder-route-comparison"')).toBeGreaterThan(html.indexOf('id="полный-маршрут"'));
      const courseAnchor = prefix ? 'curriculum' : 'полный-маршрут';
      const lead = html.slice(0, html.indexOf('id="gpt-decoder-route"'));
      expect(Array.from(lead.matchAll(/href="([^"]+)"/g), match => decodeURIComponent(match[1])))
        .toContain(`#${courseAnchor}`);
      const ffn = decoder?.match(/<img[^>]+src="([^"]*\/tajima-figure1-ffn\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(ffn).toBeDefined();
      expect(decoder).toContain('https://arxiv.org/html/2506.23225v1#S1.F1');
      expect(decoder).toContain('https://arxiv.org/html/2506.23225v1/fig_swimlu_v2.svg');
      expect(decoder).toContain('https://creativecommons.org/licenses/by/4.0/');
      const d = 4096, gptWidth = 4 * d, llamaWidth = Math.ceil(Math.floor(gptWidth * 2 / 3) / 256) * 256;
      expect((3 * d * llamaWidth) / (2 * d * gptWidth) - 1).toBe(1 / 128);
      for (const width of [gptWidth, llamaWidth])
        expect(decoder?.replace(/[,\u202f]/g, '')).toContain(String(width));
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(ffn!)))).digest('hex'))
        .toBe('e8c8f485f507726c232e1a659346b9a7c4ae242e4d6f5991e8c4f18c674f9df7');
      const data = route!.match(/id="gpt-decoder-route-data"[\s\S]*?(?=<a id="gpt-decoder-route-repeated-data")/)?.[0];
      const checkpoints = data?.match(/<img[^>]+src="([^"]*\/llama1-downstream-training\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(checkpoints).toBeDefined();
      expect(data).toContain('https://arxiv.org/html/2302.13971v1#S3.SS7');
      expect(data).toContain('https://arxiv.org/pdf/2302.13971v1#page=8');
      for (const label of ['HellaSwag', 'SIQA', 'Chinchilla', '71%', '79%']) expect(data).toContain(label);
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(checkpoints!)))).digest('hex'))
        .toBe('9bef2893d15c5ca13cd824ec3f427bb9e35127c431a52d5a15ab745a92d8e7e6');
    }
  });
  it('separates model width and depth from the context window using the original GPT-2 figure', async () => {
    const denseStack = (layers: number, width: number) => 12 * layers * width ** 2;
    expect(denseStack(12, 768)).toBe(84_934_656);
    expect(denseStack(48, 1_600)).toBe(1_474_560_000);
    expect(denseStack(48, 1_600) / denseStack(12, 768)).toBeCloseTo(17.4, 1);
    expect(1_557_611_200 / 124_439_808).toBeCloseTo(12.5, 1);
    expect(denseStack(96, 12_288)).toBe(173_946_175_488);
    expect(4 * 4096 ** 2 + 3 * 4096 * 11008).toBe(202_375_168);
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const section = html.match(/<a id="gpt-decoder-route-depth-width"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-cache")/)?.[0];
      expect(html).toContain('href="#gpt-decoder-route-depth-width"');
      expect(section).toBeDefined();
      const image = section?.match(/<img[^>]+src="([^"]*\/gpt2-sizes-hyperparameters-3\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(image).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(image!)))).digest('hex'))
        .toBe('fd6d84995b9eb0efd8cc3931e8301bacd20dfb2dd02334fdc624c823d4d3d5c3');
      for (const value of ['12Ld^2', '174', '202', 'GPT-3 175B', 'LLaMA-7B', '#gpt-llama-parameter-budget',
        'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'https://arxiv.org/html/2005.14165v4#S2.SS1'])
        expect(section).toContain(value);
      expect(section).not.toContain('katex-error');
    }
  });
  it('counts allowed attention pairs from the full and local masks at each context length', async () => {
    const countPairs = (tokens: number, window = tokens) =>
      Array.from({ length: tokens }, (_, i) => Math.min(i + 1, window))
        .reduce((sum, allowed) => sum + allowed, 0);
    expect([countPairs(5), countPairs(5, 3)]).toEqual([15, 12]);
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const pairs = html.match(/id="gpt-decoder-route-attention-pairs"><\/a><\/p>\s*<p>([\s\S]*?)<\/p>/)?.[1];
      expect(pairs).toContain('class="katex"');
      expect(pairs).not.toContain('katex-error');
      for (const length of [512, 1024, 2048]) {
        const digits = String(countPairs(length)).replace(/\B(?=(\d{3})+(?!\d))/g, '\\,');
        expect(pairs).toContain(digits);
      }
    }
  });
  it('renders the original RoFormer figure and the fixed-vector comparison in both GPT reading routes', async () => {
    const rotatedDot = (key: number, query: number) =>
      Math.cos(key) * Math.cos(query) + Math.sin(key) * Math.sin(query);
    expect(rotatedDot(0, 1)).toBeCloseTo(rotatedDot(3, 4), 12);
    expect(rotatedDot(0, 1)).not.toBeCloseTo(rotatedDot(0, 2), 4);
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const section = html.match(/<a id="gpt-decoder-route-positions"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-data")/)?.[0];
      const figure = section?.match(/<img[^>]+src="([^"]*\/roformer-figure1\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('b1a65eb5e30f1510dce571cedb548b98c1db5c5e57ca04112b856f931a0d2591');
      expect(section).toContain('https://arxiv.org/html/2104.09864v5#S3.SS2');
      expect(section).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
      expect(section).toContain('model.py#L147-L165');
      expect(section).toContain('model.py#L115-L143');
      const tableSize = String(1024 * 768).replace(/(?=\d{3}$)/, () => '\\,');
      expect(section).toContain(`1024\\times768=${tableSize}</annotation>`);
      for (const value of [rotatedDot(0, 1), rotatedDot(0, 2)])
        expect(section).toContain(value.toFixed(4));
      expect(section).toContain('class="katex"');
      expect(section).not.toContain('katex-error');
    }
  });
  it('uses the same original first-token figure in inference physics and benchmarking', async () => {
    const ceiling = (intensity: number) => Math.min(989, intensity * 3.35);
    expect([128, 256, 512].map(x => Number(ceiling(x).toFixed(1)))).toEqual([428.8, 857.6, 989]);
    const streams = [[200, 260, 320, 380], [200, 380]];
    expect(streams.map(xs => (xs.at(-1)! - xs[0]) / (4 - 1))).toEqual([60, 60]);
    expect(streams.map(xs => xs.slice(1).map((x, i) => x - xs[i]))).toEqual([[60, 60, 60], [180]]);
    for (const locale of ['', 'en']) {
      for (const page of ['llm-inference-physics', 'benchmarking-slo']) {
        const html = await readFile(join(distDir, locale, 'textbook/inference', page, 'index.html'), 'utf8');
        expect(html).not.toContain('vllm-latency_diagram.png');
        expect(html).not.toContain('vllm-roofline.png');
        expect(html).toContain('https://www.aleksagordic.com/blog/vllm');
        const originalFile = page === 'llm-inference-physics' ? 'roofline.png' : 'latency_diagram.png';
        expect(html).toContain('/assets/figures/2025-vllm-anatomy/' + originalFile);
        expect(html).toContain('fig-vol3-first-token-accounting');
        expect(html).toContain('creativecommons.org/licenses/by-nc-nd/4.0/');
        const figure = html.match(/<img[^>]+src="([^"]*\/first-token-accounting\.svg)"[^>]+alt="[^"]+"/)?.[1];
        expect(figure).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
          .toBe('cd26640ab6dab6e5877f70ee267584b5edb2486498d013078d32a4bfc4c1ecc5');
        if (page === 'llm-inference-physics') {
          expect(html).toContain('prefill-vs-decode-roofline.svg');
          expect(html).toMatch(/428(?:\{,\}|\.)8/);
          expect(html).toMatch(/857(?:\{,\}|\.)6/);
        } else {
          const example = html.match(/id="stream-events-vs-token-latency"[\s\S]*?(?=<h2\b)/)?.[0];
          expect(example).toContain('200, 260, 320, 380');
          expect(example).toContain('60, 60, 60');
          expect(example).toContain('180');
        }
        expect(html).not.toContain('katex-error');
      }
    }
  });
  it('keeps the vLLM source study illustrated without serving the archived blog copies', async () => {
    const library = JSON.parse(await readFile(join(rootDir, 'docs/audit/2026-11-11-vllm-source-library-figure-reconciliation-additions.json'), 'utf8')) as {
      figures: { asset: string; sha256: string }[];
      retired: { asset: string; figure_url: string }[];
    };
    expect(library.figures).toHaveLength(5);
    expect(library.retired).toHaveLength(10);
    for (const locale of ['', 'en/']) {
      const study = await readFile(join(distDir, locale, 'sources/implementations/vllm/index.html'), 'utf8');
      expect(study).toContain('href="https://www.aleksagordic.com/blog/vllm"');
      expect(study).toContain('href="https://vllm.ai/blog/2025-09-05-anatomy-of-vllm"');
      expect(study).toContain('scheduler.py#L204-L274');
      expect(study).toContain('A: 5 позиций/3 блока; C: 3 позиции/2 блока');
      expect(study).not.toContain('katex-error');
      const images = [...study.matchAll(/<img[^>]+src="([^"]+)"[^>]+alt="[^"]+"/g)].map(match => match[1]!);
      expect(images).toHaveLength(5);
      for (const figure of library.figures) {
        const src = images.find(value => value.endsWith('/'+figure.asset.split('/').at(-1)));
        expect(src).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!)))).digest('hex'))
          .toBe(figure.sha256);
      }
      for (const archive of library.retired) {
        expect(study).toContain(`href="${archive.figure_url}"`);
        expect(study).not.toContain(archive.asset.split('/').at(-1));
      }
    }
    for (const archive of library.retired) {
      for (const asset of [archive.asset, archive.asset.replace('00 Учебник/Assets/', '')]) {
        await expect(readFile(join(distDir, 'assets', asset))).rejects.toMatchObject({code: 'ENOENT'});
      }
    }
  });
  it('separates LMCache paper architecture from version-specific MP transfer paths', async () => {
    const routes = [
      ['textbook/inference/kv-cache-compression-offload', 'lmcache-figure3-workflow.png', 'https://arxiv.org/html/2510.09665v1#S3.F3', '75b3622a95c4c31795872a31822668e95fceb13fab7bf48f7f55559d095a4a2e'],
      ['en/textbook/inference/kv-cache-compression-offload', 'lmcache-figure3-workflow.png', 'https://arxiv.org/html/2510.09665v1#S3.F3', '75b3622a95c4c31795872a31822668e95fceb13fab7bf48f7f55559d095a4a2e'],
      ['en/sources/lmcache', 'lmcache-figure2-overview.png', 'https://arxiv.org/html/2510.09665v1#S3.F2', 'f41522ce56cf0dee4685a70a53a9368cb2f4463d7edfd52e06079a6b4059e460'],
    ];
    for (const [route, file, source, digest] of routes) {
      const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
      const asset = Array.from(html.matchAll(/<img[^>]+src="([^"]+)"[^>]+alt="[^"]+"/g))
        .map(match => match[1]!).find(src => src.endsWith('/' + file));
      expect(asset, route).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(asset!)))).digest('hex')).toBe(digest);
      expect(html).toContain(source);
      expect(html).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(html).toContain('https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/');
      expect(html).toContain('https://blog.lmcache.ai/wp-content/uploads/2026/06/3pathsUpdate-1-edited.png');
      for (const name of ['CUDA IPC', 'POSIX', 'Pickle', '32 MiB', '512'])
        expect(html).toContain(name);
      expect(html).not.toContain('mp-transfer-paths.png');
      expect(html).not.toContain('mp-mode-architecture.png');
      expect(html).not.toContain('katex-error');
    }
    for (const base of ['assets', 'assets/00 Учебник/Assets'])
      for (const file of ['mp-transfer-paths.png', 'mp-mode-architecture.png'])
        await expect(readFile(join(distDir, base, 'Figures/curated/lmcache-blog-2026', file)))
          .rejects.toMatchObject({code: 'ENOENT'});
  });
  it('renders licensed source figures on the long-context, alignment, Jamba, and Mamba reader pages', async () => {
    const pages = [
      ['research/long-context', ['long-context-review-2026-09/lost-in-the-middle.png'],
        ['source-first-37-40-audit/lost-in-the-middle-figure-1.png',
          'source-first-37-40-audit/longformer-attention-patterns.png'], ['aclanthology.org/2024.tacl-1.9.pdf#page=2', '56,1%']],
      ['research/alignment-rlhf-dpo-rlvr', ['dpo/fig1.png'], ['dpo/dpo-pipeline.png'],
        ['arxiv.org/html/2305.18290v3#S1.F1', 'Способ собрать эту пару на схеме не показан.',
          'huyenchip.com/2023/05/02/rlhf.html#rlhf-overview', 'Chip Huyen']],
      ['sources/papers/jamba-ssm-transformer-hybrid',
        ['course-follow-through/jamba-architecture.svg', 'jamba/throughput-figure3-v2.png', 'jamba/fig3.png'],
        ['jamba/fig1.png'], ['arxiv.org/pdf/2403.19887v2#page=6', 'arxiv.org/html/2403.19887v2#S6.F9', '4 ГиБ']],
      ['textbook/modern-llm/sequence-model-alternatives',
        ['mamba/mamba-v2-figure1.svg', 'mamba/s4-representations-original.png',
          'mamba/mamba-v1-selective-copying.svg', 'mamba/mamba-v1-architecture.svg'],
        ['mamba/selective-ssm.png', 'mamba/ssm-three-views.png', 'mamba/mamba-block.png',
          'alternatives-transformer-2026/mamba-selective-copying.png'],
        ['arxiv.org/html/2312.00752v2#S1.F1', 'arxiv.org/html/2312.00752v2#S3.F2',
          'arxiv.org/html/2312.00752v2#S3.F3', '/licenses/s4-e757cef5-Apache-2.0.txt', '1.68704']],
    ] as const;
    for (const [route, include, exclude, references] of pages) {
      const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
      const images = Array.from(html.matchAll(/<img\b[^>]*\bsrc="([^"]+)"[^>]*>/g), m => m[1]);
      for (const path of include)
        expect(images.some(src => src.endsWith('/' + path)), route + ': ' + path).toBe(true);
      for (const path of exclude)
        expect(images.some(src => src.endsWith('/' + path)), route + ': ' + path).toBe(false);
      for (const link of references) expect(html, route).toContain(link);
      expect(html, route).not.toContain('class="katex-error"');
    }
  });
  it('links the Longformer and LongT5 mask examples and retains the complete author figure', async () => {
    const chapter = 'textbook/modern-llm/long-context';
    const overview = 'research/long-context';
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, chapter, 'index.html'), 'utf8');
      const research = await readFile(join(distDir, locale, overview, 'index.html'), 'utf8');
      for (const id of ['longformer-global-attention', 'longt5-transient-global']) {
        expect(html).toContain(`id="${id}"`);
        // With no translated chapter, localized navigation opens the canonical source.
        expect(research).toContain(`href="/${chapter}/#${id}"`);
      }
      expect(html).toContain('https://arxiv.org/html/2004.05150v2#S1.F2');
      expect(research).toContain('https://arxiv.org/html/2004.05150v2#S1.F2');
      expect(research).not.toContain('longformer-attention-patterns.png');
      const section = html.match(/id="longt5-transient-global"[\s\S]*?(?=<h3[^>]*id="кольцевой-кэш)/)?.[0];
      expect(section).toContain('https://aclanthology.org/2022.findings-naacl.55.pdf#page=3');
      expect(section).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(section).not.toContain('katex-error');
      const figure = section!.match(/<img[^>]*src="([^"]*\/longt5-figure2\.png)"[^>]*alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      const original = await readFile(join(distDir, decodeURIComponent(figure!)));
      expect(createHash('sha256').update(original).digest('hex'))
        .toBe('fea0d185f9ce2dd7320edd05334e6ef7625795de3388c6f6ba94b6c7b52505d1');
    }
  });
  it('connects CS336 evaluation to the worked AlpacaEval correction and three original figures', async () => {
    const chapter = 'textbook/evaluation/models-contamination/';
    const paper = 'sources/papers/length-controlled-alpacaeval/';
    const course = 'sources/courses/stanford-cs336-spring-2026/';
    const anchor = '#alpacaeval-length-control';
    const sig = (value: number) => 1 / (1 + Math.exp(-value));
    const longer = sig(-0.2 + 0.8 * Math.tanh(1));
    const equal = sig(-0.2);
    const second = sig(0.6);
    const round = (value: number) => (100 * value).toFixed(1) + '%';
    const calculated = [longer, equal, second, (longer + second) / 2, (equal + second) / 2].map(round);
    expect(calculated).toEqual(['60.1%', '45.0%', '64.6%', '62.3%', '54.8%']);
    const figures = [["causal_graph.png","S3.F2","55402b453ce4ba9bb421b3ce032f366e81e241576e24f23c84267bc749e3254f"],["length_gameability.png","S4.F3","e8490330e4b63b33f689b8dca36986a0dbd6f2010d0468cf8bd0e116a35c8f73"],["chat_correlations.png","S1.F1","6a79ce54dcf68d09864f481db1119db4412caa60272cd90f05d4fd0fc0f588f8"]];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, chapter, 'index.html'), 'utf8');
      const section = html.match(/<a id="alpacaeval-length-control"><\/a>[\s\S]*?(?=<a id="wildbench-checklist")/)?.[0];
      expect(section).toBeDefined();
      for (const [file, sourceAnchor, sha256] of figures) {
        const figure = section!.match(new RegExp('<img[^>]+src="([^"]*/' + file.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'));
        expect(figure, file).toBeDefined();
        expect(section).toContain('https://arxiv.org/html/2404.04475v2#' + sourceAnchor);
        const bytes = await readFile(join(distDir, decodeURIComponent(figure![1])));
        expect(createHash('sha256').update(bytes).digest('hex')).toBe(sha256);
      }
      for (const value of [...calculated.map(value => value.slice(0, -1)), '41.4', '9.7', '0,94', '0,98',
        'creativecommons.org/licenses/by/4.0/', '2404.04475v2#S3.E2']) {
        expect(section).toContain(value);
      }
      expect(section).not.toContain('katex-error');
      expect(section).toContain(`href="/${prefix}${paper}"`);
      const sourceNote = await readFile(join(distDir, prefix, paper, 'index.html'), 'utf8');
      expect(sourceNote).toContain(`href="/${prefix}${chapter}${anchor}"`);
      // The original course hub is mirrored in the EN shell; its Russian
      // textbook links intentionally retain their canonical destinations.
      const courseHub = await readFile(join(distDir, prefix, course, 'index.html'), 'utf8');
      expect(courseHub).toContain(`href="/${chapter}${anchor}"`);
    }
  });
  it('links the original WildBench material and demonstrates the strict winning-length margin', async () => {
    const adjust = (judgment: number, a: number, b: number, k: number) =>
      Math.abs(judgment) === 0.5 && Math.sign(judgment) * (a - b) > k ? 0 : judgment;
    const after = [[0.5, 1049], [-0.5, 2050], [1, 1000]].map(([judgment, b]) =>
      adjust(judgment, 1550, b, 500));
    expect(after).toEqual([0, -0.5, 1]);
    expect(after.reduce((a, b) => a + b, 0) / 3).toBe(1 / 6);
    expect(adjust(-0.5, 1550, 2051, 500)).toBe(0);
    expect(adjust(0.5, 1550, 1049, Infinity)).toBe(0.5);
    expect((8 - 5) * 2 * 10).toBe(60);

    const chapter = 'textbook/evaluation/models-contamination/';
    const paper = 'sources/papers/wildbench/';
    const course = 'sources/courses/stanford-cs336-spring-2026/';
    const anchor = '#wildbench-checklist';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, chapter, 'index.html'), 'utf8');
      const section = html.match(/<a id="wildbench-checklist"><\/a>[\s\S]*?(?=<span id="pairwise-preference-protocol")/)?.[0];
      expect(section).toBeDefined();
      for (const [file, figure, sha256] of [["eval_pipeline_plus.png","S3.F4","b833205f2da31d01b94edc53286b9785d2bc57641d475cff5a9064d44053e91a"],["task_dist.png","S2.F3","237afaff13566084487d0fbd80b15b4c80e0c48df4740349124ac59ca2b85451"]]) {
        const tag = section!.match(new RegExp('<img[^>]+src="([^"]*/' + file.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'));
        expect(tag, file).toBeDefined();
        const bytes = await readFile(join(distDir, decodeURIComponent(tag![1])));
        expect(createHash('sha256').update(bytes).digest('hex')).toBe(sha256);
        expect(section).toContain('https://arxiv.org/html/2406.04770v2#' + figure);
      }
      for (const value of ['1049', '2050', '501', '1/6', '57%', '16%', 'WB-Score',
        'creativecommons.org/licenses/by/4.0/', 'd6b8dcaf377d173d031980f97c16e1a82618c03d/leaderboard/data_dir/_merge_results.py#L94-L99']) {
        expect(section, value).toContain(value);
      }
      expect(section).not.toContain('katex-error');
      expect(section).toContain(`href="/${prefix}${paper}"`);
      const sourceNote = await readFile(join(distDir, prefix, paper, 'index.html'), 'utf8');
      expect(sourceNote).toContain(`href="/${prefix}${chapter}${anchor}"`);
      const courseHub = await readFile(join(distDir, prefix, course, 'index.html'), 'utf8');
      expect(courseHub).toContain(`href="/${chapter}${anchor}"`);
    }
  });
  it('renders the fixed-weight label-mapping experiment and its two original figures in the LLaMA comparison', async () => {
    const llama = 'textbook/modern-llm/llama-architecture/';
    const paper = 'sources/papers/what-in-context-learning-learns/';
    for (const prefix of ['', 'en/']) {
      const chapter = await readFile(join(distDir, prefix, llama, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="gpt-llama-label-mapping"><\/a>[\s\S]*?(?=<a id="training-and-inference-budget")/)?.[0];
      expect(section).toBeDefined();
      for (const [figure, page] of [[1, 1], [2, 3]]) {
        expect(section).toMatch(new RegExp('<img[^>]+src="[^"]*/pan-icl-figure'+figure+'\\.png"[^>]+alt="[^"]+"'));
        expect(section).toContain('https://aclanthology.org/2023.findings-acl.527.pdf#page='+page);
      }
      const changes = prefix ? ['45% to 60%', '48% to 57%'] : ['45% до 60%', '48% до 57%'];
      for (const term of ['Gold', 'Random', 'Abstract', ...changes, 'CC BY 4.0']) {
        expect(section).toContain(term);
      }
      expect(chapter).toContain('href="/'+prefix+paper+'"');
      const note = await readFile(join(distDir, prefix, paper, 'index.html'), 'utf8');
      expect(note).toContain('href="/'+prefix+llama+'#gpt-llama-label-mapping"');
      expect(chapter).not.toContain('katex-error');
    }
  });
  it('renders the three GPT-to-LLaMA transitions with their original source figures', async () => {
    const route = 'textbook/modern-llm/llama-architecture/';
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="gpt-llama-comparison"><\/a>[\s\S]*?(?=<span id="сначала-вся-модель-на-одной-странице")/)?.[0];
      expect(section).toBeDefined();
      const stops = [
        'id="gpt1-to-gpt2"', 'xiong-pre-post-norm.png',
        'id="gpt2-to-gpt3"', 'min-figure2-demonstrations.png', 'mistral-sliding-window.png',
        'id="gpt3-to-llama1"', 'id="gpt-llama-position-shift"',
        'gpt2-input-embedding-positional-encoding-3.png', 'roformer-figure1.png', 'tajima-figure1-ffn.svg',
      ].map(name => section!.indexOf(name));
      expect(stops.every((offset, i) => offset > (i ? stops[i - 1] : -1))).toBe(true);
      for (const url of [
        'https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2',
        'https://aclanthology.org/2022.emnlp-main.759.pdf#page=2',
        'https://arxiv.org/html/2310.06825v1#S2.F1',
        'https://jalammar.github.io/illustrated-gpt2/#input-encoding',
        'https://arxiv.org/pdf/2104.09864v5#page=5',
        'https://arxiv.org/html/2506.23225v1#S1.F1',
      ]) expect(section).toContain(url);
      const plate = section!.match(/<img[^>]+src="([^"]*\/min-figure2-demonstrations\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(plate).toBeDefined();
      const file = await readFile(join(distDir, decodeURIComponent(plate!)));
      expect(createHash('sha256').update(file).digest('hex'))
        .toBe('0c9388f07b5f6f8bc6c2dda6e4336e837a61250dcf60568a7270b05daf69d587');
      expect(chapter).not.toContain('katex-error');
    }
  });
  it('renders LLaMA cached generation, shared-prefix boundaries and the source figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/llama-architecture/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['cs336-model-overview.png','cs336-prenorm-block.png',
      'roformer-figure1.png','cs336-silu-vs-relu.png','llama1-training-loss.png',
      'mha-gqa-mqa.png','llama1-downstream-training.png','sardana24a-figure1.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['def forward(ids, cache=None)', 'query_pos = old + torch.arange(t)',
      'Маска имеет форму 2×5','Последний выбранный ID','2+3+4=9',
      'Когда можно использовать сохранённый префикс','на WinoGrande 33B и 65B',
      '0.068866','6 738 415 616'])expect(text).toContain(fact);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    expect(html).toContain('href="/sources/papers/llama/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-3\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/decoder-block\/"/);
  });
  it('renders the later Llama generation changes and keeps the unlicensed atlas crop out of the build', async () => {
    const route = 'textbook/modern-llm/llama-architecture/';
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="llama-generations"><\/a>[\s\S]*?(?=<span id="что-llama-не-изобрела")/)?.[0];
      expect(section).toBeDefined();
      for (const fact of ['id="llama31-long-context"', 'id="llama-base-chat-instruct"',
        'Llama 2 Chat', 'Llama 3 Instruct',
        'https://aclanthology.org/2025.acl-long.366.pdf#page=2', 'CC BY 4.0',
        `href="/${locale}sources/papers/prolong/"`]) expect(section).toContain(fact);
      expect(section).toContain(locale ? '8,030,261,248' : '8 030 261 248');
      expect(section).toContain(locale ? '16 GiB' : '16 ГиБ');
      expect(section).toContain(locale ? '8,590,000,128' : '8 590 000 128');
      expect(section).toContain('https://arxiv.org/html/2407.21783v3#S5.T21');
      expect(section).toContain(`href="/${locale}textbook/modern-llm/rope/#rope-llama31-scaling"`);
      const mix = section!.match(/<a id="llama31-long-short-mix"><\/a>[\s\S]*?(?=<a id="llama-base-chat-instruct")/)?.[0];
      expect(mix).toBeDefined();
      expect(mix).toContain('https://aclanthology.org/2025.acl-long.366.pdf#page=4');
      const mixturePlot = mix!.match(/<img[^>]+src="([^"]*\/gao25-figure2-data-mix\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(mixturePlot).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(mixturePlot!)))).digest('hex'))
        .toBe('5712d20be945e96fc55055409f50c8c069260df350944317e96434016d132955');
      expect(section).toContain('https://arxiv.org/html/2307.09288v2#A1.F24');
      const gqa = section!.match(/<a id="llama-gqa-cache-step"><\/a>[\s\S]*?(?=<a id="llama31-long-context")/)?.[0];
      expect(gqa).toBeDefined();
      for (const shape of ['[1, 1, 32, 128]', '[1, 1, 8, 128]', '[1, 5, 8, 128]', '[1, 5, 32, 128]'])
        expect(gqa).toContain(shape);
      expect(gqa).toContain('llama/model.py#L153-L190');
      const comparisonPlot = section!.match(/<img[^>]+src="([^"]*\/groups-speed\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(comparisonPlot).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(comparisonPlot!)))).digest('hex'))
        .toBe('69b9738c88e8ec7cca78f523baa624714f95b69dfadf3f6484ebe13e1e41da87');
      expect(section).toMatch(/(?:&lt;|&#x3C;)\|eot_id\|(?:>|&gt;)/);
      expect(section).not.toContain('katex-error');
      const figure = section!.match(/<img[^>]+src="([^"]*\/gao25-figure1-sft\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      const bytes = await readFile(join(distDir, decodeURIComponent(figure!)));
      expect(createHash('sha256').update(bytes).digest('hex'))
        .toBe('f4a36c325e98c704409ef2d625565710a524222afcfc641cb295ed8e2a1f0730');
      const atlas = await readFile(join(distDir, locale, 'models/semeystva/llama/index.html'), 'utf8');
      expect(atlas).toContain('mha-gqa-mqa.png');
      expect(atlas).toContain('/textbook/modern-llm/llama-architecture/#llama-gqa-cache-step');
      expect(atlas).toContain('https://arxiv.org/html/2407.21783v3#S2.F1');
      expect(atlas).not.toContain('llama3-architecture.png');
    }
    for (const base of ['assets', 'assets/00 Учебник/Assets']) {
      await expect(readFile(join(distDir, base, 'Figures/curated/atlas-courses-official/llama3-architecture.png')))
        .rejects.toMatchObject({code: 'ENOENT'});
    }
  });
  it('connects Natural Questions across GPT-2, GPT-3 and LLaMA in both reading routes', async () => {
    const llama = '/textbook/modern-llm/llama-architecture/';
    const models = '/textbook/models/';
    const subject = '#natural-questions-gpt-llama';
    for (const prefix of ['', '/en']) {
      const index = await readFile(join(distDir, prefix, 'textbook/index.html'), 'utf8');
      const intro = index.match(/<h2[^>]*id="(?:от-gpt-1-к-llama|from-gpt-1-to-llama)"[^>]*>[\s\S]*?(?=<h2\b)/)?.[0];
      expect(intro).toBeDefined();
      const paths = ['gpt-1/', 'gpt-2/', 'gpt-3/'].map(name => models + name);
      const order = [...paths, llama + '#gpt3-to-llama1']
        .map(path => intro!.indexOf(`href="${prefix}${path}"`));
      expect(order.every((position, i) => position >= 0 && (i === 0 || position > order[i - 1]))).toBe(true);
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      expect(reference).toContain(`href="${prefix}${llama}${subject}"`);
      const chapter = await readFile(join(distDir, prefix, llama, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="natural-questions-gpt-llama"><\/a>[\s\S]*?(?=<a id="gpt3-demo-order")/)?.[0];
      expect(section).toBeDefined();
      const scores = ['4,1%', '14,6%', '29,9%', '20,1%', '31,9%'].map(score => prefix ? score.replace(',', '.') : score);
      for (const value of [...scores, 'Frank Sinatra', 'Charles Darwin',
        'https://arxiv.org/pdf/2302.13971v1#page=17', 'https://creativecommons.org/licenses/by/4.0/'])
        expect(section).toContain(value);
      expect(section).toMatch(/<img[^>]+src="[^"]*\/llama1-figure3-natural-questions\.png"/);
    }
  });
  it('keeps the cross-generation tokenization example and its original figure reachable', async () => {
    const route = '/textbook/modern-llm/llama-architecture/';
    const anchor = '#gpt-llama-tokenization';
    for (const prefix of ['', '/en']) {
      for (const entry of ['textbook/index.html', 'reference/index.html']) {
        const index = await readFile(join(distDir, prefix, entry), 'utf8');
        expect(index).toContain(`href="${prefix}${route}${anchor}"`);
      }
      const chapter = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="gpt-llama-tokenization"><\/a>[\s\S]*?(?=<(?:a|h3)\b[^>]*id="размер-модели-и-объём-предобучения")/)?.[0];
      expect(section).toBeDefined();
      for (const [name, count] of [['GPT-1', 5], ['GPT-2 / GPT-3', 4], ['LLaMA v1', 8]]) {
        const row = section!.match(new RegExp(`<tr>\\s*<td[^>]*>${name}</td>[\\s\\S]*?</tr>`))?.[0];
        expect(row).toMatch(new RegExp(`<td[^>]*>${count}</td>`));
      }
      for (const value of ['3797', '6635', 'Jay Alammar', 'https://creativecommons.org/licenses/by-nc-sa/4.0/'])
        expect(section).toContain(value);
      expect(section).toMatch(/<img[^>]+src="[^"]*\/gpt2-token-embeddings-wte-2\.png"/);
    }
  });
  it('compares KV storage across the four generations and preserves the source illustration', async () => {
    for (const [locale, unit] of [
      ['', ['36 КиБ', '512 КиБ', '18 МиБ', '36 МиБ', '1 ГиБ']],
      ['en/', ['36 KiB', '512 KiB', '18 MiB', '36 MiB', '1 GiB']],
    ] as const) {
      const html=await readFile(join(distDir,locale,'textbook/index.html'),'utf8');
      expect(html).toContain('href="#gpt-decoder-route-cache"');
      const section=html.match(/<a id="gpt-decoder-route-cache"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-data")/)?.[0];
      expect(section).toBeDefined();
      const rows=section!.match(/<tr>[\s\S]*?<\/tr>/g)!.slice(1);
      expect(rows).toHaveLength(4);
      for (const [i, [model, perToken, full]] of [
        ['GPT-1',unit[0],unit[2]], ['GPT-2 Small',unit[0],unit[3]],
        ['GPT-3 6.7B',unit[1],unit[4]], ['LLaMA v1 7B',unit[1],unit[4]],
      ].entries()) {
        expect(rows[i]).toContain(model);
        expect(rows[i]).toContain(perToken);
        expect(rows[i]).toContain(full);
      }
      const figure=section!.match(/<img[^>]+src="([^"]*\/hf-kv-cache\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir,decodeURIComponent(figure!)))).digest('hex'))
        .toBe('44931e339df5ebc418385dcb1880a75da44eeeaf5c4ae3a6aab853d8916bd680');
      expect(section).toContain('https://huggingface.co/blog/continuous_batching#kv-cache');
      expect(section).toContain('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(section).not.toContain('katex-error');
    }
  });
  it('compares the four pretraining corpora without conflating corpus size, training tokens and context', async () => {
    for (const [locale, numbers] of [
      ['', ['40 ГБ', '0,44', '2,9', '300 млрд', '3,3']],
      ['en/', ['40 GB', '0.44', '2.9', '300 billion', '3.3']],
    ] as const) {
      const index = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const section = index.match(/<a id="gpt-decoder-route-data"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-end")/)?.[0];
      expect(section).toBeDefined();
      const rows = section!.match(/<tr>[\s\S]*?<\/tr>/g)!.slice(1);
      expect(rows).toHaveLength(4);
      for (const [row, route] of rows.map((row, i) => [row, [
        'models/gpt-1/', 'models/gpt-2/', 'models/gpt-3/', 'modern-llm/llama-architecture/',
      ][i]] as const)) {
        expect(row).toContain(`href="/${locale}textbook/${route}#`);
      }
      for (const term of ['BooksCorpus', 'WebText2', 'Common Crawl', 'CCNet', '67%', '15%', ...numbers]) {
        expect(section).toContain(term);
      }
      expect(section).toContain(`href="/${locale}textbook/modern-llm/llama-architecture/#training-and-inference-budget"`);
      expect(section).toContain('https://arxiv.org/pdf/2302.13971v1#page=3');
      const figure = section!.match(/<img[^>]+src="([^"]*\/llama1-training-loss\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      const bytes = await readFile(join(distDir, decodeURIComponent(figure!)));
      expect(createHash('sha256').update(bytes).digest('hex'))
        .toBe('bc8c87fb2b8ecb54dd235a0276ec1bc4c26ddc198d0804858525372e33fbaeb9');
    }
  });
  it('separates prompting, fixed-compute training, and inference constraints in the GPT-to-LLaMA comparison', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      expect(html).toContain('href="#gpt-decoder-route-training-budget"');
      const section = (from: string, to: string) => {
        const start = html.indexOf(`id="gpt-decoder-route-${from}"`);
        const end = html.indexOf(`id="gpt-decoder-route-${to}"`);
        expect(start).toBeGreaterThan(-1);
        expect(end).toBeGreaterThan(start);
        return html.slice(start, end);
      };
      const prompt = section('sentiment', 'architecture');
      for (const snippet of [
        'TL;DR:', 'ROUGE-2 F1', 'ANLI R3',
        ...(locale ? ['8.27', '4.03', '34.5%', '35.1%', '40.2%'] : ['8,27', '4,03', '34,5%', '35,1%', '40,2%']),
      ]) expect(prompt).toContain(snippet);
      const training = section('training-budget', 'repeated-data');
      for (const snippet of [
        'https://arxiv.org/html/2203.15556v1#S3.SS2',
        'https://arxiv.org/html/2404.10102v2#S2.F2',
        'https://creativecommons.org/licenses/by/4.0/',
        'llama1-downstream-training.png',
        ...(locale ? ['10 billion tokens', '5 billion', '74.5%', '77.4%'] : ['10 млрд', '5 млрд', '74,5%', '77,4%']),
      ]) expect(training).toContain(snippet);
      const image = training.match(/<img[^>]+src="([^"]*\/training_flop_vs_model_size\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(image).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(image!)))).digest('hex'))
        .toBe('3fbaefa50fa091f9dc58136ae9772a1d27123bf9399751bc8be0c919a2be7841');
    }
  });
  it('distinguishes measured loss from extrapolation in the GPT-to-LLaMA data comparison', async () => {
    for (const [locale, values] of [
      ['', ['8,7 млрд', '178 млрд', '44 млрд', '25%', '0,5%', 'прогноз']],
      ['en/', ['8.7 billion', '178 billion', '44 billion', '25%', '0.5%', 'predictions']],
    ] as const) {
      const html = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const section = html.match(/<a id="gpt-decoder-route-repeated-data"><\/a>[\s\S]*?(?=<a id="gpt-decoder-route-evidence")/)?.[0];
      expect(section).toBeDefined();
      for (const value of [...values, 'https://jmlr.org/papers/volume26/24-1000/24-1000.pdf#page=14', 'https://creativecommons.org/licenses/by/4.0/']) {
        expect(section).toContain(value);
      }
      const source = section!.match(/<img[^>]+src="([^"]*\/muennighoff-jmlr-figure10\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(source).toBeDefined();
      const bytes = await readFile(join(distDir, decodeURIComponent(source!)));
      expect(createHash('sha256').update(bytes).digest('hex'))
        .toBe('b12e4dcd113ffd6d8add76b69c37cedcda8548603bd3d44ffffdef0d745ca3de');
    }
  });
  it('shows which vocabulary and positional parameters changed across causal decoders', async () => {
    const route = 'textbook/modern-llm/llama-architecture/';
    const sourceRoute = 'sources/papers/tied-language-model-architectures/';
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const section = chapter.match(/<a id="gpt-llama-parameter-budget"><\/a>[\s\S]*?(?=<(?:a|h3)\b[^>]*id="где-задаётся-нужный-ответ")/)?.[0];
      expect(section).toBeDefined();
      const counts = ['7 510 272', '393 216', '38 597 376', '262 144 000'].map(count => locale ? count.replaceAll(' ', ',') : count);
      for (const value of [...counts,
        'https://aclanthology.org/D18-1323.pdf#page=2', 'https://creativecommons.org/licenses/by/4.0/'])
        expect(section).toContain(value);
      expect(section).toMatch(locale
        ? /<img[^>]+src="[^"]*\/gulordava-figure1-ab\.png"[^>]+alt="[^"]*separate input embedding/
        : /<img[^>]+src="[^"]*\/gulordava-figure1-ab\.png"[^>]+alt="[^"]*таблица/);
      expect(chapter).toContain(`href="/${locale}${sourceRoute}"`);
      const source = await readFile(join(distDir, locale, sourceRoute, 'index.html'), 'utf8');
      expect(source).toContain(`href="/${locale}${route}#gpt-llama-parameter-budget"`);
      expect(chapter).not.toContain('katex-error');
    }
  });
  it('connects the LLaMA training curve to the same-loss compute example',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/llama-architecture/index.html'),'utf8');
    const budget=html.match(/<a id="training-and-inference-budget"><\/a>[\s\S]*?(?=<h2\b)/)?.[0];
    expect(budget).toContain('580 млрд токенов');
    expect(budget).toContain('sardana24a-figure1.png');
    expect(budget).toContain('https://proceedings.mlr.press/v235/sardana24a.html');
    expect(budget).toMatch(/href="[^"]*#llama-pretraining-data"/);
    expect(html).toMatch(/href="[^"]*#training-and-inference-budget"/);
    expect(html).toContain('id="llama-pretraining-data"');
  });
  it('renders the original inference-size contour and indexes the compute comparison in both editions', async () => {
    for (const locale of ['', 'en']) {
      const route = 'textbook/modern-llm/llama-architecture/';
      const html = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const budget = html.match(/<a id="training-and-inference-budget"><\/a>[\s\S]*?(?=<h2\b)/)?.[0];
      expect(budget).toBeDefined();
      const image = budget!.match(/<img[^>]+src="([^"]*\/sardana24a-figure2b\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(image).toBeDefined();
      const bytes = await readFile(join(distDir, decodeURIComponent(image!)));
      expect(createHash('sha256').update(bytes).digest('hex'))
        .toBe('3d1bc7fc66b9f204fe83adcd6b2975f52794e9467f0388674d4091122fc3be99');
      expect(budget).toContain('sardana24a.pdf#page=3');
      expect(budget).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(budget).toContain('75%');
      expect(budget).toContain('50%');
      expect(budget).toContain('5.34');
      expect(budget).toContain('4.76');
      expect(budget).toContain('#joint-model-data-scaling');
      expect(budget).not.toContain('katex-error');
      const ref = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
      expect(ref).toContain(`href="/${locale ? 'en/' : ''}${route}#training-and-inference-budget"`);
    }
  });
  it('renders corrected LLaMA paper protocols and the actual original downstream plots',async()=>{
    const html=await readFile(join(distDir,'sources/papers/llama/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['27 февраля 2023','52.4','68.5','50.9','69.7',
      'С голосованием по 100','Gender score 70.6','Это не точность WinoGender',
      '0.1 для pass@1','0.8 для pass@100','77.5','4.1 и 5.9'])expect(text).toContain(fact);
    expect(html).toMatch(/<img[^>]+src="[^"]*\/llama1-downstream-training\.png"/);
    expect(html).toContain('href="/textbook/modern-llm/llama-architecture/"');
    expect(text).not.toContain('Gender bias: 70.6% верных');
    expect(html).not.toContain('katex-error');
  });
  it('renders T5 corrected training pairs, cache invalidation and six original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/models/t5/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['t5-figure1.png','t5-figure2-spans.png','t5-figure3.png',
      't5-figure4.png','t5-figure5-objectives.png','t5-figure6-repeated-data.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['[14, 9, 13, 1, -100, -100, -100]',
      'Что можно сохранить между запросами','EncoderDecoderCache.is_updated',
      'сбрасывают и self-KV decoder','Сам график показывает только training loss',
      '59.29 против 71.36','36 и 8 токенов','222 903 552'])expect(text).toContain(fact);
    expect(text).not.toContain('[14, 9, 12, 1, -100');
    expect(html).toContain('href="/sources/papers/t5/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/bert-roberta-deberta\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-1\/"/);
  });
  it('renders corrected T5 source protocols without fabricated architecture claims',async()=>{
    const html=await readFile(join(distDir,'sources/papers/t5/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['23 октября 2019','2023 года','SQuAD 1.1 exact match',
      '91.26','96.22','18 из 24','59.29','32.1','на 24 октября 2019',
      'не выделяет независимый вклад'])expect(text).toContain(fact);
    expect(text).not.toContain('Минимум 3 предложения, 5 слов');
    expect(html).toContain('href="/textbook/models/t5/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-3 source-faithful protocols, five licensed originals and native configuration data',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-3/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['gpt3-demonstrations-frame-171.png','10-gpt3-fine-tuning.gif',
      '07-gpt3-processing-transformer-blocks.gif','tokens-through-stack-frame-385.png',
      'context-window-frame-229.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['Повторное использование демонстраций','10 000 входных','2080 новых позиций',
      '72 пары','57','Перевод: направление и параллельные примеры','SacreBLEU',
      'Преобразования слов и новое значение','0.44%','Генерация новостей и оценка человеком',
      '49–54%','48–57%','фильтр оставил 2944 из 5153','На них accuracy составила 86.0%',
      'Остальные 2209 примеров с пересечениями имеют accuracy 86.9%'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/sources/papers/gpt-3-0/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-2\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/llama-architecture\/"/);
  });
  it('renders the GPT-3 / PET comparison, both original figures and the source-note backlink', async () => {
    for (const locale of ['', 'en']) {
      const prefix = locale ? '/en' : '';
      const page = await readFile(join(distDir, locale, 'textbook/models/gpt-3/index.html'), 'utf8');
      const start = page.indexOf('id="gpt3-few-shot-pet"');
      expect(start).toBeGreaterThan(-1);
      const section = page.slice(start, page.indexOf('<h2', start));
      for (const name of ['schick21-superglue-figure1', 'schick21-entailment-figure2'])
        expect(section).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name+'\\.png"[^>]+alt="[^"]+"'));
      for (const n of [1,2,6,8])
        expect(section).toContain('https://aclanthology.org/2021.naacl-main.185.pdf#page='+n);
      expect(section).toContain('0.288');
      expect(section).not.toContain('katex-error');
      expect(page).toContain(`href="${prefix}/sources/papers/few-shot-pet/"`);
    }
    const note = await readFile(join(distDir, 'sources/papers/few-shot-pet/index.html'), 'utf8');
    expect(note).toContain('href="/textbook/models/gpt-3/#gpt3-few-shot-pet"');
    const noteEn = await readFile(join(distDir, 'en/sources/papers/few-shot-pet/index.html'), 'utf8');
    expect(noteEn).toContain('href="/en/textbook/models/gpt-3/#gpt3-few-shot-pet"');
  });
  it('renders corrected GPT-3 paper-note comparisons without false historical priorities',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-3-0/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['31 автор','28 мая 2020','22 июля 2020','разница 5.7 пункта',
      '10.2 процентного пункта','2944 чистых примера с accuracy 86.0%',
      '2209 пересекающихся с accuracy 86.9%','82.8%','звёздочкой'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/textbook/models/gpt-3/"');
    expect(text).not.toContain('45 авторов');
    expect(text).not.toContain('впервые задачи задаются');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-2 cache contracts, the six originals and complete model examples',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-2/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'');
    for(const name of ['gpt2-sizes-hyperparameters-3.png','gpt2-input-embedding-positional-encoding-3.png',
      'gpt2-transformer-block-vectors-2.png','gpt2-output-projection-2.png',
      'gpt2-autoregression-2.gif','gpt2-self-attention-qkv-3-2.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['class TinyGPT2','семь позиций вместо четырёх','индекс 7 вместо 4',
      'Последний выбранный ID уже входит','18','874','368','124 439 808','two-shot'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/licenses/openai-gpt2-Modified-MIT.txt"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-1\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-3\/"/);
  });
  it('renders the GPT-2 translation example and its GPT-3 continuation in both editions', async () => {
    for (const locale of ['', 'en']) {
      const prefix = locale ? '/en' : '';
      const html = await readFile(join(distDir, locale, 'textbook/models/gpt-2/index.html'), 'utf8');
      const start = html.indexOf('id="gpt2-translation-prefix"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start, html.indexOf('id="gpt2-gpt3-coqa"', start));
      expect(section).toMatch(/<img[^>]+src="[^"]*liu22-figure3-translation-prefix\.png"[^>]+alt="[^"]+"/);
      expect(section).toContain('https://aclanthology.org/2022.deelio-1.10.pdf#page=12');
      expect(section).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(section).toContain('language_models_are_unsupervised_multitask_learners.pdf#page=3');
      expect(section).toContain('[406, 6, 25678, 559, 442, 12427, 13]');
      expect(section).toContain(`href="${prefix}/textbook/models/gpt-3/#`);
      expect(section).not.toContain('katex-error');
    }
  });
  it('renders the CoQA bridge and its attributed original in both editions', async () => {
    for (const locale of ['', 'en']) {
      const prefix = locale ? '/en' : '';
      const html = await readFile(join(distDir, locale, 'textbook/models/gpt-2/index.html'), 'utf8');
      const start = html.indexOf('id="gpt2-gpt3-coqa"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start, html.indexOf('</main>', start));
      expect(section).toMatch(/<img[^>]+src="[^"]*gpt-coqa-2026-11-11\/coqa-figure4-conversation-flow\.png"/);
      expect(section).toContain('https://aclanthology.org/Q19-1016.pdf#page=8');
      expect(section).toContain('https://transacl.org/ojs/index.php/tacl/article/view/1572');
      expect(section).toContain('https://arxiv.org/html/2005.14165v4#A8.T1');
      expect(section).toContain(`href="${prefix}/textbook/models/gpt-3/#gpt3-relation-shots"`);
      const rows = [...section.matchAll(/<tr>(.*?)<\/tr>/gs)].map(([, row]) =>
        [...row.matchAll(/<td[^>]*>(.*?)<\/td>/gs)].map(([, cell]) => cell.replace(',', '.')));
      expect(rows).toContainEqual(['GPT-3 350M', '55.0', '52.1', '52.0']);
      expect(rows).toContainEqual(['GPT-3 175B', '81.5', '84.0', '85.0']);
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders the LAMBADA comparison with the source figure and local GPT-3 destination in both editions', async () => {
    for (const locale of ['', 'en']) {
      const prefix = locale ? '/en' : '';
      const readModel = (model: string) =>
        readFile(join(distDir, locale, `textbook/models/${model}/index.html`), 'utf8');
      const html = await readModel('gpt-2');
      const start = html.indexOf('id="gpt2-gpt3-lambada"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start, html.indexOf('</main>', start));
      expect(section).toMatch(/<img[^>]+src="[^"]*paperno-lambada-figure2a\.png"[^>]+alt="[^"]+"/);
      expect(section).toContain('https://aclanthology.org/P16-1144.pdf#page=6');
      expect(section).toContain('https://aclanthology.org/faq/copyright/');
      const normalized = section.replace(/(\d),(\d)/g, '$1.$2');
      for (const score of ['52.66%', '63.24%', '76.2%', '86.4%'])
        expect(normalized).toContain(score);
      expect(section).toContain('very old bicycle');
      const destination = [...section.matchAll(/href="([^"]+)"/g)]
        .map(([, href]) => href)
        .find(href => href === `${prefix}/textbook/models/gpt-3/#lambada-gpt3-fill-blank`);
      expect(destination).toBeDefined();
      const anchor = decodeURIComponent(destination!.split('#')[1]);
      const gpt3 = await readModel('gpt-3');
      expect(gpt3).toContain(`id="${anchor}"`);
      expect(gpt3).toContain(`href="${prefix}/textbook/models/gpt-2/#gpt2-gpt3-lambada"`);
    }
  });
  it('connects the illustrated GPT chronology and both reference indices to the owner sections', async () => {
    const sections = [
      ['gpt-2', 'gpt2-gpt3-lambada'],
      ['gpt-3', 'lambada-gpt3-fill-blank'],
      ['gpt-3', 'gpt-llama-completion-scoring'],
    ];
    for (const locale of ['', 'en']) {
      const prefix = locale ? '/en' : '';
      const contents = await readFile(join(distDir, locale, 'textbook/index.html'), 'utf8');
      const reference = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
      const examples = contents.slice(contents.indexOf('id="gpt-decoder-route-evidence"'),
        contents.indexOf('id="gpt-decoder-route-prompt-order"'));
      for (const [model, id] of sections) {
        const path = `${prefix}/textbook/models/${model}/#${id}`;
        expect(examples).toContain(`href="${path}"`);
        expect(reference).toContain(`href="${path}"`);
        const owner = await readFile(join(distDir, locale, `textbook/models/${model}/index.html`), 'utf8');
        expect(owner.match(new RegExp(`id="${id}"`, 'g'))).toHaveLength(1);
      }
      for (const [model, id] of [
        ['gpt-2', 'gpt2-gpt3-coqa'], ['gpt-3', 'gpt3-surface-form-competition'],
      ]) expect(reference).toContain(`href="${prefix}/textbook/models/${model}/#${id}"`);
    }
  });
  it('renders corrected GPT-2 paper-note protocols and released model sizes',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-2-0/index.html'),'utf8');
    for(const fact of ['124M','355M','774M','52.66%','93.30%','18.34','42.16',
      '29.34/8.27/26.58','демонстрации в контексте','93.45','93.30'])
      expect(html).toContain(fact);
    expect(html).toContain('href="/textbook/models/gpt-2/"');
    expect(html).not.toContain('ROUGE-L ~15');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-1 gradients, the full candidate example and licensed source figures',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-1/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'');
    const en=await readFile(join(distDir,'en/textbook/models/gpt-1/index.html'),'utf8');
    for(const page of [html,en]){
      expect(page).toMatch(/<img[^>]+src="[^"]*\/bert-gpt-elmo-figure3\.png"/);
      expect(page).toMatch(/<img[^>]+src="[^"]*\/d2l-gpt-decoder\.png"/);
      expect(page).toMatch(/<img[^>]+src="[^"]*\/cs224n-decoder-classifier\.png"[^>]+alt="[^"]+"/);
      expect(page).toContain('cs224n-2023-lecture9-pretraining.pdf#page=41');
      expect(page).toContain('cs224n-2023-lecture9-pretraining.pdf#page=44');
      expect(page).toContain('cs224n.1234/#logistics');
      expect(page).not.toMatch(/<img[^>]+src="[^"]*\/gpt1-figure[12]/);
      expect(page).toContain('https://aclanthology.org/N19-1423.pdf#page=13');
      expect(page).toContain('language_understanding_paper.pdf#page=4');
      expect(page.match(/language_understanding_paper\.pdf#page=7/g)).toHaveLength(2);
      expect(page).not.toContain('katex-error');
    }
    expect(html).toContain('href="/textbook/models/gpt-2/#gpt2-relation-prefix"');
    expect(en).toContain('href="/en/textbook/models/gpt-2/#gpt2-relation-prefix"');
    for(const fact of ['head.grad','scores.retain_grad','token_loss.retain_grad',
      '0.020833','0.718351','2.085067','Неверный кандидат тоже участвует в LM-обучении',
      'Скрытое состояние заново вычисляется'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/licenses/openai-gpt1-MIT.txt"');
    expect(html.match(/<pre/g)).toHaveLength(5);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/t5\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-2\/"/);
  });
  it('renders the corrected GPT-1 paper note and its link to the full chapter',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-1-0/index.html'),'utf8');
    expect(html).toContain('href="/textbook/models/gpt-1/"');
    for(const fact of ['не было изобретено','85.8%','59.9','75.0','69.1',
      '56.0 против 61.7','82.3 F1 против 86.0','few-shot с демонстрациями'])
      expect(html).toContain(fact);
    expect(html).not.toContain('katex-error');
  });
  it('renders BERT training, both source excerpts and eleven original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/models/bert-roberta-deberta/index.html'),'utf8');
    for(const name of ['bert-base-bert-large-encoders.png','bert-figure1.png','bert-figure2.png',
      'BERT-language-modeling-masked-lm.png','bert-next-sentence-prediction.png',
      'bert-figure4.png','bert-feature-extraction-contextualized-embeddings.png',
      'spanbert-boundary-objective.png','spanbert-span-lengths.png',
      'debertav3-figure1-sharing.png','debertav3-figure2-mlm-loss.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const id of ['архитектура-и-формы-тензоров','mlm-как-получить-двунаправленный-обучающий-сигнал',
      'вторая-цель-bert-отношения-между-сегментами','данные-и-рецепт-bert',
      'один-encoder--разные-способы-чтения-выхода','как-читать-результаты-bert',
      'roberta-когда-новый-результат-даёт-не-новая-архитектура',
      'deberta-разделить-содержание-и-относительное-положение',
      'ограничения-и-наследие','источники-и-дальнейшее-чтение'])
      expect(html).toContain('id="'+id+'"');
    for(const fact of ['TinyBertPretraining','emd_context_layer','disentangled',
      'mlm_input_grad','0.118682','first read','second read','2.390310','89.875',
      '3.674235','Microsoft Corporation','microsoft-deberta-MIT.txt'])
      expect(html).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(12);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/transformer\/encoder-decoder-patterns\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/t5\/"/);
  });
  it('renders grouped masking, SOP, RTD and GDES in both BERT locales', async()=>{
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir,prefix+'textbook/models/bert-roberta-deberta/index.html'),'utf8');
      for (const anchor of ['whole-word-masking','spanbert-boundary-objective',
        'spanbert-span-sampling','sentence-order-prediction','electra-replaced-token-detection',
        'debertav3-gradient-disentangling'])
        expect(html).toContain('id="'+anchor+'"');
      for (const fact of ['choose_word_targets','direct_sbo_gradient','same_offset_logits',
        'clamped_geometric_mean','assemble_sop','next_sentence_labels',
        'google-bert-eedf5716-Apache-2.0.txt','google-albert-b772393d-Apache-2.0.txt',
        'debertav3-figure1-sharing.png','debertav3-figure2-mlm-loss.png',
        'https://arxiv.org/pdf/2111.09543v4#page=5','https://arxiv.org/pdf/2111.09543v4#page=6',
        'grad_g_from_rtd'])
        expect(html).toContain(fact);
      expect(html.match(/<pre/g)).toHaveLength(12);
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders corrected RoBERTa and DeBERTa paper notes with links to the full chapter', async()=>{
    const deberta=await readFile(join(distDir,'sources/papers/deberta/index.html'),'utf8');
    const roberta=await readFile(join(distDir,'sources/papers/roberta/index.html'),'utf8');
    for(const html of [deberta,roberta])
      expect(html).toContain('href="/textbook/models/bert-roberta-deberta/"');
    for(const fact of ['81.3/78.0','91.8/85.8','1.3 пункта','DeBERTa-MT','89.875'])
      expect(deberta).toContain(fact);
    for(const fact of ['78.3/78.7','84.3/84.0','92.5/92.9','начальному опыту на 100K шагов'])
      expect(roberta).toContain(fact);
  });
  it('renders six architecture originals, all four examples and 27 previous anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/encoder-decoder-patterns/index.html'),'utf8');
    for (const name of ['t5-figure1.png','t5-figure2.png','t5-figure3.png',
      't5-figure4.png','bert-figure4.png','BERT-language-modeling-masked-lm.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['1.832581','0.880797','11.125','12.25','corrupt_spans',
      'full after append','new causal query','cross_attention_cache',
      'decoder_ids','two_layers','CC BY 4.0','CC BY-NC-SA 4.0'])
      expect(html).toContain(fact);
    for (const id of ["один-пример-на-всю-главу","главный-переключатель-матрица-видимости","encoder-only-построить-контекстное-представление","как-bert-учится-без-готовой-разметки","что-выходит-из-encoder","почему-encoder-неудобен-для-обычной-генерации","decoder-only-предсказать-следующий-токен","почему-из-next-token-prediction-получается-general-purpose-модель","минимальный-causal-attention-в-коде","encoderdecoder-прочитать-один-текст-написать-другой","один-шаг-перевода","teacher-forcing","t5-все-задачи-как-text-to-text","почему-слово-decoder-сбивает-с-толку","три-маски-можно-объединить-в-одной-модели","что-выбрать-на-практике","сравнение-без-рекламных-упрощений","ошибки-которые-стоит-научиться-замечать","практика-без-большого-gpu","1-нарисовать-маски","2-сравнить-representations","3-прочитать-настоящий-код","контроль-понимания","литература-и-дополнительные-материалы","основные-papers","лучшие-объяснения-и-код","связанные-заметки-базы"]) expect(html).toContain('id="'+id+'"');
    expect(html.match(/<pre/g)).toHaveLength(8);
    expect(html).not.toContain('katex-error');
  });
  it('routes the architecture introduction back to Transformer and forward to BERT', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/encoder-decoder-patterns/index.html'),'utf8');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/transformer\/full-transformer\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/bert-roberta-deberta\/"/);
  });
  it('renders both complete Transformer models, seven originals and preserved headings', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/full-transformer/index.html'),'utf8');
    for (const name of ['Transformer_encoder.png','The_transformer_encoder_decoder_stack.png',
      'transformer_resideual_layer_norm_2.png','transformer_resideual_layer_norm_3.png',
      'transformer_decoder_output_softmax.png','transformer_decoding_2.gif',
      'vaswani-transformer-figure1.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['TinyTranslation','TinyCausalLM','EncoderBlock','DecoderBlock',
      '0.349012','0.549012','0.365148','SublayerConnection','CC BY-NC-SA 4.0',
      '9.786662','12.110340','9.352585','8.497538','7.482068',
      'per-target loss','shared PAD row gradient norm','A Transformer block contains two'])
      expect(html).toContain(fact);
    for (const id of ['1-главная-декомпозиция','2-остаточный-поток','3-feed-forward-network',
      '4-информация-о-позиции','5-оригинальный-encoder-block','6-оригинальный-decoder-block',
      '7-обучение-и-генерация','8-три-архитектурные-ветви','encoder-only-bert',
      'decoder-only-gpt','modern-decoder-only-llama','9-архитектурный-diff',
      '10-архитектурное-решение-и-эмпирический-рецепт--не-одно-и-то-же',
      '11-почему-transformer-вытеснил-рекуррентную-основу',
      '12-минимальный-современный-causal-block','13-как-читать-реальный-код',
      '14-проверка-форм-тензоров','краткие-итоги','источники-и-интерактивы',
      'первичные','код-и-объяснения','architecture-lineage','architecture-evidence-boundary',
      'transformer-original-figure'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
    for (const locale of ['', 'en/']) {
      const page = await readFile(join(distDir,locale+'textbook/transformer/full-transformer/index.html'),'utf8');
      expect(page).toMatch(/<img[^>]+src="[^"]*\/vaswani-transformer-figure1\.png"/);
      expect(page).not.toMatch(/<img[^>]+src="[^"]*\/gpt-2-transformer-xl-bert-3\.png"/);
      expect(page).toContain('href="/'+locale+'textbook/models/gpt-1/#gpt1-decoder-architecture"');
      expect(page).toContain('href="/'+locale+'textbook/transformer/from-seq2seq-to-transformer/#translation-two-reads"');
      expect(page).toContain('href="#transformer-original-figure"');
    }
  });
  it('renders complete positional mechanisms, six source figures and stable section anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/positional-information/index.html'),'utf8');
    for (const name of ['transformer_positional_encoding_vectors.png',
      'd2l-sinusoidal-curves.svg','d2l-positional-heatmap.svg',
      'shaw-relative-figure1.png','roformer-figure1.png','alibi-original.jpeg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['0.804430','18.517688','0.576117','32.344406','0.960170',
      't5_bucket','apply_rope','alibi_slopes','reset-position error',
      '3.061223','1.580556','incremental RoPE outputs','double-rotated cache output',
      'CC BY-NC-SA 4.0','CC BY-SA 4.0','CC BY-NC-ND 4.0'])
      expect(html).toContain(fact);
    for (const id of ['отправная-точка-где-position-входит-в-attention',
      'абсолютные-позиции-адрес-каждой-строки','обучаемая-таблица','синусоидальное-кодирование',
      'относительные-представления-shaw-расстояние-становится-частью-связи',
      't5-relative-position-bias-один-скаляр-на-bucket-и-голову',
      'rope-позиция-как-поворот-query-и-key','маленький-численный-пример-rope',
      'alibi-линейный-штраф-за-дальность','один-пример-шесть-разных-вмешательств',
      'сравнение-методов','экстраполяция-что-именно-следует-проверять',
      'практические-ошибки-которые-не-видны-по-shapes','краткие-итоги',
      'источники-и-дальнейшее-чтение','rope-relative-position-derivation',
      'rope-и-kv-cache-позиция-сохраняется-вместе-с-ключом'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
  });
  it('renders device-placement semantics beside the original CUDA stream figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-cuda-memory-hierarchy/index.html'), 'utf8');
    expect(html).toContain('id="torch-device-placement"');
    for (const text of ['source_cpu', 'on_device', 'explicit_cpu', 'same_cpu',
      'independent_cpu', 'CUDA transfer executed', 'CPU-ветка',
      'nvidia-same-stream.png', 'nvidia-two-streams.png']) expect(html).toContain(text);
    expect(html).toContain('https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html');
    const chapter = await readFile(join(rootDir, '00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md'), 'utf8');
    const meta = YAML.parse(chapter.split('---')[1]);
    expect(html).toContain('datetime="' + new Date(meta.last_updated).toISOString() + '"');
    expect(html).not.toContain('class="katex-error');
  });

  it('renders masking/head examples, source figures and existing chapter anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/masking-heads-shapes/index.html'),'utf8');
    for (const name of ['causal-mask-detailed.png', 'vaswani-multihead-figure2.png',
      'voita-trained-heads-figure1.png',
      'transformer_attention_heads_qkv.png','transformer_attention_heads_z.png',
      'transformer_attention_heads_weight_matrix_o.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['12.689414','11.546979','0.909969','32','-6',
      'absolute-position mask', 'upper-left causal mask','two-query absolute mask',
      'two-query outputs','block_unmasked','causal_lower_right','masked_attention',
      'MultiHeadSelfAttention','loss.backward','CC BY 4.0','CC BY-NC-SA 4.0'])
      expect(html).toContain(fact);
    for (const namedAxisExample of ['named_scores', 'norm_sq',
      'scores of first query', 'squared row norms',
      'https://einops.rocks/api/rearrange/', 'https://einops.rocks/api/einsum/',
      'https://einops.rocks/api/reduce/']) expect(html).toContain(namedAxisExample);
    for (const id of ['маска-меняет-множество-доступных-ключей',
      'причинная-маска-не-подсматривать-ответ','padding-mask-не-читать-заполнители',
      'зачем-одной-позиции-несколько-голов','split-и-merge-одна-операция-две-записи',
      'полная-трассировка-форм','broadcasting-масок','ошибки-которые-выглядят-правдоподобно',
      'минимальная-проверка-реализации','источники-и-визуальные-продолжения',
      'named-attention-shapes','einops-attention-shapes','shape-evidence',
      'multihead-source-figure','two-heads-trace','measuring-head-specialization'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
    for (const locale of ['', 'en/']) {
      const page = await readFile(join(distDir,locale+'textbook/transformer/masking-heads-shapes/index.html'),'utf8');
      expect(page.match(/<pre data-language="python"/g)).toHaveLength(6);
      expect(page.match(/<img[^>]+src="\/assets\//g)).toHaveLength(6);
      expect(page).not.toMatch(/<img[^>]+src="[^"]*\/qkv-for-heads\.png"/);
      expect(page).toContain('href="https://arxiv.org/html/1706.03762v7#S3.F2.g2"');
      expect(page).toContain('href="https://aclanthology.org/P19-1580.pdf#page=3"');
      expect(page).toContain('id="two-heads-trace"');
      expect(page).toContain('id="measuring-head-specialization"');
      expect(page).toContain('causal=True');
      expect(page).not.toContain('katex-error');
    }
  });
  it('renders projected self-attention, complete examples and four original figures', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/self-attention/index.html'),'utf8');
    for (const name of ['transformer_self_attention_vectors.png',
      'self-attention-matrix-calculation-2.png', 'self-attention-output.png',
      'cnn-rnn-self-attention.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['2.046556','1.838140','0.196612','1.758725',
      '2.093112','0.436612','0.718517','0.058699',
      'one SGD step', 'torch.testing', 'CC BY-NC-SA 4.0', 'CC BY-SA 4.0'])
      expect(html).toContain(fact);
    for (const id of ['зачем-слову-смотреть-на-другие-слова',
      '1-самый-простой-канал-среднее-прошлого',
      '2-сделаем-веса-зависимыми-от-содержимого','3-почему-нужен-отдельный-value',
      'всё-вычисление-на-двух-рисунках','4-численный-пример-который-можно-проверить-вручную',
      'шаг-a-dot-products','шаг-b-scaling','шаг-c-softmax','шаг-d-weighted-sum-values',
      '5-зачем-делить-на-dksqrtd_kdk','6-минимальная-реализация-одной-головы',
      'шаг-a-скалярные-произведения','шаг-b-масштабирование',
      'шаг-d-взвешенная-сумма-значений',
      'обратный-проход-на-том-же-примере','код-прямого-и-обратного-прохода',
      'что-легко-перепутать','different-weights-the-same-output',
      'лаборатория-по-карпати','источники'])
      expect(html).toContain('id="'+id+'"');
    expect(html).toContain('different weights:');
    expect(html).toContain('same output:');
    expect(html).toContain('https://arxiv.org/html/1908.04211v3#S3');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
  });

  it('renders recurrent-attention mechanisms, full examples and all eight source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/transformer/from-seq2seq-to-transformer/index.html'), 'utf8');
    for (const name of ['d2l-seq2seq-state.svg', 'd2l-bahdanau-pytorch-weights.svg',
      'd2l-attention-output.svg', 'distill-rnn_read.svg', 'seq2seq-attention.svg',
      'luong-global.png', 'luong-local.png', 'luong-input-feeding.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const fact of ['1.955285', '0.000560', '0.969', 'AdditiveAttention',
      '192', '768', 'detach()', 'EMNLP 2015',
      'AttentiveSeq2Seq', 'context_normalized', 'input feeding', 'CC BY-NC-SA 3.0',
      '<strong>0.726</strong>', '<strong>0.673</strong>', '#seq2seq-attention',
      'https://arxiv.org/html/1409.0473v7#S4.F3'])
      expect(html).toContain(fact);
    expect(html).not.toMatch(/<img[^>]+src="[^"]*\/(?:bottleneck-min|bahdanau_examples-min)\.png"/);
    for (const id of ['перевод-как-условная-языковая-модель',
      '1-до-attention-один-вектор-на-всё-предложение', '2-оставим-не-резюме-а-память',
      '2-позиционные-представления-и-чтение-памяти',
      '2-представления-исходных-позиций-и-чтение-памяти',
      'translation-two-reads', '3-проследим-один-перевод', '4-bahdanau-и-luong--похожий-принцип-разные-сборки',
      '5-что-именно-улучшил-attention', '6-почему-этого-всё-ещё-недостаточно',
      '7-три-термина-которые-нельзя-смешивать', 'мини-реализация-recurrent-attention',
      'краткие-итоги', 'источники-и-хорошие-продолжения', 'как-получить-веса',
      'динамический-доступ', 'короткий-путь-для-информации-и-градиента',
      'переменная-длина-памяти', 'наблюдаемый-alignment',
      'два-объяснения-которые-стоит-прочитать-в-оригинале', 'первичные', 'объяснения'])
      expect(html).toContain('id="' + id + '"');
    expect(html).not.toContain('katex-error');
  });

  it('renders seq2seq bridge and beam-state examples with original diagrams in both locales',async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-03-seq2seq-state-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^    anchor: (seq2seq-[a-z-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(23);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/recurrent-networks/seq2seq-bottleneck/index.html'),'utf8');
      for(const id of anchors)expect(html.split('id="'+id+'"')).toHaveLength(2);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
      expect(html.match(/<img\b[^>]+\bsrc="\/assets\//g)).toHaveLength(9);
      for(const phrase of ['state_bridge','selected_parents','child_states',
        'initial_decoder','original English excerpt','/d2l-encoder-decoder.svg',
        'fixed number of source coordinates','CC BY-SA 4.0',
        '/cho-2014-figure4.png','https://aclanthology.org/W14-4012.pdf#page=7',
        '#translation-two-reads'])
        expect(html).toContain(phrase);
      expect(html).not.toContain('class="katex-error');
      expect(html).toContain('/textbook/transformer/from-seq2seq-to-transformer/');
      if(locale)expect(html).toContain('Canonical-language chapter.');
    }
  });

  it('renders the complete seq2seq examples, seven source SVGs and preserved sections', async () => {
    const html = await readFile(join(distDir, 'textbook/recurrent-networks/seq2seq-bottleneck/index.html'), 'utf8');
    expect(html).toContain('0.000583');
    expect(html).toContain('pack_padded_sequence');
    expect(html).toContain('enumerate_finished');
    expect(html).toContain('exposure bias');
    expect(html).not.toContain('katex-error');
    for (const id of [
      'условная-языковая-модель', 'формы-на-одном-примере',
      'почему-фиксированный-контекст-становится-узким-местом',
      'декодирование--не-обучение', 'мост-к-attention',
      'ограничения-и-частые-смешения', 'источники-и-дальнейшее-чтение'
    ]) expect(html).toContain('id="' + id + '"');
    for (const name of ['seq2seq', 'seq2seq-details', 'seq2seq-predict',
      's2s-prob1', 's2s-prob2', 'beam-search', 'seq2seq-attention']) {
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name + '\\.svg"'));
    }
  });

  it('renders activation mechanisms, faithful source figures and the output-loss exception', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/activation-functions/index.html'), 'utf8');
    for (const figure of ['d2l-sigmoid.svg', 'd2l-sigmoid-gradient.svg', 'd2l-tanh.svg',
      'd2l-tanh-gradient.svg', 'd2l-relu.svg', 'd2l-relu-gradient.svg',
      'gelu-figure-1.png', 'cs336-silu-vs-relu.png'])
      expect(html.includes('/' + figure), 'missing original activation figure: ' + figure).toBe(true);
    for (const anchor of ['sigmoid', 'tanh', 'relu', 'elu', 'gelu', 'silu--swish',
      'сравнение-по-поведению', 'активация-и-инициализация-связаны',
      'где-активация-не-нужна', 'краткие-итоги', 'источники'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable activation anchor: ' + anchor).toBe(true);
    for (const fact of ['binary_cross_entropy_with_logits', '-0.993307', '0{,}006648',
      'Leaky ReLU', '0{,}6875', 'SwiGLU', '2726505c4767eed99dfd89ce0f836fbca44d6a6d',
      'Левый и правый наклоны', 'В минимуме', 'производная по', '0.655293',
      'Замена ReLU в одном нейроне MLP', '0,88888', '0,00383'])
      expect(html.includes(fact), 'missing activation explanation: ' + fact).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    expect((html.match(/class="katex-display"/g) ?? []).length).toBeGreaterThanOrEqual(20);
    const license = await readFile(join(distDir, 'licenses/gelus-MIT.txt'), 'utf8');
    expect(license.includes('Copyright (c) 2016 hendrycks')).toBe(true);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html).toContain('Maxout: максимум между обучаемыми ветвями');
    expect(html).toContain('goodfellow13.pdf#page=2');
    expect(html).toContain('кусочно-линейных');
    const anchors=["activation-chain-rule","sigmoid-saturation","sigmoid-bce-logits","tanh-centering","relu-inactive-leaky","elu-slopes","gelu-exact-approximation","silu-swish","swiglu-branch-gradients","maxout-affine-branches","activation-moments","activation-output-logits"];
    for(const anchor of anchors){
      expect(html).toContain('id="'+anchor+'"');
      for(const locale of ['','en/']){
        const reference=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
        expect(reference).toContain('href="/'+locale+'textbook/neural-networks/activation-functions/#'+anchor+'"');
      }
    }
    const english=await readFile(join(distDir,'en/textbook/neural-networks/activation-functions/index.html'),'utf8');
    expect(english).toContain('Canonical-language chapter.');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(english).not.toContain('katex-error');
  });
  it('renders the original Maxout surface, partition and worked example', async () => {
    const chapter = 'textbook/neural-networks/activation-functions/';
    const paper = 'sources/papers/piecewise-linear-regions/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, chapter, 'index.html'), 'utf8');
      const region = html.match(/<a id="maxout-input-regions"><\/a>[\s\S]*?(?=<h3\b[^>]*id="размерности-и-число-параметров")/)?.[0];
      expect(region).toBeDefined();
      expect(region).toMatch(/<img[^>]+src="[^"]*\/goujon-maxout-figure6\.png"[^>]+alt="[^"]*грани/);
      expect(region).toContain('https://bigwww.epfl.ch/publications/goujon2401.pdf#page=7');
      expect(region).toContain('CC BY 4.0');
      expect(region).toContain('Пересечение полуплоскостей');
      expect(html).toMatch(/<details>\s*<summary>Разобрать Maxout:[^<]+<\/summary>[\s\S]*?id="maxout-input-regions"[\s\S]*?<\/details>/);
      expect(region).not.toContain('katex-error');
      expect(html).toContain('href="/'+prefix+paper+'"');
      const source = await readFile(join(distDir, prefix, paper, 'index.html'), 'utf8');
      expect(source).toContain('href="/'+prefix+chapter+'#maxout-input-regions"');
    }
  });
  it('renders the reviewed MLP calculation, original figures and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/mlp/index.html'), 'utf8');
    for (const figure of ['cs231n-mlp-layered.jpeg', 'cs231n-depth-width.jpeg',
      'cs231n-decision-regions.jpeg', 'moon-mlp.png'])
      expect(html.includes('/' + figure), 'missing MLP source figure: ' + figure).toBe(true);
    for (const anchor of ['один-нейрон', 'слой-считает-много-нейронов-одновременно',
      'зачем-нужна-нелинейность', 'xor-как-минимальный-контрпример-линейной-модели',
      'что-именно-обучается', 'ширина-глубина-и-число-параметров',
      'mlp-внутри-transformer', 'краткие-итоги', 'источники'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable MLP anchor: ' + anchor).toBe(true);
    for (const fact of ['0{,}126928', '0{,}098018', '-0{,}298007', 'optimizer.step',
      '-0{,}023841', '1{,}002384', 'weight.copy_', 'Глубина одинакова', 'Positionwise Feed-Forward Networks',
      'grad', 'hinge loss', 'MIT License'])
      expect(html.includes(fact), 'missing worked MLP explanation: ' + fact).toBe(true);
    for (const mechanism of ['model-state-parameters-buffers-and-submodules',
      'registration-does-not-depend-on-a-gradient', 'a-list-of-layers-must-register-its-contents',
      'saving-values-and-rebuilding-the-computation', 'evaluation-mode-and-gradient-recording',
      'CustomLayer', 'ModuleList', 'probe_optimizer', 'checkpoint_bytes', 'weights_only',
      'does not encode the ReLU', 'does not disable gradient recording',
      '/licenses/pytorch-v2.14.1-LICENSE.txt'])
      expect(html.replace(/\s+/g, ' ').includes(mechanism), 'missing module-state explanation: ' + mechanism).toBe(true);
    expect(html.includes('Слева показан нейрон как вычислительный узел')).toBe(false);
    expect(html.indexOf('id="mlp-width-depth-parameters"')).toBeLessThan(html.indexOf('id="mlp-positionwise-ffn"'));
    expect(html.indexOf('id="mlp-positionwise-ffn"')).toBeLessThan(html.indexOf('id="pytorch-module-state"'));
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
    expect(html).toContain('/licenses/cs231n-MIT.txt');
    const anchors=['neuron-affine-activation','mlp-batch-shapes','mlp-xor','mlp-training-step',
      'mlp-width-depth-parameters','mlp-positionwise-ffn','pytorch-module-state','pytorch-module-hooks'];
    for(const anchor of anchors){
      expect(html).toContain('id="'+anchor+'"');
      for(const locale of ['','en/']){
        const reference=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
        expect(reference).toContain('href="/'+locale+'textbook/neural-networks/mlp/#'+anchor+'"');
      }
    }
    const english=await readFile(join(distDir,'en/textbook/neural-networks/mlp/index.html'),'utf8');
    expect(english).toContain('Canonical-language chapter.');
    expect(english.match(/<pre data-language="python"/g)).toHaveLength(11);
    expect(english).not.toContain('katex-error');
    expect(html.includes('Сети различной глубины и ширины')).toBe(false);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    expect((html.match(/class="katex-display"/g) ?? []).length).toBeGreaterThanOrEqual(15);
  });
  it('renders complete module hooks and distinct input/output/weight derivatives in both locales',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/mlp/index.html'),'utf8');
      for(const fact of ['observing-a-layer-forward-and-backward-hooks','one-layer-three-different-derivatives',
        'when-the-backward-hook-fires','inspect_linear','capture_backward','input_gradient',
        'output_gradient','no_grad_records','bypass_records','13.5','CC BY-NC-SA 4.0'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L3871');
      expect(html).toMatch(/<details>\s*<summary>Открыть приложение на английском<\/summary>[\s\S]*?id="pytorch-module-state"[\s\S]*?id="pytorch-module-hooks"[\s\S]*?<\/details>/);
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });
  it('renders formal mathematics mechanisms, source figures and checked proof examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/formal-math/index.html'), 'utf8');
    for (const figure of ['leanstar-pipeline.png', 'leanstar-search-sampling.png',
      'm09-p97.png', 'm10-p78.png', 'm11-p37.png'])
      expect(html.includes('/' + figure), 'missing original formal-math figure: ' + figure).toBe(true);
    for (const mechanism of ['swap_and', 'larger_exists', 'no_largest',
      'SMT_VERIF', 'associated_of_dvd_dvd', 'Chain-of-States',
      'compile_one_instr', 'sample_with_continuation', 'insufficient_stack',
      'Coq.Lists.List', 'app_nil_r', '46,3%', '31,65%', 'CC BY 4.0'])
      expect(html.includes(mechanism), 'missing formal-math explanation: ' + mechanism).toBe(true);
    for (const anchor of ['один-шаг-lean-proof-state-tactic-kernel-check',
      'три-разрыва-theorem-statement-proof-steps-and-semantic-equivalence',
      'lean-star-thought-before-tactic-and-expert-iteration',
      'copra-stateful-backtracking-with-prover-feedback', 'воспроизводимый-research-workflow'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('renders AlphaProof compute scaling and both Lean proof checks in the canonical editions', async () => {
    const route = '/textbook/agents/formal-math/';
    for (const prefix of ['', '/en']) {
      const page = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      for (const id of ['alphaproof-search-vs-ttrl', 'lean-proof-acceptance']) {
        expect(page).toContain(`id="${id}"`);
        expect(reference).toContain(`href="${prefix}${route}#${id}"`);
      }
      for (const datum of ['33,1', '43,7%', '53,9%', '58,3%', '4000', 'Given.Goal', 'Candidate.easy', 'sorryAx', 'type mismatch'])
        expect(page).toContain(datum);
      expect(page).toContain('https://www.nature.com/articles/s41586-025-09833-y/figures/4');
      expect(page).toMatch(/<img[^>]+src="[^"]*\/figure-4-search-vs-ttrl\.png"/);
      const map = await readFile(join(distDir, prefix, 'textbook/agents/module-map/index.html'), 'utf8');
      expect(map).toContain(`href="${prefix}${route}#lean-proof-acceptance"`);
      expect(map).toContain(`href="${prefix}/textbook/agents/coding-web-computer-use/#web-agent-observation-bound-marks"`);
      if (prefix) expectCanonicalEditionNotice(page);
    }
    const card = await readFile(join(distDir, 'sources/papers/alphaproof/index.html'), 'utf8');
    expect(card).toContain(`href="${route}#alphaproof-search-vs-ttrl"`);
    expect(card).toContain('10.1038/s41586-025-09833-y');
  });
  it('renders coding, web and computer-use mechanisms with original figures and examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/coding-web-computer-use/index.html'), 'utf8');
    for (const figure of ['taco-cota-examples.png', 'swe-agent-overview.png',
      'm06-p18-visualwebarena-shopping.png', 'm06-p60-search-backtracking.png',
      'osworld-task-environment.png', 'm07-p72-aguvis-two-stage-training.png'])
      expect(html.includes('/' + figure), 'missing original figure: ' + figure).toBe(true);
    for (const source of ['https://arxiv.org/html/2405.15793v3#S1.F1',
      'https://arxiv.org/html/2404.07972v1#S2.F2', 'https://arxiv.org/html/2404.07972v1#A3.SS1'])
      expect(html).toContain('href="' + source + '"');
    const repair = 'https://arxiv.org/html/2403.17134v2/states_diagram2.drawio.svg';
    expect(html).toContain('href="' + repair + '"');
    expect(html).not.toContain('src="' + repair + '"');
    const svg = html.match(/<img[^>]+src="([^"]*\/repairagent-figure2\.svg)"[^>]+alt="[^"]+"/)?.[1];
    expect(svg).toBeDefined();
    expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(svg!)))).digest('hex'))
      .toBe('8c81df177e7f97bed18d45bd0df30de6edd5d73a2928699278bdff4dd703dea1');
    const agentTrekReplay = 'https://arxiv.org/html/2412.09605v2#S2.F5';
    expect(html).toContain('href="' + agentTrekReplay + '"');
    expect(html).not.toContain('src="https://arxiv.org/html/2412.09605v2/images/part2_v6.png"');
    const agentTrek = html.match(/<img[^>]+src="([^"]*\/agenttrek-figure4\.png)"[^>]+alt="[^"]+"/)?.[1];
    expect(agentTrek).toBeDefined();
    expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(agentTrek!)))).digest('hex'))
      .toBe('8d3b3dc9060f631ef25c42b8085e7b1a79fd6b2312ea6dfd1aed12930709fc77');
    for (const value of ['Web 2.24M', 'CC BY-NC-ND 4.0', 'https://arxiv.org/html/2412.09605v2#S2.F4'])
      expect(html).toContain(value);
    for (const name of ['best_first_fixture', 'parse_assignment', 'express_hypothesis',
      'collect_more_information', 'CC BY 4.0', '14,6%', '16,7%', '6 963',
      'heap-buffer-overflow', 'Step Accuracy', 'None', 'dispatch_click', 'stale_observation'])
      expect(html.includes(name), 'missing mechanism: ' + name).toBe(true);
    for (const anchor of ['свободный-loop-aci-и-процедурный-workflow',
      'html-accessibility-tree-screenshot-и-set-of-marks',
      'web-agent-observation-bound-marks',
      'почему-реактивный-следующий-клик-ломается-на-длинном-горизонте',
      'osworld-resettable-vm-executable-action-and-final-state-grader',
      'taco-совместно-обучать-reasoning-и-action-tokens',
      'минимальный-parser-repair-с-наблюдаемым-результатом'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('shows the original ReAct trajectory in the agentic-learning overview and retires the project PNG', async () => {
    for (const prefix of ['', 'en/']) {
      const page = await readFile(join(distDir, prefix, 'research/agentic-learning/index.html'), 'utf8');
      expect(page).toMatch(/<img[^>]+src="[^"]*\/react-hotpotqa-primary\.png"[^>]+alt="[^"]+"/);
      // This overview is canonical Russian content in both shells.
      expect(page).toContain('href="/textbook/agents/tool-use/#react-hotpotqa-trajectories"');
      expect(page).toContain('href="https://react-lm.github.io/files/diagram.png"');
      expect(page).not.toMatch(/<img[^>]+src="[^"]*(?:react-patterns\.png|react-lm.github.io\/files\/diagram\.png)"/);
    }
    const retired = 'Figures/curated/source-first-65-68/react-patterns.png';
    for (const path of ['assets', 'assets/00 Учебник/Assets']) {
      await expect(readFile(join(distDir, path, retired))).rejects.toMatchObject({code: 'ENOENT'});
    }
  });
  it('links the Berkeley reading to the EnIGMA section and displays both proceedings figures locally', async () => {
    const route = 'textbook/agents/coding-web-computer-use/';
    const anchor = 'ctf-состояние-специализированные-инструменты-и-обратная-связь';
    const note = 'sources/courses/berkeley-advanced-llm-agents-spring-2025/readings/meeting-05-reading-01/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + route, 'index.html'), 'utf8');
      const ctf = html.match(new RegExp('id="' + anchor + '"[\\s\\S]*?(?=<h2\\b)'))?.[0];
      expect(ctf).toBeDefined();
      for (const figure of ['enigma-figure1.png', 'enigma-figure3.png']) {
        expect(ctf).toMatch(new RegExp('<img[^>]+src="[^"]*/' + figure.replace('.', '\\.') + '"'));
      }
      for (const text of ['debug_add_breakpoint parse_header', 'starti', '170 строк', 'ещё 155',
        '29,5%', '27,4%', '20,45%', '11,36%', '17,24%', 'https://proceedings.mlr.press/v267/abramovich25a.html'])
        expect(ctf).toContain(text);
      expect(ctf).not.toContain('EnIGMA_Figure1.png');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    const source = await readFile(join(distDir, note, 'index.html'), 'utf8');
    expect(source).toContain('href="/' + route + '#' + encodeURIComponent(anchor) + '"');
  });
  it('renders agent evaluation mechanisms, original figures and worked examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/evaluation/index.html'), 'utf8');
    for (const figure of ['swe-bench-construction.png', 'swe-bench-evaluation.png',
      'tau-bench-task-components.png', 'paperbench-rubric-tree.png', 'agentbench-environments-results.png',
      'appworld-task-generator.png'])
      expect(html.includes('/' + figure), 'missing evaluation figure: ' + figure).toBe(true);
    for (const original of [
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/bd42e7b2f3e9bb5218142796d3ede4816588dec0-4584x2834.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/0205b36f9639fc27f2f6566f73cb56b06f59d555-4584x2580.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/3ddac5be07a0773922ec9df06afec55922f8194a-4584x2580.png']) {
      expect(html.includes('src="' + original + '"')).toBe(false);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    for (const name of ['grade_booking', 'estimate_pass_at_k', 'resampled_means', 'weighted_score',
      'FAIL_TO_PASS', 'PASS_TO_PASS', 'reproduce.sh', 'CC BY 4.0', '/licenses/human-eval-b9453e8f-MIT.txt'])
      expect(html.includes(name), 'missing evaluation mechanism: ' + name).toBe(true);
    for (const anchor of ['срок-жизни-benchmark-и-несмещённый-passk', '9-модель-судья',
      '10-безопасность-оценивается-как-инвариант', '12-сопоставимость-тестовой-и-рабочей-среды'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('renders the licensed ToolSandbox and τ-bench originals on both evaluation routes', async () => {
    const route = 'textbook/agents/evaluation/';
    const source = 'sources/papers/toolsandbox/';
    const originals = [
      ['toolsandbox-roles.png', 'c751b9a8d69298152c2ff1b2669122429972f263d517d1b56edd86c721f7f315', 'https://arxiv.org/html/2408.04682v2#S2.F2', 'https://creativecommons.org/licenses/by-sa/4.0/'],
      ['toolsandbox-trace.png', '90d2eb262f57820781ce514e30ea2d8cc3a40885667d19741324701a163cd069', 'https://arxiv.org/html/2408.04682v2#S1.F1', 'https://creativecommons.org/licenses/by-sa/4.0/'],
      ['tau-bench-passatk.svg', '8fb2df6aa42ed221cecb9a5fcb70354fdc43ef2a449c31305d476f90effce8c9', 'https://arxiv.org/html/2406.12045v1#S5.F4', 'https://creativecommons.org/licenses/by/4.0/'],
    ];
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const note = await readFile(join(distDir, locale, source, 'index.html'), 'utf8');
      for (const [name, hash, figure, license] of originals) {
        const img = chapter.match(new RegExp('<img[^>]+src="([^"]*/' + name.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(img, name).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(img!)))).digest('hex')).toBe(hash);
        expect(chapter).toContain(figure);
        expect(chapter).toContain(license);
      }
      const trace = chapter.match(/id="agent-eval-toolsandbox"[\s\S]*?(?=<div class="sl-heading-wrapper level-h2")/)?.[0];
      expect(trace).toContain('toolsandbox-trace.png');
      expect(trace).toContain('ConnectionError');
      expect(trace).toContain('https://arxiv.org/html/2408.04682v2#S2.SS3');
      expect(chapter).toContain('id="agent-eval-tau-repeatability"');
      expect(chapter).toContain(`href="/${locale}${source}"`);
      expect(note).toContain(`href="/${locale}${route}#agent-eval-toolsandbox"`);
      expect(chapter).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(chapter);
    }
  });
  it('keeps the stateful evaluation case and AppWorld source connected in both readers', async () => {
    const route = 'textbook/agents/evaluation/';
    const source = 'sources/papers/appworld/';
    const ids = ['agent-eval-state-side-effects', 'agent-eval-appworld', 'agent-eval-coverage'];
    for (const locale of ['', 'en/']) {
      const chapter = await readFile(join(distDir, locale, route, 'index.html'), 'utf8');
      const index = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
      const note = await readFile(join(distDir, locale, source, 'index.html'), 'utf8');
      for (const id of ids) {
        expect(chapter).toContain(`id="${id}"`);
        expect(index).toContain(`href="/${locale}${route}#${id}"`);
      }
      expect(chapter).toMatch(/<img[^>]+src="[^"]*\/appworld-task-generator\.png"/);
      expect(chapter).toContain('https://arxiv.org/html/2407.18901v1#S3.F3');
      expect(chapter).toContain('70–72%');
      expect(chapter).toContain('Score.unscored()');
      expect(chapter).toContain(`href="/${locale}${source}"`);
      expect(note).toContain(`href="/${locale}${route}#agent-eval-appworld"`);
      expect(chapter).not.toContain('katex-error');
    }
  });
  it('renders memory retrieval, planning and orchestration with original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/memory-planning-orchestration/index.html'), 'utf8');
    for (const figure of ['tot-game24-search.png', 'reflexion-loop.png',
      'memgpt-figure3-memory-hierarchy.png', 'webdreamer-planning-paradigms.png',
      'm03-p31-hipporag-index-and-retrieval.png', 'm03-p70-webdreamer-worked-example.png',
      'm01-p64-self-debugging-feedback.png', 'm01-p67-self-correction-no-oracle.png',
      'm01-p69-debate-vs-self-consistency.png'])
      expect(html.includes('/' + figure), 'missing original figure: ' + figure).toBe(true);
    for (const original of [
      'https://arxiv.org/html/2304.03442v2/figures/figure_reflection6.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png']) {
      expect(html.includes('src="' + original + '"')).toBe(false);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    for (const name of ['minmax', 'ppr_step', 'try_candidates', 'accept_result',
      'passed_given_cases', 'accepted_for_review', 'request_heartbeat', 'CC BY 4.0'])
      expect(html.includes(name), 'missing mechanism: ' + name).toBe(true);
    for (const name of ['method.png', 'sequencing.png', 'nut-assembly.png'])
      expect(html.includes('/planseqlearn-2026-10-26/' + name)).toBe(true);
    for (const original of ['Planning Module: Zero-Shot High-level Planning',
      'Sequencing Module: Vision-based Plan Tracking', 'Efficiently Learning Local Control',
      'Stage Termination Details', '0{,}6561']) expect(html.includes(original)).toBe(true);
    expect(html.includes('id="plan-seq-learn"')).toBe(true);
    expect(html.includes('alt="PSL: языковой план, движение к объекту и локальное управление"')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
    for (const anchor of ['1-память-находится-вне-весов', '5-извлечение-не-только-nearest-neighbors',
      'реактивный-агент-поиск-в-реальной-среде-и-model-based-planning',
      'когда-более-сильной-модели-нужен-менее-жёсткий-scaffold',
      '7-рефлексия-полезна-только-через-наблюдаемую-обратную-связь', '10-контракт-между-агентами'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
  });
  it('renders harness state, context budgets, original figures and preserved deep links', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/harness-context-engineering/index.html'), 'utf8');
    for (const name of ['swe-agent-prompt-workflow.png', 'lost-middle-task-controls.png', 'lost-middle-position-effect.png'])
      expect(html.includes('/' + name), 'missing harness figure: ' + name).toBe(true);
    const unescaped = html.replace(/&(?:amp|#x26|#38);/gi, '&');
    for (const original of [
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/faa261102e46c7f090a2402a49000ffae18c5dd6-2292x1290.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/0442fe138158e84ffce92bed1624dd09f37ac46f-2292x1288.png',
      'https://mintcdn.com/allhandsai/hwL6uO0ZqM-lmlYG/sdk/guides/assets/condenser_overview_light_mode.png?fit=max&auto=format&n=hwL6uO0ZqM-lmlYG&q=85&s=dc85afa9f2821b1f7a305ad1406ab3d9']) {
      expect(unescaped.includes('src="' + original + '"')).toBe(false);
      expect(unescaped.includes('href="' + original + '"')).toBe(true);
    }
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['history_capacity', 'common_prefix_length', 'valid_checkpoint',
      'update_reminders', 'run_episode', 'tested_revision', 'Appendix C', 'CC BY 4.0'])
      expect(html.includes(value), 'missing harness explanation: ' + value).toBe(true);
    for (const anchor of ['5-контекст-со-временем-портится',
      '6-compaction-как-преобразование-состояния', '7-reminder-tokens--не-особый-вид-токена',
      '1-из-чего-состоит-один-шаг', '10-наблюдаемость-и-доказательство-завершения'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable harness anchor: ' + anchor).toBe(true);
  });
  it('keeps agent chapter illustrations local on both reader routes', async () => {
    const fixtures = [
      ['tool-use', 'toolsandbox-roles.png', 'ConnectionError'],
      ['harness-context-engineering', 'management-strategies.png', 'timeout=45'],
      ['memory-planning-orchestration', 'reflexion-loop.png', 'часовой пояс'],
    ];
    for (const locale of ['', 'en/']) {
      for (const [chapter, figure, example] of fixtures) {
        const html = await readFile(join(distDir, locale, 'textbook/agents', chapter, 'index.html'), 'utf8');
        const images = [...html.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/gi)].map(([, src]) => src);
        expect(images.filter(src => /^(?:https?:)?\/\//i.test(src)), chapter).toEqual([]);
        expect(images.some(src => src.endsWith('/' + figure)), chapter).toBe(true);
        expect(html, chapter).toContain(example);
      }
    }
  });
  it('joins agent harness boundaries, source evidence and the subject index in both locales', async () => {
    for (const prefix of ['', 'en/']) {
      const chapter = 'textbook/agents/harness-context-engineering/';
      const note = 'sources/papers/complexity-trap-context-management/';
      const html = await readFile(join(distDir, prefix, chapter, 'index.html'), 'utf8');
      const source = await readFile(join(distDir, prefix, note, 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      const library = await readFile(join(distDir, prefix, 'sources/index.html'), 'utf8');
      for (const term of ['harness-observation-masking', 'harness-compaction-boundaries',
        'harness-handoff-unknown', 'harness-reminder-tokens']) {
        expect(html).toContain('id="' + term + '"');
        expect(reference).toContain('href="/' + prefix + chapter + '#' + term + '"');
      }
      expect(html).toMatch(/<img[^>]+src="[^"]*\/harness-context-2026-11-09\/management-strategies\.png"/);
      for (const value of ['54,8%', '53,8%', 'safe_cuts', 'ToolCallMatchingProperty'])
        expect(html).toContain(value);
      expect(html).toContain('href="https://arxiv.org/html/2508.21433v3#S3.F3"');
      expect(html).toContain('href="/' + prefix + 'textbook/agents/tool-use/#mcp-response-correlation"');
      expect(html).toContain('href="/' + prefix + note + '"');
      expect(library).toContain('href="/' + note + '"');
      expect(source).toContain('href="/' + prefix + chapter + '#harness-observation-masking"');
    }
  });
  it('renders the tool-use mechanisms, original ReAct panels and executable fixtures', async () => {
    for (const locale of ['', 'en/']) {
    const html = await readFile(join(distDir, locale, 'textbook/agents/tool-use/index.html'), 'utf8');
    for (const name of ['react-hotpotqa-primary.png', 'react-alfworld-primary.png'])
      expect(html.includes('/' + name), 'missing ReAct panel: ' + name).toBe(true);
    const project = 'https://react-lm.github.io/files/diagram.png';
    expect(html).toContain('href="' + project + '"');
    expect(html).not.toContain('src="' + project + '"');
    const worked = html.match(/<a id="react-hotpotqa-trajectories"><\/a>[\s\S]*?(?=<h2\b)/)?.[0];
    expect(worked).toMatch(/<img[^>]+src="[^"]*\/react-hotpotqa-primary\.png"/);
    expect(html.match(/<img[^>]+src="[^"]*\/react-hotpotqa-primary\.png"/g)).toHaveLength(1);
    expect(html).toContain('href="#react-hotpotqa-trajectories"');
    const loop = 'https://www-cdn.anthropic.com/images/4zrzovbb/website/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png';
    expect(html).not.toContain('src="' + loop + '"');
    expect(html).toContain('href="' + loop + '"');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/agent-evaluation-2026-11-10\/toolsandbox-roles\.png"/);
    expect(html).toContain('ConnectionError');
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['continuation_loss', '1.3025', 'idempotency_parameter_mismatch',
      'run_episode', 'readOnlyHint', 'CC BY 4.0', '2025-11-25'])
      expect(html.includes(value), 'missing tool-use explanation: ' + value).toBe(true);
    for (const anchor of ['где-начинается-агент-модель-память-действие-и-среда',
      'два-способа-проектировать-агента-llm-first-и-agent-first',
      'как-toolformer-получает-обучающие-примеры', '4-наблюдение-должно-отвечать-на-следующий-вопрос',
      '5-ошибка--часть-протокола', '8-как-проверять-tool-use'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable tool-use anchor: ' + anchor).toBe(true);
    for (const anchor of ['react-prompting-learning','toolformer-curation-evaluation'])
      expect(html).toContain('id="'+anchor+'"');
    for (const file of ['react-cot-sampling.png','react-prompt-vs-finetune.png'])
      expect(html).toContain('agent-reconciliation-2026-10-30/'+file);
    for (const example of ['insertion_sites','fallback_to_react','calls_already_made'])
      expect(html).toContain(example);
    expect(html.match(/<pre\b/g)).toHaveLength(11);
    for (const anchor of ['mcp-host-client-server','mcp-response-correlation','tool-retry-idempotency'])
      expect(html).toContain('id="'+anchor+'"');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/mcp-simple-diagram\.png"/);
    expect(html).toContain('href="https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/LICENSE"');
    for (const mechanism of ['correlate_response','settings.txt not found','tool_error','rpc_error'])
      expect(html).toContain(mechanism);
    const reference = await readFile(join(distDir, locale, 'reference/index.html'), 'utf8');
    expect(reference).toContain('href="/'+locale+'textbook/agents/tool-use/#mcp-response-correlation"');
    }
  });
  it('renders Chameleon decoding, the local VQ diagram and the Emu3 comparison', async () => {
    const route = 'textbook/multimodal/unified-sequence-chameleon/';
    const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
    for (const name of ['chameleon-architecture.png', 'chameleon-mixed-document.png',
      'chameleon-norm-qk-dropout.png', 'chameleon-training-ablations.png', 'chameleon-human-evaluation.png',
      'chameleon-recipe-response.png'])
      expect(html.includes('/' + name), 'missing Chameleon figure: ' + name).toBe(true);
    const original = 'https://arxiv.org/html/1711.00937v2/figures/Figure1_9.png';
    expect(html.includes('src="' + original + '"')).toBe(false);
    expect(html.includes('href="' + original + '"')).toBe(true);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['encoder_grad', 'gemini_plus_score_from_figure', '58.75', '3552', 'CC BY 4.0',
      'allowed_mass', 'masked_before', 'admits_begin_marker', 'ReplicatedInputTokenSelector'])
      expect(html.includes(value), 'missing Chameleon explanation: ' + value).toBe(true);
    for (const anchor of ['3-как-изображение-становится-дискретными-кодами',
      '4-одна-авторегрессионная-последовательность', '5-что-происходит-при-генерации',
      '6-почему-совместное-обучение-становится-нестабильным',
      '8-что-даёт-единый-словарь-а-чего-он-не-гарантирует'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable Chameleon anchor: ' + anchor).toBe(true);
    for (const prefix of ['', 'en/']) {
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      const chapter = prefix ? await readFile(join(distDir, prefix, route, 'index.html'), 'utf8') : html;
      if (prefix) expectCanonicalEditionNotice(chapter);
      for (const [figure, digest, source] of [
        ['samuel-ovrelid-vq-figure1.png', 'e888c94fa80891363d546d8baad62a585168810ecdd82023eeb666b380e58c5d', 'https://aclanthology.org/2023.findings-acl.890.pdf#page=2'],
        ['emu3-figure1.png', 'c16ba6fccd32d9db9f748c171ecf5eefc408b57b54aa2dcf8b9667a3f6c7aafc', 'https://arxiv.org/html/2409.18869v1/emu3_arch5.png'],
      ]) {
        const image = chapter.match(new RegExp('<img[^>]+src="([^"]*/' + figure.replace('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(image, figure).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(image!)))).digest('hex')).toBe(digest);
        expect(chapter).toContain(source);
      }
      const emu3 = chapter.match(/id="chameleon-to-emu3"[\s\S]*?(?=<div class="sl-heading-wrapper level-h2")/)?.[0];
      expect(emu3).toContain('4096');
      expect(emu3).toContain('1024');
      expect(emu3).toContain('https://arxiv.org/html/2409.18869v1#S2.SS4');
      for (const id of ["chameleon-mixed-response","chameleon-vq-codebook","chameleon-vq-gradient","chameleon-token-sequence","chameleon-output-mask","chameleon-image-cfg","chameleon-image-budget","chameleon-qknorm","chameleon-zloss","chameleon-mixed-evaluation"]) {
        expect(reference).toContain('href="/' + prefix + route + '#' + id + '"');
        expect(chapter).toContain('id="' + id + '"');
      }
    }
  });
  it('renders diffusion mechanisms, ten source figures and stable chapter anchors', async () => {
    const route = 'textbook/multimodal/diffusion-flow-images/';
    const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
    for (const name of ['mit-cfg-mnist.png', 'mit-unet.png', 'dit-architecture.png',
      'mit-conditional-marginal.png', 'mit-conditional-marginal-trajectories.png',
      'mit-flow-training-result.png', 'sd3-mmdit.png', 'sd3-fid-vs-steps.png',
      'stable-diffusion-text-dataset-v2.png', 'stable-diffusion-components-and-tensors.png'])
      expect(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"').test(html),
        'missing source figure: ' + name).toBe(true);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const anchor of ['1-как-создаётся-учебный-пример',
      '2-почему-локальное-предсказание-позволяет-генерировать-целую-картинку',
      '3-latent-diffusion-denoising-не-в-rgb', '4-как-текст-влияет-на-пространственные-признаки',
      '5-u-net-dit-и-objective--разные-оси', '6-flow-matching-вместо-шума--скорость',
      '7-быстрая-проверка-понимания'])
      expect(html.includes('id="' + anchor + '"'), 'missing anchor: ' + anchor).toBe(true);
    for (const value of ['CC BY-NC-ND 4.0', 'mean_endpoint', 'importance_corrected', '154',
      'from_paper', 'wrong_sign', 'rf/lognorm(0.50, 0.60)', 'CLIP L/14'])
      expect(html.includes(value), 'missing chapter content: ' + value).toBe(true);
    for (const prefix of ['', 'en/']) {
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      const chapter = prefix ? await readFile(join(distDir, prefix, route, 'index.html'), 'utf8') : html;
      if (prefix) expectCanonicalEditionNotice(chapter);
      for (const url of ['https://hojonathanho.github.io/diffusion/assets/img/algorithms.png',
        'https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure3-1-1024x508.png']) {
        expect(chapter).toContain('href="' + url + '"');
        expect(chapter).not.toContain('src="' + url + '"');
      }
      for (const [anchor, name, sha, explanation] of [
        ['diffusion-training-row', 'stable-diffusion-text-dataset-v2.png', 'c599107951905474b4aa6d3aaf9b96c8499f51c0918021d21f2648df4b6a0569', 'Step = 2'],
        ['latent-diffusion-inference-shapes', 'stable-diffusion-components-and-tensors.png', '329cc9d2251837e26578f1330aee55feba5bd2307934105a48ca7eaee9ee6b1e', 'отдельного'],
      ]) {
        const section = chapter.split('id="' + anchor + '"')[1]?.split('<h2')[0];
        const source = section?.match(new RegExp('<img[^>]+src="([^"]*/' + name.replaceAll('.', '\\.') + ')"[^>]+alt="[^"]+"'))?.[1];
        expect(source, name).toBeDefined();
        expect(section).toContain(explanation);
        expect(section).toContain('https://jalammar.github.io/images/stable-diffusion/' + name);
        expect(section).toContain('https://creativecommons.org/licenses/by-nc-sa/4.0/');
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(source!)))).digest('hex')).toBe(sha);
      }
      for (const id of ["diffusion-forward-noise","diffusion-score","diffusion-ddim","diffusion-latent","diffusion-cfg","diffusion-dit","flow-marginal-field","flow-sd3-time","flow-sampling-budget"]) {
        expect(reference).toContain('href="/' + prefix + route + '#' + id + '"');
        expect(chapter).toContain('id="' + id + '"');
      }
    }
  });
  it('includes one accessible source-figure viewer without replacing image-file links', async () => {
    const html = await readFile(join(distDir, 'textbook/retrieval/sparse-dense-hybrid/index.html'), 'utf8');
    expect(html.match(/<dialog\b[^>]*data-figure-viewer/g)).toHaveLength(1);
    expect(html).toContain('aria-labelledby="figure-viewer-title"');
    for (const control of ['data-viewer-minus', 'data-viewer-plus', 'data-viewer-fit', 'data-viewer-close'])
      expect(html).toContain(control);
    expect(html).toMatch(/<a\b[^>]*href="[^"]*\/ivf-build\.png"[^>]*>\s*<img/);
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
  });
  it('links the 404 header to the configured publication root', async () => {
    const html = await readFile(join(distDir, '404.html'), 'utf8');
    const titleLink = html.match(/<a\b[^>]*class="[^"]*\bsite-title\b[^"]*"[^>]*>/)?.[0];
    const href = titleLink?.match(/\bhref="([^"]+)"/)?.[1];
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    expect(href).toBe(`${base}/`);
    expect(html).not.toContain('class="sl-link-button');
  });

  // This walks the complete emitted book (currently 1,961 pages), not one page.
  // Disk-dependent wall time is not the acceptance criterion; every link is.
  it('has no broken emitted internal routes or fragments', async () => {
    await expect(findBrokenBuiltLinks(distDir)).resolves.toEqual([]);
  }, 30_000);
  it('does not publish local file edit links or workstation paths', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );
    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).not.toMatch(/file:\/\/|\/Users\//);
    }
  });

  it('renders formulas and source figures instead of exposing Markdown image syntax', async () => {
    const formula = await readFile(
      join(distDir, 'textbook', 'neural-networks', 'backpropagation', 'index.html'),
      'utf8'
    );
    const figure = await readFile(
      join(distDir, 'textbook', 'rag', 'full-pipeline', 'index.html'),
      'utf8'
    );

    expect(formula).toContain('class="katex');
    expect(figure).toContain('<img');
    for (const name of ['rag-sequence-token.png', 'hf-rag-evaluation-workflow.png',
      'm03-p31-hipporag-index-and-retrieval.png', 'lost-middle-position.png',
      'alce-citation-evaluation.png', 'openscholar-inference.png',
      'openscholar-training.png', 'hf-rag-settings-accuracy.png'])
      expect(figure).toContain(name);
    expect(figure).toContain('class="katex');
    expect(figure).not.toContain('class="katex-error');
    for (const anchor of [
      'почему-похожий-фрагмент-не-всегда-является-нужной-памятью',
      'hipporag-ассоциативное-извлечение-по-графу',
      'openscholar-литературный-поиск-как-итеративный-retrieval-and-synthesis-loop'
    ]) expect(figure).toContain('id="' + anchor + '"');
    expect(figure).not.toContain('<p>![');
  });

  it('renders AD modes, complete paper traces and both custom interfaces in root and EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/backpropagation/index.html'),'utf8');
      for(const filename of ['baydin-figure4-graph.png','baydin-table2-forward.png','baydin-table3-reverse.png',
        'cs231n-chain.svg','cs231n-gates.svg','cs231n-sigmoid.svg','computational-graph.svg'])
        expect(html).toContain(filename);
      for(const fact of ['MultiplyAddAD','setup_context','save_for_forward','curvature_loss',
        'expected_hvp','broadcast_inputs','check_forward_ad','gradgradcheck','CC BY 4.0'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L2324');
      expect(html).toContain('17-468.pdf#page=9');
      for(const fact of ['Обратный проход MLP по всему батчу', '-0{,}023841', '-0{,}298007',
        'id="4-правило-цепочки-и-разветвлённый-граф"', 'id="11-как-проверить-собственный-обратный-проход"'])
        expect(html).toContain(fact);
      expect(html).toMatch(/<details>\s*<summary>Открыть примеры JVP,[^<]+<\/summary>[\s\S]*?id="forward-jvp-dual"[\s\S]*?id="custom-forward-transform-rules"[\s\S]*?<\/details>/);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(20);
      expect(html.indexOf('id="reverse-vjp"')).toBeLessThan(html.indexOf('id="autograd-gradient-stability"'));
      expect(html.indexOf('id="autograd-gradient-stability"')).toBeLessThan(html.indexOf('id="finite-difference-gradcheck"'));
      expect(html.indexOf('id="finite-difference-gradcheck"')).toBeLessThan(html.indexOf('id="forward-jvp-dual"'));
      const reference=await readFile(join(distDir,prefix+'reference/index.html'),'utf8');
      for(const anchor of ["backprop-chain-rule","backprop-shared-nodes","affine-layer-backward","micrograd-topological-backward","backward-versus-step","gradient-accumulation-weighted","reverse-vjp","autograd-gradient-stability","autograd-invalid-operations","autograd-recorded-graph","finite-difference-gradcheck","forward-jvp-dual","forward-reverse-traces","hessian-vector-product","gradient-function-transforms","custom-backward-contract","custom-forward-transform-rules"]){
        expect(html).toContain('id="'+anchor+'"');
        expect(reference).toContain('href="/'+prefix+'textbook/neural-networks/backpropagation/#'+anchor+'"');
      }
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders the complete JAX transformation example and its worked gradients in root and EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/backpropagation/index.html'),'utf8');
      for(const fact of ['Gradients as functions: grad, vmap and jit','fast_batched_grad',
        'compiled_per_example','mean_gradient','updated_params','column_targets',
        'wrong_gradient','torch_grad_fn','torch_per_example','1.1925','pytree'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L4003');
      expect(html).toContain('https://docs.jax.dev/en/latest/101/transformations.html');
      expect(html).toContain('not a speedup or a multi-GPU run');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders the JAX runtime examples and complete source diagram in root and explicit EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/graph-compilation-pytorch-jax/index.html'),'utf8');
      for(const fact of ['JAX: tracing, state and compiled control flow','fast_broken_count',
        'threaded_values','trace_signatures','hidden_gain','compiled_branch','sum_cumulative',
        'add_one_donated','single_mesh','psum.png','expected_columns',
        'four logical CPU devices','not four GPUs']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/jax-dc4af684-Apache-2.0.txt');
      expect(html).toContain('jax.ref.new_ref.html');
      expect(html).not.toContain('class="katex-error');
      expect(html).not.toContain('jax.new_ref.html');
      if(prefix) expectCanonicalEditionNotice(html);
    }
    expect(await readFile(join(distDir,'licenses/jax-dc4af684-Apache-2.0.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/jax-dc4af684-Apache-2.0.txt'),'utf8'));
  });

  it('renders the micro-runtime source figure, complete fragments and lifetime example in both locales',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/deployment/edge-federated/index.html'),'utf8');
      for(const fact of ['model-buffer-and-arena.png','microcontroller-inference-planning-a-tensor-arena',
        'lifetimes-determine-which-buffers-can-share-storage','check_arena_plan',
        'branch_offsets','phase_totals','RecordingMicroInterpreter','PrintAllocations',
        '9568','2048','not the RFC','No C++ build or board']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/tflm-Apache-2.0.txt');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
    const license=await readFile(join(distDir,'licenses/tflm-Apache-2.0.txt'),'utf8');
    expect(license).toContain('Apache License');
  });

  it('links the Edge privacy and FedBuff explanations from both subject indexes', async () => {
    const chapter = 'textbook/deployment/edge-federated';
    const paper = 'sources/papers/fedbuff';
    const anchors = ['edge-secure-aggregation', 'edge-weighted-dp', 'edge-fedbuff'];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + chapter, 'index.html'), 'utf8');
      const index = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      for (const anchor of anchors) {
        expect(html).toContain('id="' + anchor + '"');
        expect(index).toContain('href="/' + prefix + chapter + '/#' + anchor + '"');
      }
      expect(html).toContain('fedbuff-concurrency-figure2.png');
      expect(html).toContain('https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf#page=3');
      expect(html).toContain('href="/' + prefix + paper + '/"');
      expect(html).not.toContain('class="katex-error');
      const note = await readFile(join(distDir, prefix + paper, 'index.html'), 'utf8');
      expect(note).toContain('href="/' + prefix + chapter + '/#edge-fedbuff"');
    }
  });

  it('publishes the membership and memorization audits with original ROC axes and links', async () => {
    const chapter = 'textbook/deployment/security-privacy';
    const paper = 'sources/papers/membership-memorization-audits';
    const anchors = ['security-membership-audit', 'security-canary-exposure'];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + chapter, 'index.html'), 'utf8');
      const index = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      const note = await readFile(join(distDir, prefix + paper, 'index.html'), 'utf8');
      for (const anchor of anchors) {
        const href = 'href="/' + prefix + chapter + '/#' + anchor + '"';
        expect(html).toContain('id="' + anchor + '"');
        expect(index).toContain(href);
        expect(note).toContain(href);
      }
      expect(html).toContain('rmia-low-fpr-figure1.png');
      expect(html).toContain('https://proceedings.mlr.press/v235/zarifzadeh24a.html');
      expect(html).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(html).toContain('href="/' + prefix + paper + '/"');
      expect(html).not.toContain('class="katex-error');
    }
  });

  it('publishes worked robustness examples and preserves the separate Harvard course source', async () => {
    const chapter = 'textbook/deployment/robustness';
    const paper = 'sources/papers/label-shift-certified-robustness';
    const anchors = ['robust-label-shift', 'robust-randomized-smoothing', 'robust-accuracy-bounds'];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + chapter, 'index.html'), 'utf8');
      const index = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      const note = await readFile(join(distDir, prefix + paper, 'index.html'), 'utf8');
      for (const anchor of anchors) {
        const href = 'href="/' + prefix + chapter + '/#' + anchor + '"';
        expect(html).toContain('id="' + anchor + '"');
        expect(index).toContain(href);
        if (anchor !== 'robust-accuracy-bounds') expect(note).toContain(href);
      }
      expect(html).toContain('smoothing-votes-figure1.png');
      expect(html).toContain('smoothing-halfspace-figure3.png');
      expect(html).toMatch(/<img[^>]+src="[^"]*\/fgsm-geometry\.svg"/);
      expect(html).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
      expect(html).toContain('https://www.mdpi.com/2227-7390/12/2/247');
      expect(html).not.toMatch(/<img[^>]+src="[^"]*\/gradient-attack\.svg"/);
      expect(html).toContain('href="/' + prefix + paper + '/"');
      expect(html).not.toContain('shift-types.svg');
      expect(html).not.toContain('class="katex-error');
    }
    const course = await readFile(join(distDir, 'sources/courses/harvard-ml-systems/vol2/robust-ai/index.html'), 'utf8');
    expect(course).toContain('shift-types.svg');
    expect(course).toContain('Archived Figure 13: Adversarial Perturbations');
    expect(course).not.toMatch(/<img[^>]+src="[^"]*\/gradient-attack\.svg"/);
  });

  it('publishes the original Gender Shades plot and the bounds for selective outcomes', async () => {
    const chapter = 'textbook/evaluation/responsible-systems';
    const paper = 'sources/papers/gender-shades-selective-labels';
    const anchors = ["responsible-group-metrics","responsible-intersectional-audit","responsible-selective-labels-bounds","responsible-shap-attribution","responsible-unlearning-sisa","responsible-energy-carbon","responsible-safety-audit"];
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + chapter, 'index.html'), 'utf8');
      const ref = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      const note = await readFile(join(distDir, prefix + paper, 'index.html'), 'utf8');
      for (const anchor of anchors) {
        const href = 'href="/' + prefix + chapter + '/#' + anchor + '"';
        expect(html).toContain('id="' + anchor + '"');
        expect(ref).toContain(href);
        if (anchor.includes('intersectional') || anchor.includes('selective')) expect(note).toContain(href);
      }
      expect(html).toMatch(/<img[^>]+gender-shades-figure3\.png/);
      expect(html).toContain('78.3–100%');
      expect(html).toContain('18.4–100%');
      expect(html).toContain('href="/' + prefix + 'textbook/evaluation/imbalanced-classification/#imbalance-selective-labels"');
      expect(html).toContain('href="/' + prefix + paper + '/"');
      expect(html).not.toContain('class="katex-error');
    }
  });

  it('renders the vision-encoder chapter with all six source figures and its stable sections', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/models/index.html'), 'utf8');
    for (const name of ['vit-figure1-patch-sequence.png', 'clip-figure1-training-and-zero-shot.png',
      'clip-objectives.png', 'clip-zero-shot-linear-probe.png', 'siglip-parallelism.png',
      'siglip-batch-size.png']) {
      expect(html).toContain(name);
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    }
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const anchor of ['1-изображение-как-тензор', '2-цена-более-высокого-разрешения',
      '4-как-из-clip-получается-zero-shot-классификатор',
      'clip-и-siglip-две-формы-контрастивного-обучения', '7-диагностические-проверки'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('publishes both vision-objective editions with the paper figures and five subject links', async () => {
    const route = 'textbook/multimodal/models/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + route + 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      for (const file of ['clip-figure1-training-and-zero-shot.png', 'siglip-parallelism.png'])
        expect(html).toContain('/vision-objectives-2026-11-09/' + file + '"');
      expect(html).not.toContain('/vlm-2026/clip-figure1-training-and-zero-shot.png"');
      expect(html).not.toContain('/stanford-cs336-2026/multimodal/siglip-parallelism.png"');
      for (const anchor of ['vlm-patch-projection', 'vlm-clip-objective', 'vlm-multi-positive', 'vlm-siglip-blocks', 'vlm-winoground']) {
        expect(html).toContain('id="' + anchor + '"');
        expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      }
      for (const rounded of ['0.2231', '0.1054', '1.2629', '0.0111', '0.4000'])
        expect(html).toContain(rounded);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });

  it('renders the connector chapter with source figures, code and stable section anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/connectors-fusion/index.html'), 'utf8');
    for (const name of ['llava-figure1-projector.png', 'llava-onevision.png',
      'blip2-qformer.png', 'blip2-attention-masks.png', 'blip2-llm-connection.png',
      'pheye-dense-cross-attention.png', 'qwen3-vl.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    expect(html).toContain('predictor_positions');
    expect(html).toContain('requires_grad_');
    for (const anchor of ['3-blip-2-q-former-как-обучаемое-узкое-место',
      '4-flamingo-отдельная-визуальная-память', 'что-именно-вычисляет-resampler',
      '6-early-fusion-и-deep-fusion', '7-что-обучать-а-что-замораживать'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('publishes the connector follow-up, paper figure and subject links in both editions', async () => {
    const route = 'textbook/multimodal/connectors-fusion/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + route + 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      expect(html).toContain('/connectors-2026-11-09/llava-figure1-projector.png"');
      expect(html).not.toContain('/vlm-2026/llava-figure1-projector.png"');
      expect(html).toContain('href="/' + prefix + 'textbook/multimodal/models/#vlm-multi-positive"');
      for (const anchor of ['vlm-projector-prefix', 'vlm-qformer-masks', 'vlm-qformer-positives', 'vlm-visual-kv', 'vlm-deepstack']) {
        expect(html).toContain('id="' + anchor + '"');
        expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      }
      for (const result of ['0.3833', '0.0667', '308 MiB', '36 MiB', '20 MiB', '8 MiB'])
        expect(html).toContain(result);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });

  it('publishes the whole AnyRes original, spatial examples and six index links in both editions', async () => {
    const route = 'textbook/multimodal/visual-tokenization-resolution/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + route + 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      expect(html).toContain('/resolution-2026-11-09/llava-onevision-anyres.png"');
      expect(html).not.toContain('/stanford-cs336-2026/multimodal/llava-onevision-anyres.png"');
      for (const anchor of ['vlm-pixel-budget', 'vlm-molmo-overlap', 'vlm-anyres', 'vlm-pixtral-packing', 'vlm-mrope-frequencies', 'vlm-tile-boxes']) {
        expect(html).toContain('id="' + anchor + '"');
        expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      }
      for (const value of ['T H W T H W T T', '0.8776', '0.9988', '0.7337', '0.56'])
        expect(html).toContain(value);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });

  it('renders resolution, packing, MRoPE examples and retained original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/visual-tokenization-resolution/index.html'), 'utf8');
    for (const name of ['llava-onevision-anyres.png',
      'qwen2-vl-architecture.png',
      'pixtral-vision-packing.png', 'qwen2vl-figure3-mrope.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    expect(html).toContain('rotate_three');
    expect(html).toContain('molmo_axis');
    expect(html).toContain('to_original');
    for (const anchor of ['1-бюджет-визуальных-токенов',
      'anyres-глобальный-кадр-и-локальные-плитки', '3-dynamic-high-resolution-в-internvl',
      'qwen2-vl-native-resolution-merger-и-mrope',
      '6-почему-одной-позиции-в-последовательности-недостаточно',
      '7-координаты-после-preprocessing', '8-практический-выбор'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('publishes VLM training ablations, weighted updates and paired tests in both locales', async () => {
    const route = 'textbook/multimodal/training-data/';
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + route + 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      for (const name of ['prismatic-figure4.png', 'prismatic-figure5.png'])
        expect(html).toContain('/vlm-training-2026-11-09/' + name + '"');
      for (const anchor of ['vlm-training-freeze', 'vlm-training-targets', 'vlm-training-packing', 'vlm-training-mix', 'vlm-training-reduction', 'vlm-training-image-control']) {
        expect(html).toContain('id="' + anchor + '"');
        expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      }
      for (const value of ['51.26', '54.33', '46.44', '44.45', '64.08', '42.56',
        '0.125', '0.4', '75%', '50%', '0%'])
        expect(html).toContain(value);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });

  it('renders VLM training steps, loss masks, normalization and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/training-data/index.html'), 'utf8');
    for (const name of ['internvl35-training-process.png', 'molmopoint-guisyn-figure3.png',
      'llava-gen.png', 'llava-onevision-training.png', 'llava-onevision-transfer-s1.png',
      'llava-onevision-transfer-s2.png', 'qwen-vl-stages.png', 'qwen3-vl-pretraining.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const name of ['active_rows', 'length_weighted_loss', 'frozen_head', '2661.12', 'SAPO'])
      expect(html).toContain(name);
    for (const anchor of ['1-что-происходит-на-последовательных-стадиях',
      '2-alignment-научить-llm-читать-визуальный-вход', '3-какие-параметры-размораживать',
      '4-как-устроен-loss-mask-в-диалоге', '5-какие-данные-создают-какие-способности',
      '6-как-смешивать-неодинаковые-наборы', '7-curriculum-по-разрешению-и-типам-задач',
      '8-post-training-должен-проверять-опору-на-изображение', '9-карточка-воспроизводимости'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('separates the two InternVL training schedules and does not serve the retired raster', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/multimodal/training-data/index.html'), 'utf8');
      const staged = html.match(/id="vlm-internvl25-stages"[\s\S]*?(?=id="vlm-internvl-native-training")/)?.[0];
      const native = html.match(/id="vlm-internvl-native-training"[\s\S]*?(?=id="vlm-training-freeze")/)?.[0];
      expect(staged).toBeDefined();
      expect(native).toBeDefined();
      expect(staged).toContain('https://arxiv.org/pdf/2412.05271v1#page=6');
      expect(staged).toMatch(/<table>[\s\S]*?1\.5\.[\s\S]*?<\/table>/);
      expect(staged).toContain('shared weights');
      for (const value of ['https://arxiv.org/html/2508.18265v1#S2.F3', 'https://arxiv.org/html/2508.18265v1/training_process.png',
        '250B tokens', '130B tokens', '270K samples', 'InternVL2-8B', 'CC BY 4.0']) {
        expect(native).toContain(value);
      }
      const figure = native!.match(/<img[^>]+src="([^"]*\/internvl35-training-process\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('5234140e21251cbfe871417a995edf37f8ee968b44586f2b49253c433ebb14db');
      expect(html).not.toContain('internvl25-figure4-training-pipeline.png');
    }
    const retired = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure4-training-pipeline.png');
    await expect(access(retired)).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('shows code-guided point annotation with the authors’ figure and links the archived PixMo overview', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/multimodal/training-data/index.html'), 'utf8');
      expect(html).toContain('https://arxiv.org/pdf/2409.17146v2#page=2');
      expect(html).not.toContain('molmo-figure1-data-and-capabilities.png');
      const start = html.indexOf('id="vlm-pixmo-code-guided"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start, html.indexOf('<h', start));
      expect(section).toMatch(/<code\b[^>]*>Measure<\/code>/);
      for (const value of ['Playwright', 'CC BY 4.0',
        'https://arxiv.org/html/2603.28069v1#S4.F3', 'https://arxiv.org/html/2603.28069v1#S10',
        'https://arxiv.org/html/2603.28069v1/MolmoPSyn_overview.png'])
        expect(section).toContain(value);
      const figure = section.match(/<img[^>]+src="([^"]*\/molmopoint-guisyn-figure3\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('dcb476acd4a19140b49e2ef0d9d69611e88833f402dd25f7789451d6ea9aad6d');
    }
    const retired = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure1-data-and-capabilities.png');
    await expect(access(retired)).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('renders document mechanisms, source figures, metrics and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/documents-ocr-grounding/index.html'), 'utf8');
    for (const name of ['layoutlmv3-objectives.png', 'donut-overview.png',
      'pix2struct-screenshot-parsing.png', 'pix2struct-variable-resolution.png',
      'grits-figure1-alignment.png', 'grits-figure2-representations.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['anls_one', 'scale_crop', 'coordinate_space', 'CC BY-NC-SA 4.0'])
      expect(html).toContain(text);
    for (const anchor of ['1-из-каких-подзадач-складывается-ответ', '2-четыре-архитектурных-подхода',
      '3-как-проходит-один-запрос', '4-почему-разрешение-и-структура-неразделимы',
      '5-таблица-и-график-не-являются-линейным-текстом', '6-grounding-как-язык-доказательства',
      '7-возврат-координат-в-оригинал', '8-метрики-на-конкретных-примерах',
      '9-диагностические-вмешательства', '10-контракт-документной-системы',
      'doc-grits', 'doc-coordinates', 'doc-anls'])
      expect(html).toContain('id="' + anchor + '"');
  });


  it('keeps Pix2Struct parsing visual and links the original downstream tasks', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/multimodal/documents-ocr-grounding/index.html'), 'utf8');
      expect(html).toMatch(/<img[^>]+src="[^"]*\/pix2struct-screenshot-parsing\.png"/);
      expect(html).not.toContain('pix2struct-figure1-tasks.png');
      const start = html.indexOf('id="doc-pix2struct-transfer"');
      expect(start).toBeGreaterThan(-1);
      const section = html.slice(start, html.indexOf('<h', start));
      for (const value of ['https://proceedings.mlr.press/v202/lee23g/lee23g.pdf#page=2',
        'AI2D', 'carnivore', 'Screen2Words', 'DocVQA', 'RefExp', 'true', 'false'])
        expect(section).toContain(value);
    }
    const retired = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/vlm-2026/pix2struct-figure1-tasks.png');
    await expect(access(retired)).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('renders video/audio mechanisms, original diagrams and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/video-audio-omni/index.html'), 'utf8');
    for (const name of ['llava-onevision-modalities.png', 'gens-frame-selection.png', 'mimi-split-rvq.png',
      'qwen25omni-figure3-tmrope.png',
      'qwen25omni-figure2-thinker-talker.png', 'qwen3omni-architecture.png',
      'moshi-rq-transformer.png', 'moshi-duplex-streams.png', 'qwen25omni-block-attention.png',
      'qwen3omni-figure3-aut.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['words_at', 'semantic_with_flush', 'audio_durations', '12.5', 'CC BY-NC-SA 4.0',
      'uninterrupted_start', 'omni-first-packet', 'omni-playback-gaps'])
      expect(html).toContain(text);
    for (const anchor of ['1-от-видеофайла-до-visual-tokens', '1-от-видеофайла-до-визуальных-токенов',
      'xgen-video-temporal-compression', 'gens-question-aware-frame-selection',
      '2-номер-кадра-и-физическое-время', '3-от-waveform-до-признаков-и-дискретных-аудиотокенов',
      '3-от-звуковой-волны-к-признакам-и-аудиотокенам', '4-cascade-и-end-to-end-speech-model',
      '4-каскадная-и-сквозная-обработка-речи', '5-общая-временная-ось-аудио-и-видео',
      '6-thinkertalker-понимание-и-речь', '7-streaming-меняет-причинный-контракт',
      '7-потоковая-обработка-и-прерывание-ответа', '8-что-именно-означает-omni',
      '9-матрица-оценивания', '9-оценивание-видео-и-речи'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('continues to serve the byte-identical TMRoPE figure in both languages', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/multimodal/video-audio-omni/index.html'), 'utf8');
      const figure = html.match(/<img[^>]+src="([^"]*\/qwen25omni-figure3-tmrope\.png)"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('c0db3d7a142bc4e61ce52a93e55951c362b30cc06049252fd657db879b457d25');
    }
  });

  it('links the three archived VLM illustrations without serving their PNG copies', async () => {
    const examples = [
      ['textbook/multimodal/visual-tokenization-resolution/', 'internvl25-figure2-dynamic-tiles.png',
        'https://arxiv.org/pdf/2412.05271v5#page=3', '1792', 'llava-onevision-anyres.png'],
      ['textbook/multimodal/documents-ocr-grounding/', 'qwen2vl-figure11-dense-ocr.png',
        'https://arxiv.org/pdf/2409.12191v1#page=29', '12840', 'donut-overview.png'],
      ['textbook/multimodal/video-audio-omni/', 'imagebind-figure1-capabilities.png',
        'https://arxiv.org/pdf/2305.05665v2#page=1', '0.8', 'qwen25omni-figure2-thinker-talker.png'],
      ['models/families/multimodal/', 'internvl25-figure2-dynamic-tiles.png',
        'https://arxiv.org/pdf/2412.05271v5#page=3', '256', 'qwen25omni-figure2-thinker-talker.png'],
    ];
    for (const prefix of ['', 'en/']) {
      for (const [route, archive, original, step, localFigure] of examples) {
        const html = await readFile(join(distDir, prefix + route + 'index.html'), 'utf8');
        expect(html, prefix + route).not.toContain('/' + archive);
        expect(html, prefix + route).toContain('href="' + original + '"');
        expect(html, prefix + route).toContain(step);
        expect(html, prefix + route).toContain('/' + localFigure + '"');
        const asset = join(distDir, 'assets/00 Учебник/Assets/Figures/curated/vlm-2026', archive);
        await expect(access(asset), archive).rejects.toMatchObject({ code: 'ENOENT' });
      }
    }
  });

  it('serves Pheye and links the original Flamingo and Molmo figures from textbook and atlas', async () => {
    const retired = ['vlm-2026/flamingo-figure3-architecture.png',
      'vlm-2026/molmo-figure5-image-tokenization.png', 'atlas-remainder-official/flamingo-architecture.png'];
    const flamingo = 'https://arxiv.org/pdf/2204.14198v2#page=4';
    const molmo = 'https://arxiv.org/pdf/2409.17146v2#page=9';
    for (const prefix of ['', 'en/']) {
      const [connectors, spatial, atlas] = await Promise.all([
        'textbook/multimodal/connectors-fusion/', 'textbook/multimodal/visual-tokenization-resolution/',
        'models/families/multimodal/'
      ].map(route => readFile(join(distDir, prefix + route, 'index.html'), 'utf8')));
      for (const path of retired)
        for (const html of [connectors, spatial, atlas]) expect(html).not.toContain('/' + path);
      expect(connectors).toContain(flamingo);
      expect(atlas).toContain(flamingo);
      expect(spatial).toContain(molmo);
      expect(atlas).toContain(molmo);
      expect(spatial).toContain('0–23');
      expect(spatial).toContain('20–39');
      expect(connectors).toContain('id="vlm-pheye-cross-attention"');
      expect(connectors).toContain('https://arxiv.org/html/2501.02584v1#S2.F2');
      expect(atlas).toContain('textbook/multimodal/connectors-fusion/#vlm-pheye-cross-attention');
      expect(atlas).toContain('textbook/multimodal/visual-tokenization-resolution/#vlm-molmo-overlap');
      const figure = connectors.match(/<img[^>]+src="([^"]*\/pheye-dense-cross-attention\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('f74aaed3c0c517217b2449ad8ae4b2f89c2bff39981908a6310243e1a68e4bfc');
    }
    for (const path of retired) {
      const asset = join(distDir, 'assets/00 Учебник/Assets/Figures/curated', path);
      await expect(access(asset)).rejects.toMatchObject({code: 'ENOENT'});
    }
  });

  it('renders the authored connector and ImageBind application figures in both locales', async () => {
    expect(6 * 256).toBe(1536);
    expect(4 * 256 + 2 * 64).toBe(1152);
    expect((1536 - 1152) / 1536).toBe(0.25);
    const cases = [
      ['visual-tokenization-resolution', 'vlm-internvl-resolution-router', 'internvl35-figure2c.png',
        '37ea06166d89b9735cbf2aa3aa158e949557baab53156aea409d940760414f56',
        'https://arxiv.org/html/2508.18265v1#S2.F2', 'https://creativecommons.org/licenses/by/4.0/',
        ['1536', '1152', 'High Resolution', 'Low Resolution']],
      ['video-audio-omni', 'imagebind-audio-retrieval', 'panda-act-figure2.jpg',
        '4c924d9c74c7fc79ff32e2737ef73e347783c0cd7618a4de85f7b2d43c5737cd',
        'https://www.nature.com/articles/s41598-025-17015-z#Fig2', 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
        ['A_1', 'I_1', 'scores', 'argmax', '0.8', '0.6']],
    ] as const;
    for (const prefix of ['', 'en/']) {
      for (const [slug, anchor, filename, hash, source, license, terms] of cases) {
        const html = await readFile(join(distDir, prefix + 'textbook/multimodal', slug, 'index.html'), 'utf8');
        const section = html.match(new RegExp('id="' + anchor + '"[\\s\\S]*?(?=<h[234]\\b|</main>)'))?.[0];
        const body = section && html.slice(html.indexOf(section) + section.length)
          .match(/^<h[234]\b[\s\S]*?(?=<h[234]\b|<\/main>)/)?.[0];
        expect(body).toBeDefined();
        const tag = Array.from(body!.matchAll(/<img[^>]+src="([^"]+)"[^>]*>/g))
          .find(match => match[1].endsWith('/' + filename));
        expect(tag?.[0]).toMatch(/alt="[^"]+"/);
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(tag![1])))).digest('hex')).toBe(hash);
        for (const term of [...terms, source, license]) expect(body).toContain(term);
        expect(body).not.toContain('katex-error');
      }
    }
  });

  it('joins the PP-DocLayoutV2 walkthrough, source note and reference in both locales', async () => {
    const route = 'textbook/multimodal/documents-ocr-grounding/';
    const source = 'sources/papers/paddleocr-vl/';
    const anchor = 'doc-paddleocr-reading-order';
    const figure = 'pp-doclayout-v2-reading-order.png';
    const bytes = await readFile(join(distDir, 'assets/00 Учебник/Assets/Figures/curated/paddleocr-vl-2026-11-11', figure));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe('f02cc9356b2a659757ed83f1477881a0cd95b17e9408588daf0ade82811643e0');
    for (const prefix of ['', 'en/']) {
      const chapter = await readFile(join(distDir, prefix + route, 'index.html'), 'utf8');
      const ref = await readFile(join(distDir, prefix + 'reference/index.html'), 'utf8');
      const note = await readFile(join(distDir, prefix + source, 'index.html'), 'utf8');
      expect(chapter).toContain('id="' + anchor + '"');
      const section = chapter.split('id="' + anchor + '"')[1].split(/<h2\b|<h3\b/g).slice(0, 2).join(' ');
      expect(section).toContain('/' + figure + '"');
      expect(section).toContain('https://arxiv.org/html/2510.14528v1#S2.F3');
      for (const value of ['3,4,0,2,1', '3,5,4,1,2', '12840', 'PP-DocLayoutV2'])
        expect(section).toContain(value);
      expect(chapter).not.toContain('qwen2vl-figure11-dense-ocr.png');
      expect(chapter).not.toContain('katex-error');
      expect(chapter).toContain('href="/' + prefix + source + '"');
      expect(ref).toContain('href="/' + prefix + route + '#' + anchor + '"');
      expect(note).toContain('href="/' + prefix + route + '#' + anchor + '"');
    }
  });

  it('renders VLM evaluation mechanisms, original figures and preserved route anchors', async () => {
    const route = 'textbook/multimodal/evaluation-serving/';
    const html = await readFile(join(distDir, route, 'index.html'), 'utf8');
    for (const name of ['mmvet-integrated-capabilities.png', 'epd-figure5-slo.png',
      'pope-question-construction.png', 'chair-caption-errors.png', 'mm-detect-contamination-figure1.png',
      'epd-stage-interference.png', 'epd-system-architecture.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['winoground_scores', 'binary_metrics', 'critical_path',
      'processor_key', 'block_key', 'quality_ok'])
      expect(html.includes(text), text).toBe(true);
    expect(/(?:&lt;|&#x3[Cc];|&#60;)AND(?:&gt;|&#x3[Ee];|&#62;|>)/.test(html),
      'AND must be rendered as text, not an HTML element').toBe(true);
    for (const anchor of ['1-карта-способностей-и-тестов', '2-как-winoground-изолирует-композиционность',
      '2-как-winoground-проверяет-соответствие-пар', '3-pope-когда-модель-видит-объект-которого-нет',
      '4-проверка-причинной-опоры-на-изображение', '4-проверка-зависимости-ответа-от-изображения',
      '5-открытые-ответы-и-llm-as-a-judge', '6-контаминация-и-открытость',
      '7-serving-полный-путь-одного-запроса', '8-из-чего-состоит-ttft',
      '9-head-of-line-blocking-и-batching', '10-кэш-и-наблюдаемость', '11-безопасность-и-отказ'])
      expect(html).toContain('id="' + anchor + '"');
    for (const prefix of ['', 'en/']) {
      const [chapter, reference] = await Promise.all([
        readFile(join(distDir, prefix, route, 'index.html'), 'utf8'),
        readFile(join(distDir, prefix, 'reference/index.html'), 'utf8'),
      ]);
      expect(chapter).toContain('href="https://cs231n.stanford.edu/slides/2025/lecture_16.pdf#page=50"');
      expect(chapter).not.toContain('cs231n-lecture16-winoground.png');
      expect(chapter).toMatch(/<img[^>]+src="[^"]*\/epd-figure5-slo\.png"/);
      expect(chapter).toContain('runs');
      expect(chapter).not.toContain('class="katex-error');
      for (const anchor of ['vlm-eval-winoground', 'vlm-eval-pope', 'vlm-eval-ttft',
        'vlm-slo-goodput', 'vlm-eval-cache', 'vlm-eval-abstention', 'vlm-eval-joint-quality']) {
        expect(chapter).toContain('id="' + anchor + '"');
        expect(reference).toContain('href="/' + prefix + route + '#' + anchor + '"');
      }
    }
  });

  it('shows both leakage paths and links the archived Molmo openness table', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix, 'textbook/multimodal/evaluation-serving/index.html'), 'utf8');
      const [start, end] = ['vlm-eval-leakage-routes', 'vlm-eval-open-components']
        .map(id => html.indexOf('id="' + id + '"'));
      expect(start).toBeGreaterThan(-1);
      expect(end).toBeGreaterThan(start);
      const section = html.slice(start, end);
      for (const value of ['https://arxiv.org/html/2411.03823v3#S1.F1', 'https://arxiv.org/html/2411.03823v3/fig1_arr.png',
        'CC BY 4.0', 'MMStar', 'LoRA', '478', 'LLaVA']) expect(section).toContain(value);
      const figure = section.match(/<img[^>]+src="([^"]*\/mm-detect-contamination-figure1\.png)"[^>]+alt="[^"]+"/)?.[1];
      expect(figure).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(figure!)))).digest('hex'))
        .toBe('26bb5ad1fbe11732b8b742269fe9d983d66dc5984d2a8c7da6ce23a1cae6ccf8');
      const openness = html.slice(end);
      for (const value of ['https://arxiv.org/pdf/2409.17146v2#page=19', 'OLMo', 'Qwen2 7B', 'Distilled'])
        expect(openness).toContain(value);
      expect(html).not.toContain('molmo-figure11-openness.png');
    }
    const retired = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png');
    await expect(access(retired)).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('ships responsive rules for figures, formulas, tables, and attributed source blocks', async () => {
    const assetsDir = join(distDir, '_astro');
    const { readdir } = await import('node:fs/promises');
    const styles = (await readdir(assetsDir)).filter((file) => file.endsWith('.css'));
    const css = (await Promise.all(styles.map((file) => readFile(join(assetsDir, file), 'utf8')))).join('\n');

    expect(css).toContain('.sl-markdown-content img');
    expect(css).toContain('max-width:100%');
    expect(css).toContain('.katex-display');
    expect(css).toContain('overflow-x:auto');
    expect(css).toContain('.source-attribution');
    expect(css).toContain('overflow-wrap:anywhere');
  });

  it('emits every manifest route as Russian HTML', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );

    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).toMatch(/^<!DOCTYPE html><html lang="ru"/);
    }
  });

  it('does not silently emit empty textbook articles', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) =>
        section.pages.map((page) => page.route).filter((route) => route.startsWith('textbook/'))
    );

    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).not.toContain('<div class="sl-markdown-content"></div>');
    }
  });

  it('renders scientific discovery mechanisms, examples and original source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/scientific-discovery/index.html'), 'utf8');
    for (const figure of ['lasr-overview.svg', 'lasr-evolution.svg', 'lasr-ablations.svg',
      'escher-overview.svg', 'escher-history-prompt.svg', 'escher-bird-concepts.svg',
      'escher-figure2-original.svg', 'm11-p75-pysr-coulomb.png', 'm11-p76-lasr-coulomb.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['fit_inverse_power', 'lasr_score_v3', 'count_pairs',
      'get_confusion_matrix', 'lasr-experiments', 'f5c96df60ebda4dff1e933a54c9f6f8f03677cf3'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['нейронный-proposal-символическая-проверка-and-search',
      'научный-поиск-как-пространство-проверяемых-гипотез', 'escher-visual-concept-library',
      'что-показали-feynman-equations-and-scaling-law-case'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/textbook/agents/formal-math/');
    expect(html).toContain('/textbook/agents/security/');
    expect(html).not.toContain('https://github.com/trishullab/lasr');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders FunSearch and AlphaEvolve originals, evidence and topic links', async () => {
    const route = '/textbook/agents/scientific-discovery/';
    for (const prefix of ['', '/en']) {
      const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const ref = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      for (const anchor of ['funsearch-program-search', 'alphaevolve-evaluation-cascade']) {
        expect(html).toContain(`id="${anchor}"`);
        expect(ref).toContain(`href="${prefix}${route}#${anchor}"`);
      }
      for (const v of ['funsearch-figure-1.png', 'alphaevolve-figure-2.png', 'paraboloid_priority',
        '4096', '48', '4226acbf237ff9ad10ba7673a2af127a2d8a5971', 'https://creativecommons.org/licenses/by-nc-nd/4.0/'])
        expect(html).toContain(v);
      expect(html).not.toContain('katex-error');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    for (const [paper, anchor] of [
      ['funsearch', 'funsearch-program-search'], ['alphaevolve', 'alphaevolve-evaluation-cascade'],
    ]) {
      const card = await readFile(join(distDir, 'sources/papers', paper, 'index.html'), 'utf8');
      expect(card).toContain(`href="${route}#${anchor}"`);
    }
  });

  it('renders TinyBERT originals and 38 distillation subjects in both locales', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-distillation-tinybert-subject-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^  - anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(38);
    for(const locale of ['', 'en/']){
      const route=locale+'textbook/post-training/reasoning-distillation';
      const html=await readFile(join(distDir,route+'/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+route+'/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(10);
      for(const value of ['tinybert-layer-detail.png','tinybert-overview.png',
        'tinybert_score_loss','tinybert_padded_loss','fit_dense','--pred_distill'])
        expect(html,value).toContain(value);
      expect(html).toContain('/'+locale+'sources/papers/tinybert/');
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
      const note=await readFile(join(distDir,locale+'sources/papers/tinybert/index.html'),'utf8');
      expect(note).toContain('/'+route+'/#tinybert-transformer-distillation');
    }
  });
  it('shows both original Wen panels beside the DeepSeek comparison in both reader routes', async () => {
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/reasoning-distillation/index.html'), 'utf8');
      const section = html.match(/<a id="distillation-vs-direct-rl"><\/a>[\s\S]*?(?=<a id="s1-selection-funnel")/)?.[0];
      expect(section).toBeDefined();
      const keys = [
        ['6a', 'cc9b9f8cfe64c75936bb20b99ab979363fd1628eb9225a48c91a729bdc4a8241'],
        ['6b', '48cb10349731a10ef465b97f63d3be5f5e1349cd4ee46cc8510cb2ed8599bada'],
      ];
      for (const [panel, hash] of keys) {
        const path = section!.match(new RegExp(`<img[^>]+src="([^"]*/wen-figure${panel}\\.svg)"[^>]+alt="[^"]+"`))?.[1];
        expect(path).toBeDefined();
        expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(path!.slice(base.length))))).digest('hex'))
          .toBe(hash);
      }
      for (const detail of ['2501.12948v2#A6.T16', '2506.14245v2#S6.F6.sf1', '2506.14245v2#S6.F6.sf2',
        'https://creativecommons.org/licenses/by/4.0/', '47,0', '72,6', '25,6', 'cons@64'])
        expect(section).toContain(detail);
    }
  });
  it('renders distillation mechanisms, actual examples and complete source comparisons', async () => {
    const html=await readFile(join(distDir,'textbook/post-training/reasoning-distillation/index.html'),'utf8');
    for(const figure of ['kim-rush-three-approaches.png','gkd-forward-reverse-kl.svg',
      'sequence-distillation.png','token-distributions.png','gkd-task-comparison.png',
      'm04-p128-s1-data-funnel.png']) expect(html,figure).toContain(figure);
    for(const mechanism of ['seq_full_gradient','seq_mode_gradient','mixed_gradient',
      'div_candidates','gkd_prefix_length','gkd_loss_after','gkd_teacher_before',
      'sequence_distillation_loss','804 745','generalized_jsd_loss'])
      expect(html,mechanism).toContain(mechanism);
    for(const anchor of ['sequence-distillation','selection-bias','distillation-vs-rft',
      'reasoning-recipes','distillation-evaluation','evaluation'])
      expect(html).toContain('id="'+anchor+'"');
    expect(html).toContain('https://arxiv.org/html/2306.13649v3#A1.F16');
    expect(html).toContain('c6eac19f63be27498f2a7c8108d1e8a4dffde23f');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders 34 GRPO subjects, 11 examples and 9 figures in both locale routes', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-05-grpo-subject-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^    anchor: ([a-z0-9-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(34);
    for(const locale of ['', 'en/']){
      const route=locale+'textbook/post-training/grpo-deepseek-r1';
      const html=await readFile(join(distDir,route+'/index.html'),'utf8');
      const index=await readFile(join(distDir,locale+'reference/index.html'),'utf8');
      for(const anchor of anchors){
        expect(html.match(new RegExp('id="'+anchor+'"','g'))).toHaveLength(1);
        expect(index).toContain('href="/'+route+'/#'+anchor+'"');
      }
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(9);
      expect(html).not.toContain('katex-error');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders GRPO mechanisms, executed examples and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/grpo-deepseek-r1/index.html'), 'utf8');
    for (const figure of ['ppo-framework.png', 'group-advantages.png', 'length-difficulty-bias.png',
      'length-and-accuracy-ablation.png', 'r1-zero-training-dynamics.png', 'figure2-pipeline.png',
      'kimi-partial-rollouts.svg', 'maxrl-weight-functions.svg'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['group_advantages', 'clipped_token_loss', 'process_token_advantages',
      'microbatch_loss', 'baseline_expectations', 'maxrl_theoretical', 'maxrl_practical',
      'GRPOConfig', 'offpolicy_surrogate', 'offpolicy_gradients',
      'gspo_gradient', 'kimi_gradient', '0.288', '0.16'])
      expect(html, mechanism).toContain(mechanism);
    for (const anchor of ['rlvr-notation', 'grpo-estimator', 'grpo-normalizations',
      'drgrpo-rft-maxrl', 'offpolicy-ratios', 'gspo', 'reasoning-case-studies'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('https://arxiv.org/html/2602.02710v1#S5.F2');
    expect(html).not.toContain('https://arxiv.org/html/2602.02710v1#S5.F3');
    expect(html).toContain('CC BY-NC-ND 4.0');
    expect(html).toContain('/textbook/post-training/reasoning-distillation/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('publishes the DAPO walkthrough and the unmodified source figure in both GRPO routes', async () => {
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/grpo-deepseek-r1/index.html'), 'utf8');
      const section = html.match(/<a id="dapo-dynamic-sampling"><\/a>[\s\S]*?(?=<a id="grpo-kl-settings")/)?.[0];
      expect(section).toBeDefined();
      const src = section!.match(/<img[^>]+src="([^"]*\/wen-figure4\.svg)"[^>]+alt="[^"]+"/)?.[1];
      expect(src).toBeDefined();
      expect(createHash('sha256').update(await readFile(join(distDir, decodeURIComponent(src!.slice(base.length))))).digest('hex'))
        .toBe('98db9ba58301c872a6dfbb5d01b4117e884ecbe870a2544d584ec3ccfef70c33');
      for (const detail of ['https://arxiv.org/html/2506.14245v2#S5.F4', 'https://arxiv.org/html/2503.14476v2#S4.T1',
        'https://creativecommons.org/licenses/by/4.0/', 'Clip-Higher', '0,0394', '0,875', '18 432', 'avg@32'])
        expect(section).toContain(detail);
    }
  });

  it('renders worked tensor foundations, source illustrations and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/tensors/index.html'), 'utf8');
    for (const figure of ['vec-add.svg', 'vec-angle.svg', 'broadcasting_2.svg',
      'broadcasting_3.svg', 'broadcasting_4.svg', 'np_transposing_reshaping.png',
      'np_reshape.png', 'tensor-memory-layout.svg'])
      expect(html, figure).toContain(figure);
    expect(html).not.toContain('mml-vector-geometry.png');
    expect(html).not.toContain('d2l-broadcasting-example.png');
    expect(html).toContain('source_result');
    expect(html).toContain('Jay Alammar');
    for (const mechanism of ['keepdims', 'centered', 'Осей', 'краткие-итоги'])
      expect(html, mechanism).toContain(mechanism);
    expect(html).toContain('id="суммирование-и-усреднение-по-оси"');
    expect(html).toContain('id="что-происходит-с-несколькими-матрицами"');
    expect(html).toContain('id="почему-нельзя-сразу-сделать-reshape-в-форму-голов"');
    for (const example of ['X_batch', 'W_batch', 'Q_one', 'K_batch', 'swapaxes', 'restored'])
      expect(html, example).toContain(example);
    expect(html).toContain('/textbook/ml-foundations/derivatives-gradients/');
    const broadcasting = html.split('id="поэлементные-операции-и-broadcasting"')[1]
      ?.split('id="суммирование-и-усреднение-по-оси"')[0];
    const dotProduct = html.split('id="скалярное-произведение"')[1]
      ?.split('id="умножение-матриц-как-набор-скалярных-произведений"')[0];
    expect(broadcasting).toBeDefined();
    expect(broadcasting).not.toContain('<ol>');
    expect(dotProduct?.match(/<ol>/g)).toHaveLength(1);
    expect(dotProduct?.match(/<li>/g)).toHaveLength(2);
    expect(html.indexOf('id="нормы"')).toBeLessThan(html.indexOf('id="скалярное-произведение"'));
    expect(html).toContain('X_text[0,1]');
    expect(html).toContain('первую позицию второй фразы');
    for (const alt of ['Второй вектор начинается в конце первого',
      'Столбец из четырёх чисел и строка из трёх чисел',
      'Одна матрица два на три в памяти по строкам и по столбцам']) {
      expect(html).toContain('alt="' + alt);
    }
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the complete tensor storage figure and address walkthrough in root and explicit EN fallback', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + 'textbook/ml-foundations/tensors/index.html'), 'utf8');
      expect(html).toContain('id="shape-strides-and-shared-storage"');
      expect(html).toContain('tensor-memory-layout.svg');
      expect(html).toContain('layout_c');
      expect(html).toContain('layout_f');
      expect(html).toContain('layout_slice');
      expect(html).toContain('CC BY-NC-SA 4.0');
      expect(html).toContain('frameworks.qmd#L2904');
      expect(html).toContain('not the result of transposing');
      expect(html).not.toContain('class="katex-error');
      if (prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('preserves all 21 Frameworks source images independently of thematic selection', async () => {
    const report=JSON.parse(await readFile(join(import.meta.dirname,
      '../../docs/audit/2026-10-27-frameworks-native-figures.json'),'utf8'));
    const route = 'sources/courses/harvard-ml-systems/vol1/frameworks';
    const html=await readFile(join(distDir,route,'index.html'),'utf8');
    expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(21);
    for (const row of report.figures) expect(html).toContain(row.asset.split('/').at(-1));
    expectCourseOriginalAlias(await readFile(join(distDir,'en',route,'index.html'),'utf8'), route);
  });

  it('renders derivative reasoning, Jacobians, checks and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/derivatives-gradients/index.html'), 'utf8');
    for (const figure of ['d2l-function-tangent.svg', 'google-gradient-descent.png',
      'cs231n-chain.svg', 'd2l-chain-paths.svg'])
      expect(html, figure).toContain(figure);
    for (const retired of ['mml-secant-to-tangent.png', 'mml-computation-graph-forward.png',
      'mml-matrix-gradient-shape.png', 'mml-matrix-gradient-flatten.png'])
      expect(html).not.toContain(retired);
    for (const page of ['147', '161', '166'])
      expect(html).toContain('mml-book.pdf#page=' + page);
    expect(html).toContain('id="differentiation-rules"');
    expect(html).toContain('id="several-paths-through-a-graph"');
    for (const mechanism of ['numerical_gradient', '1e-18',
      'Неодинаковые множители и векторные состояния'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['краткие-итоги', 'производная-векторной-функции-якобиан',
      'проверка-градиента-по-координатам', 'точка-излома-требует-отдельной-проверки'])
      expect(html).toContain('id="' + alias + '"');
    expect(html).toContain('/textbook/ml-foundations/tensors/');
    expect(html).toContain('id="прямой-и-обратный-проход-линейного-отображения"');
    expect(html).toContain('numeric_linear');
    expect(html).toContain('numeric_W');
    expect(html).toContain('id="производные-по-весам-линейного-слоя"');
    expect(html).toContain('alt="К параболе');
    expect(html).toContain('alt="Три пути из y в f');
    expect(html).toContain('/textbook/ml-foundations/probability-likelihood-logarithms/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders probability examples, original figures and the NLL bridge', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/probability-likelihood-logarithms/index.html'), 'utf8');
    for (const figure of ['d2l-density-interval.svg', 'd2l-marginal.svg', 'd2l-likelihood-coin.svg',
      'd2l-coin-frequency-convergence.svg']) expect(html).toContain(figure);
    expect(html).not.toContain('mml-discrete-continuous-density.png');
    expect(html).toContain('mml-book.pdf#page=188');
    for (const anchor of ['from-interval-probabilities-to-a-density',
      'cumulative-distribution-function', 'marginalization-for-a-joint-density',
      'maximum-likelihood-for-continuous-variables', 'one-observation-from-an-exponential-distribution'])
      expect(html).toContain('id="' + anchor + '"');
    for (const value of ['stable_log_probability', '0.863422', '715', 'априорной',
      'маргинализац', 'формула Байеса'])
      expect(html.toLowerCase(), value).toContain(value.toLowerCase());
    for (const anchor of ['краткие-итоги', 'независимость-нельзя-предполагать-по-привычке',
      'вероятность-последовательности', 'максимум-правдоподобия-монеты'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('id="условное-правдоподобие-что-считается-входом"');
    expect(html).toContain('conditional_likelihood');
    expect(html).toContain('conditional_nll');
    expect(html).toContain('answer_nll');
    expect(html).toContain('alt="Под кривой плотности');
    const sequenceLoss = html.indexOf('id="nll-последовательности-и-средняя-потеря-на-токен"');
    const continuous = html.indexOf('id="непрерывные-распределения-и-правдоподобие"');
    expect(sequenceLoss).toBeGreaterThan(-1);
    expect(continuous).toBeGreaterThan(sequenceLoss);
    expect(html).toContain('/textbook/ml-foundations/derivatives-gradients/');
    expect(html).toContain('/textbook/ml-foundations/loss-functions/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the worked loss derivations, masks and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/loss-functions/index.html'), 'utf8');
    expect(html).toContain('id="class-weights-and-the-meaning-of-mean"');
    expect(html).toContain('weighted_elements');
    expect(html).toContain('scaled_probability_mean');
    expect(html).toContain('lm_nll');
    expect(html).toContain('answer_mask');
    expect(html.indexOf('id="маски-и-усреднение-по-токенам"'))
      .toBeLessThan(html.indexOf('id="градиент-по-логитам"'));
    expect(html).toContain('alt="Полносвязный слой');
    for (const figure of ['google-softmax-multiclass.png',
      'cs231n-pixelspace-original.jpeg', 'google-mse-outliers.png',
      'google-mae-outliers.png', 'd2l-gaussian-densities.svg', 'd2l-softmaxreg.svg',
      'margin.jpg', 'svmvssoftmax.png'])
      expect(html).toContain(figure);
    for (const code of ['def log_softmax', 'token_mean', 'per_example', 'mask.any', '2.04035515280017'])
      expect(html).toContain(code);
    expect(html).toContain('id="полный-шаг-по-параметрам-классификатора"');
    expect(html).toContain('id="hinge-loss-отступ-между-оценками-классов"');
    expect(html).toContain('id="одни-логиты-разные-функции-потерь"');
    for (const code of ['multiclass_hinge', 'diagram_W', 'diagram_scores', 'comparison_ce'])
      expect(html).toContain(code);
    expect(html).toContain('Ehsan Adeli');
    expect(html).not.toContain('Fei-Fei Li, Ranjay Krishna, Danfei Xu');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    expect(html).toContain('id="почему-mse-даёт-среднее-а-mae--медиану"');
    for (const code of ['temperatures', 'mse_values', 'mae_values', 'even_sample'])
      expect(html).toContain(code);
    for (const code of ['W_new', 'b_new', 'loss_new', '1.21178042875392'])
      expect(html).toContain(code);
    for (const anchor of ['почему-градиент-softmax-cross-entropy-особенно-прост', 'краткие-итоги',
      'градиент-по-логитам', 'маски-и-усреднение-по-токенам', 'бинарная-cross-entropy'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('/textbook/ml-foundations/probability-likelihood-logarithms/');
    expect(html).toContain('/textbook/ml-foundations/train-validation-test/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the licensed softmax figure and keeps both CS231n slides as source links', async () => {
    const chapter = 'textbook/ml-foundations/loss-functions/index.html';
    const dir = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/foundations-source-first');
    const retired = ['cs231n-softmax-cross-entropy.png', 'cs231n-linear-decision-boundaries.png'];
    for (const locale of ['', 'en']) {
      const html = await readFile(join(distDir, locale, chapter), 'utf8');
      expect(html).toMatch(/<img[^>]*src="[^"]*\/google-softmax-multiclass\.png"[^>]*alt="[^"]*груши/);
      for (const fact of ['Multi-class classification, Figure 9', 'CC BY 4.0',
        'cs231n.stanford.edu/slides/2025/lecture_2.pdf#page=75',
        'cs231n.stanford.edu/slides/2025/lecture_2.pdf#page=61',
        'cs231n-pixelspace-original.jpeg', 'Линия равенства двух логитов'])
        expect(html).toContain(fact);
      expect(html).not.toContain('class="katex-error');
      for (const name of retired) expect(html).not.toContain(name);
      if (locale) expectCanonicalEditionNotice(html);
    }
    for (const name of retired) await expect(access(join(dir, name))).rejects.toMatchObject({ code: 'ENOENT' });
    const png = await readFile(join(dir, 'google-softmax-multiclass.png'));
    expect(createHash('sha256').update(png).digest('hex')).toBe('7b0f01cc61b917b9c76a698af949c21acce117dad1d46247f1fe028555f3ad2c');
  });

  it('renders the reviewed data split chapter with original diagrams and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/train-validation-test/index.html'), 'utf8');
    for (const figure of ['train-validation-test.png', 'model-selection-loop.svg', 'google-hashing-split.gif',
      'mooc-cross-validation-train-test.png', 'sklearn-group-kfold.png', 'sklearn-time-series-split.png'])
      expect(html).toContain(figure);
    for (const anchor of ['три-части--три-разных-вопроса', 'цикл-разработки', 'кросс-валидация',
      'вложенная-кросс-валидация', 'почему-случайный-split-подходит-не-всегда', 'data-leakage',
      'каким-должен-быть-test-set', 'llm-одного-held-out-loss-недостаточно', 'краткие-итоги'])
      expect(html).toContain('id="' + anchor + '"');
    for (const mechanism of ['fold_losses', 'split_group', 'Вложенная кросс-валидация', '0.366', '0.00949',
      'Выбор и проверка фильтра спама', 'незнакомом адресе', '0.9465', 'Подбор преобразований на train'])
      expect(html).toContain(mechanism);
    expect(html).toContain('alt="Четыре фолда GroupKFold');
    expect(html).toContain('id="отбор-признаков-высокая-accuracy-на-случайных-метках"');
    for (const code of ['SelectKBest', 'HistGradientBoostingClassifier', 'X_train_selected', 'make_pipeline'])
      expect(html).toContain(code);
    expect(html).toContain('common_pitfalls.rst#data-leakage-during-pre-processing');
    expect(html).toContain('/textbook/ml-foundations/loss-functions/');
    expect(html).toContain('/textbook/ml-foundations/generalization-regularization/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed generalization chapter with derivations and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/generalization-regularization/index.html'), 'utf8');
    for (const figure of ['mooc-polynomial-fits.svg', 'mooc-underfit-resamples.svg', 'mooc-overfit-resamples.svg',
      'google-underfit-fit-overfit.svg', 'generalization-curves.png', 'sklearn-tree-bagging-bias-variance.png',
      'google-complex-model.svg', 'google-simple-model.svg', 'google-high-lambda.svg', 'google-low-lambda.svg'])
      expect(html).toContain(figure);
    for (const anchor of ['underfitting-fitting-и-overfitting', 'bias-и-variance-как-диагностическая-модель',
      'регуляризация-меняет-предпочтение-между-решениями', 'weight-decay', 'early-stopping', 'dropout',
      'больше-и-лучше-данных', 'оценка-должна-содержать-неопределённость',
      'почему-масштаб-llm-не-отменяет-overfitting', 'краткие-итоги', 'источники'])
      expect(html).toContain('id="' + anchor + '"');
    for (const example of ['Polynomial.fit', 'bootstrap_means', 'empirical_quantile', '0.634624', '0.9998', '1.93'])
      expect(html).toContain(example);
    for (const example of ['ridge_errors', '0.006555', '0.050435', 'Ранняя остановка', 'метрику выбирают до проверки'])
      expect(html).toContain(example);
    expect(html).toContain('id="почему-остановка-может-уменьшить-ошибку-на-новых-данных"');
    for (const example of ['early_path', 'best_step', '0.28125', '0.1953125'])
      expect(html).toContain(example);
    expect(html).toContain('/textbook/ml-foundations/train-validation-test/');
    expect(html).toContain('/textbook/classical-ml/predictive-pipeline/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('publishes the original bagging experiment in both routes without serving the retired CS229 crop', async () => {
    const chapter = 'textbook/ml-foundations/generalization-regularization/index.html';
    const figureDir = join(distDir, 'assets', '00 Учебник/Assets/Figures/curated/foundations-source-first');
    for (const prefix of ['', 'en']) {
      const html = await readFile(join(distDir, prefix, chapter), 'utf8');
      const example = html.match(/<a id="bias-variance-bagging-example"><\/a>[\s\S]*?(?=<table>)/)?.[0];
      expect(example).toBeDefined();
      expect(example).toMatch(/<img[^>]*src="[^"]*\/sklearn-tree-bagging-bias-variance\.png"[^>]*alt="[^"]*бэггинг/);
      for (const term of ['0.0152', '0.0092', '0.0255', '0.0196', 'BSD-3-Clause',
        'scikit-learn.org/1.9/auto_examples/ensemble/plot_bias_variance.html'])
        expect(example).toContain(term);
      expect(html).toContain('https://cs229.stanford.edu/summer2023/BiasVarianceAnalysis.pdf#page=7');
      expect(html).not.toContain('cs229-bias-variance-decomposition.png');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    await expect(access(join(figureDir, 'cs229-bias-variance-decomposition.png'))).rejects.toMatchObject({ code: 'ENOENT' });
    const bytes = await readFile(join(figureDir, 'sklearn-tree-bagging-bias-variance.png'));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe('8fa36b8ac57a9fdfa12f7856d4d55fb70f82dc09d62f8b1a70604473e06a3a08');
  });

  it('renders the reviewed linear models with full derivations and original plots', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/linear-models/index.html'), 'utf8');
    for (const figure of ['mooc-residuals.svg', 'mooc-regression-plane.svg', 'mooc-sigmoid.svg',
      'mooc-logistic-surface.svg', 'mooc-multiclass.svg', 'mooc-nonseparable.svg',
      'mooc-ridge-0.svg', 'mooc-ridge-50.svg', 'mooc-ridge-500.svg',
      'mooc-interpolation-overfit.svg', 'sklearn-polynomial-spline-basis.png',
      'sklearn-polynomial-basis.png', 'sklearn-spline-basis.png',
      'mooc-logistic-boundary.svg', 'mooc-binning-classification.png',
      'mooc-splines-classification.png', 'mooc-polynomial-classification.png'])
      expect(html, figure).toContain(figure);
    for (const anchor of ['linear-regression', 'logistic-regression', 'regularization',
      'геометрия-обучения-и-роль-масштаба', 'мотивирующий-пример-цена-квартиры',
      'нелинейность-через-подготовку-признаков', 'от-оценки-к-классификационному-решению',
      'l1-l2-и-цена-разреженности'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['lstsq', 'gradient_scores', 'loss_after', 'class_probabilities', 'GridSearchCV',
      'minimum_norm', 'residual_sums', 'matrix_rank'])
      expect(html).toContain(code);
    expect(html).toContain('https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html');
    for (const detail of ['raw_intercept', 'standardized_point', 'bin_encoded', 'joint_model',
      'id="coefficients-after-standardization"', 'id="binning-splines-and-feature-interactions"'])
      expect(html).toContain(detail);
    expect(html).toContain('/textbook/classical-ml/predictive-pipeline/');
    expect(html).toContain('/textbook/classical-ml/decision-trees/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    const license = await readFile(join(distDir, 'licenses/scikit-learn-1.9.1-BSD-3-Clause.txt'), 'utf8');
    expect(license).toContain('Copyright (c) 2007-2026 The scikit-learn developers.');
  });

  it('renders clustering mechanisms and real silhouette plots, not a flower', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/clustering/index.html'), 'utf8');
    for (const locale of ['', 'en/']) {
      const edition = await readFile(join(distDir, locale, 'textbook/classical-ml/clustering/index.html'), 'utf8');
      for (const detail of ['mooc-news-external-metrics.png', 'mooc-news-silhouette.png',
        'mooc-rfm-quantile-layers.png', 'why-the-metrics-select-different-numbers-of-clusters',
        'quantile-preprocessing-and-discrete-layers', 'purchase_counts', 'count_positions'])
        expect(edition, locale + detail).toContain(detail);
      expect(edition.match(/<pre\b/g)).toHaveLength(4);
      expect(edition).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(edition);
    }
    for (const name of ["elbow.png","silhouette.png","shapes-kmeans.png","shapes-hdbscan.png","dbscan-core-border.png","hdbscan-mst.png","hdbscan-condensed-selected.png","hdbscan-mutual-reachability.svg","lloyd-step-1.svg","lloyd-step-2.svg","lloyd-step-3.svg","lloyd-step-4.svg"]) expect(html, name).toContain(name);
    for (const anchor of ['k-means', 'предположения-k-means', 'как-оценивать-кластеризацию',
      'устойчивость', 'density-based-clustering', 'density-based-методы-на-практике',
      'механика-k-means-как-чередующаяся-оптимизация', 'как-сопоставлять-кластеры-между-запусками',
      'что-считать-успешным-результатом', 'что-нельзя-заключать-по-двумерной-картинке'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['core_sample_indices_', 'n_init', 'cross_val_score']) expect(html).toContain(code);
    for (const explanation of ['Квадраты остаются на прежних местах',
      'не результаты двух', 'Нормировка только наблюдений', 'Weston.pace', 'CC BY-SA 3.0'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/evaluation/');
    expect(html).toContain('/textbook/classical-ml/spectral-clustering/');
    expect(html).not.toContain('iris-silhouette.svg');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    const license = await readFile(join(distDir, 'licenses/hdbscan-BSD-3-Clause.txt'), 'utf8');
    expect(license).toContain('Copyright (c) 2015, Leland McInnes');
  });

  it('renders evaluation examples, real PR/ROC curves and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/evaluation/index.html'), 'utf8');
    for (const locale of ['', 'en/']) {
      const edition = await readFile(join(distDir, locale, 'textbook/classical-ml/evaluation/index.html'), 'utf8');
      for (const detail of ['target-transformation-and-residuals', 'regression-transformed-target.png',
        'raw_constant', 'log_constant', 'geometric mean', 'CC BY 4.0. Original English excerpt'])
        expect(edition, locale + detail).toContain(detail);
      expect(edition.match(/<pre\b/g)).toHaveLength(3);
      expect(edition).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(edition);
    }
    for (const figure of ["pr-curve.png","roc-curve.png","pr-roc.png","calibration-curve.png","calibration-legend.png","calibration.png","regression-actual.png","regression-residual.png","regression-residuals.png","nested-first-outer.png","nested-legend.png","group-kfold.png","time-series-split.png"]) expect(html, figure).toContain(figure);
    for (const anchor of ['cross-validation', 'nested-cross-validation', 'метрики-классификации',
      'score-probability-и-decision', 'calibration', 'regression-metrics',
      'порог-выбирается-по-out-of-fold-предсказаниям', 'минимальный-отчёт',
      'стратификация', 'новые-группы', 'прогноз-будущего'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['TunedThresholdClassifierCV', 'mean_cost', 'average_precision_score'])
      expect(html).toContain(code);
    for (const explanation of ['Зазор можно проследить на 12 объектах',
      'Три сравнения верны, одно нет', 'StratifiedGroupKFold',
      'Он не является способом калибровки или улучшения модели'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/ensembles/');
    expect(html).not.toContain('evaluation_quiz_precision_recall_C1.svg');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the ensemble mechanisms with real residual plots and former anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/ensembles/index.html'), 'utf8');
    for (const name of ['mooc-bagging-polynomial.png', 'xgboost-split-scan.png',
      'lightgbm-level-wise.png', 'lightgbm-leaf-wise.png']) expect(html, name).toContain(name);
    for (const detail of ['Bagging a complete pipeline', 'Histogram statistics and a scan over bins',
      'Choosing which leaf to split', 'poly_bag', 'hist_G', 'hist_gains']) expect(html).toContain(detail);
    const english = await readFile(join(distDir, 'en/textbook/classical-ml/ensembles/index.html'), 'utf8');
    for (const detail of ['Canonical-language chapter', 'no separate English edition yet',
      'mooc-bagging-polynomial.png', 'xgboost-split-scan.png', 'hist_gains',
      'lightgbm-leaf-wise.png']) expect(english).toContain(detail);
    for (const name of ["bagging-hundred-average.png","bagging-three-average.png","bagging-three-trees.png","bootstrap-sample.png","ensemble-boosting1.svg","ensemble-boosting2.svg","ensemble-boosting3.svg","gradient-fit-residuals.png","gradient-initial-legend.png","gradient-initial-plot.png","gradient-initial-residuals.png","gradient-residual-legend.png","gradient-residual-plot.png"]) expect(html, name).toContain(name);
    for (const anchor of ['bagging', 'random-forest', 'gradient-boosting', 'histogram-gradient-boosting',
      'bagging-и-boosting--не-одно-и-то-же', 'один-источник-ошибок-две-стратегии',
      'boosting-как-исправление-текущих-ошибок', 'histogram-boosting-и-вычислительный-компромисс',
      'практическая-настройка-и-диагностика'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['estimators_samples_', 'oob_prediction_', 'newton_model', 'stages'])
      expect(html).toContain(code);
    for (const explanation of ['Начальный прогноз тоже выбирают по функции потерь',
      'Начальный логит не обязан равняться нулю', 'До первого дерева все объекты получают вероятность',
      'Голоса с весами', 'В строке AdaBoost приведён разобранный выше бинарный классификатор'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/decision-trees/');
    expect(html).toContain('/textbook/classical-ml/evaluation/');
    expect(html).not.toContain('boosting_iter_orange');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed decision-tree mechanisms, examples, and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/decision-trees/index.html'), 'utf8');
    for (const name of ['mooc-asymmetric-depth2.png', 'mooc-asymmetric-depth6.png',
      'mooc-asymmetric-tree6.png', 'mooc-tree-linear-extrapolation.png'])
      expect(html, name).toContain(name);
    for (const detail of ['Extrapolation beyond the training range', 'Branches of different depths',
      'asym_models', 'linear_small', 'not to estimate test error']) expect(html).toContain(detail);
    const english = await readFile(join(distDir, 'en/textbook/classical-ml/decision-trees/index.html'), 'utf8');
    for (const detail of ['Canonical-language chapter', 'no separate English edition yet',
      'Non-parametric does not mean assumption-free', 'mooc-asymmetric-depth6.png',
      'mooc-tree-linear-extrapolation.png', 'Джини после разбиения']) expect(english).toContain(detail);
    for (const name of ["dt_fit.svg","dt_overfit.svg","dt_underfit.svg","tree2D_1split.svg","tree2D_2split.svg","tree2D_3split.svg","tree_example.svg","tree-example-tight.svg","tree_regression1.svg","tree_regression2.svg","tree_regression4.svg","penguins-tree.png","penguins-probabilities.png","permutation-predictive-feature.png"])
      expect(html, name).toContain(name);
    for (const anchor of ['classification-tree', 'regression-tree', 'размер-дерева-и-generalization',
      'как-дерево-принимает-решение-один-узел-вручную', 'почему-глубокое-дерево-запоминает-данные',
      'рабочий-конвейер', 'интерпретация-без-самообмана'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['cost_complexity_pruning_path', 'ccp_alphas', 'min_impurity_decrease', 'predict_proba',
      'duplicate_stump', 'X_duplicate', 'y_check'])
      expect(html).toContain(code);
    expect(html).toContain('/textbook/classical-ml/linear-models/');
    expect(html).toContain('/textbook/classical-ml/ensembles/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed predictive pipeline with source diagrams and former anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/predictive-pipeline/index.html'), 'utf8');
    for (const figure of ['mooc-predictor-fit.svg', 'mooc-predictor-predict.svg', 'mooc-predictor-score.svg',
      'mooc-transform.svg', 'mooc-columns.svg', 'mooc-pipeline-fit.svg', 'mooc-pipeline-predict.svg',
      'mooc-transformer-fit.svg', 'mooc-transformer-fit-transform.svg',
      'mooc-adult-histograms.png', 'mooc-adult-pairplot.png'])
      expect(html, figure).toContain(figure);
    for (const anchor of ['baseline-задаёт-смысл-метрике', 'train-validation-и-test--разные-полномочия',
      'полный-цикл', 'сквозной-пример-кому-позвонить-сегодня', 'смешанные-типы-данных-без-утечки',
      'почему-предварительная-обработка-входит-в-модель', 'от-функции-потерь-к-качеству-решения'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('mean_before');
    expect(html).toContain('sparse_output');
    expect(html).toContain('id="от-обученных-коэффициентов-к-вероятности"');
    for (const example of ['logits_from_parts', 'positive_probability', 'positive_column', '0.662584'])
      expect(html).toContain(example);
    for (const example of ['ordered_sizes', 'stump_scores', 'no_rare_bucket', 'infrequent_if_exist',
      'Original MOOC explanation resumes', 'first 5,000 rows'])
      expect(html).toContain(example);
    expect(html).toContain('/textbook/classical-ml/linear-models/');
    expect(html).toContain('/textbook/ml-foundations/generalization-regularization/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders executable tokenization examples and the limits of reversibility', async () => {
    const html = await readFile(join(distDir, 'textbook/text-representation/tokenization/index.html'), 'utf8');
    for (const figure of ['tokenized-example.png', 'hf-tokenization-pipeline.svg', 'tokenizer-choice-figure1.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['train_bpe', 'unicodedata', 'byte_fallback', '109/53', '48/109',
      'e5c5054474f583d6d9499624649353995d63c70a', 'Как Unigram обучает вероятности'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['tokenizer-contract', 'bpe-merge-trace', 'tokenizer-evidence'])
      expect(html).toContain(`id="${alias}"`);
    for (const example of ['Почему недостаточно искать самый длинный токен',
      'a · b · cd', 'a · ##bc', 'eedf5716ce1268e56f0a50264a88cafad334ac61',
      '524 288 000', '750 MiB', '9 437 184', 'weight tying'])
      expect(html, example).toContain(example);
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders source-faithful word-vector data functions and all mechanism anchors in both locales', async()=>{
    const journal=JSON.parse(await readFile(join(rootDir,'docs/audit/2026-11-03-word-embeddings-data-additions.json'),'utf8'));
    const source=await readFile(join(rootDir,journal.files[0].source),'utf8');
    const anchors=[...source.matchAll(/^    anchor: (emb-[a-z-]+)$/gm)].map(m=>m[1]);
    expect(anchors).toHaveLength(25);
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale+'textbook/text-representation/word2vec-glove/index.html'),'utf8');
      for(const id of [...anchors,'краткие-итоги'])expect(html.split('id="'+id+'"')).toHaveLength(2);
      expect((html.match(/data-language="python"/g)??[]).length).toBe(7);
      for(const phrase of ['get_centers_and_contexts','random.randint','batchify',
        'contexts_negatives','keep','binary_cross_entropy_with_logits','16/11'])
        expect(html).toContain(phrase);
      expect((html.match(/<img\b[^>]*\bsrc="\/assets\//g)??[]).length).toBe(9);
      expect(html).toContain('Полное, тонкое, компактное и усечённое SVD матрицы 4 на 3');
      expect(html).not.toContain('class="katex-error');
      expect(html).toContain('/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt');
      if(locale)expect(html).toContain('Canonical-language chapter.');
    }
  });

  it('renders attributed one-hot, SVD and RLHF originals and excludes the three retired files', async () => {
    const figures = [
      ['textbook/text-representation/numerical-representation', 'tensorflow-one-hot.png',
        'tensorflow.org/text/guide/word_embeddings#one-hot_encodings', 'creativecommons.org/licenses/by/4.0/'],
      ['textbook/text-representation/word2vec-glove', 'reduced-svd.svg',
        'commons.wikimedia.org/wiki/File:Reduced_Singular_Value_Decompositions.svg', 'creativecommons.org/licenses/by-sa/4.0/'],
      ['textbook/post-training/sft-instruction-data', 'rlhf-basic.png',
        'rlhfbook.com/c/03-training-overview#fig:rlhf-basic-repeat', '/licenses/rlhf-book-MIT.txt'],
      ['research/alignment-rlhf-dpo-rlvr', 'rlhf-basic.png',
        'arxiv.org/html/2203.02155v1#A1.T6', '/licenses/rlhf-book-MIT.txt'],
    ] as const;
    for (const [route, file, source, license] of figures) for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale + route, 'index.html'), 'utf8');
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + file.replaceAll('.', '\\.') + '"'));
      expect(html).toContain(source);
      expect(html).toContain(license);
      expect(html).not.toMatch(/<img[^>]+src="[^"]*\/(?:chatgpt-training-pipeline|count-based-idea|one-hot)\.png"/);
    }
    for (const path of ['instructgpt/chatgpt-training-pipeline.png', 'voita-word-embeddings/count-based-idea.png', 'voita-word-embeddings/one-hot.png'])
      await expect(readFile(join(distDir, 'assets/00 Учебник/Assets/Figures/curated', path))).rejects.toMatchObject({ code: 'ENOENT' });
  });
  it('renders the interactive BPE lab, original paper figure, and reciprocal links in both locales', async () => {
    for (const lang of ['', 'en/']) {
      const lab = await readFile(join(distDir, lang, 'practice/byte-pair-encoding/index.html'), 'utf8');
      const chapter = await readFile(join(distDir, lang, 'textbook/text-representation/tokenization/index.html'), 'utf8');
      const index = await readFile(join(distDir, lang, 'practice/index.html'), 'utf8');
      for (const id of ['bpe-lab', 'bpe-training', 'bpe-probe', 'bpe-limit', 'bpe-next', 'bpe-back', 'bpe-decoded'])
        expect(lab).toContain('id="' + id + '"');
      expect(lab).toContain('href="#bpe-lab"');
      expect(lab).toContain('209 143 240 159 166 138');
      expect(lab).toContain('258 100 258 97 99');
      expect(lab).toContain('aria-live="polite"');
      expect(lab).toContain('<noscript>');
      expect(lab.match(/<h1(?:\s|>)/g)).toHaveLength(1);
      expect(lab).toContain('href="/' + lang + 'textbook/text-representation/tokenization/#bpe-merge-trace"');
      expect(chapter).toContain('href="/' + lang + 'practice/byte-pair-encoding/"');
      expect(index).toContain('href="/' + lang + 'practice/byte-pair-encoding/"');
      expect(chapter).toContain('zouhar-bpe-figure1.png');
      expect(chapter).toContain('2023.findings-acl.38v1.pdf#page=3');
      expect(chapter).not.toContain('class="katex-error');
      if (lang) expect(lab).not.toContain('Canonical-language chapter.');
    }
  });

  it('renders word-vector objectives, shared-table gradients and original author figures', async () => {
    const html = await readFile(join(distDir, 'textbook/text-representation/word2vec-glove/index.html'), 'utf8');
    for (const figure of ['reduced-svd.svg', 'd2l-skip-gram.svg', 'd2l-cbow.svg',
      'alammar-word2vec-lookup-embeddings.png', 'd2l-hi-softmax.svg',
      'glove-table1-ratios.png', 'glove-figure1-weighting.png',
      'bert-feature-extraction-contextualized-embeddings.png', 'gpt2-output-projection-2.png'])
      expect(html, figure).toContain(figure);
    for (const retired of ['count-based-idea.png', 'cbow-skipgram.png', 'negative-sampling.png', 'glove-idea.png', 'glove-loss.png'])
      expect(html, retired).not.toContain(retired);
    for (const attribution of ['word2vec.html#the-skip-gram-model', 'word2vec.html#the-continuous-bag-of-words-cbow-model',
      'illustrated-word2vec/#word2vec-training-process', 'D14-1162.pdf#page=3', 'D14-1162.pdf#page=4',
      'word_embeddings.html#w2v_negative_sampling', 'word_embeddings.html#w2v_skipgram_cbow'])
      expect(html, attribution).toContain(attribution);
    for (const mechanism of ['input_table', 'output_table', 'separate_loss',
      '1.175168994008524', 'getWordVector', '5b5943c118b0ec5fb9cd8d20587de2b2d3966dfe'])
      expect(html, mechanism).toContain(mechanism);
    for (const anchor of ['краткие-итоги', 'два-вклада-в-градиент-общей-таблицы'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).toContain('/textbook/text-representation/tokenization/');
    expect(html).toContain('/textbook/language-modeling/next-token-probability/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('shows the real GPT-2 embedding table with a separate link to the original lookup animation', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/text-representation/numerical-representation/index.html'), 'utf8');
      expect(html).toContain('gpt2-token-embeddings-wte-2.png');
      expect(html).toContain('illustrated-gpt2/#input-encoding');
      expect(html).toContain('word_embeddings.html#intro');
      expect(html).not.toContain('lookup-table.gif');
      expect(html).not.toContain('class="katex-error');
      expect((html.match(/data-language="python"/g) ?? []).length).toBe(5);
    }
  });

  it('publishes bound approval, its queue semantics and the original outbox diagram', async () => {
    const route = '/textbook/agents/security/';
    for (const prefix of ['', '/en']) {
      const html = await readFile(join(distDir, prefix, route, 'index.html'), 'utf8');
      const reference = await readFile(join(distDir, prefix, 'reference/index.html'), 'utf8');
      for (const id of ['agent-security-bound-approval', 'agent-security-transactional-outbox']) {
        expect(html).toContain(`id="${id}"`);
        expect(reference).toContain(`href="${prefix}${route}#${id}"`);
      }
      for (const value of ['approval_mismatch', 'draft_changed', 'grant_changed',
        'BEGIN IMMEDIATE', 'before_commit', 'after_commit', 'OrderCreated'])
        expect(html).toContain(value);
      expect(html).toMatch(/<img[^>]+src="[^"]*\/microsoft-transactional-outbox\.svg"/);
      expect(html).toContain('https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match');
      if (prefix) expectCanonicalEditionNotice(html);
    }
    const note = await readFile(join(distDir, 'sources/implementations/conditional-actions-outbox/index.html'), 'utf8');
    expect(note).toContain(`href="${route}#agent-security-bound-approval"`);
  });

  it('renders agent-security mechanisms, version distinctions and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/security/index.html'), 'utf8');
    for (const figure of ['m12-p13.png', 'm12-p27.png', 'm12-p48.png',
      'agentpoison-overview.png', 'm12-p65.png',
      '12-dawn-song-privilege-control-sequence.png', 'm12-p90.png',
      'datasentinel-detection.png', 'datasentinel-training.svg'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['sqlite3', 'decide', 'frozenset',
      '2504.11703v1', '2504.11703v3', 'Algorithm 4', 'ASR-r', 'TOCTOU', 'FNR 0,87'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['safety-security-and-the-hybrid-system-boundary',
      'threat-model-before-defenses', 'direct-indirect-and-retrieval-memory-poisoning',
      'progent-least-privilege-as-an-executable-tool-call-policy',
      'monitoring-information-flow-datasentinel-and-formal-policy-verification'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/textbook/agents/tool-use/');
    expect(html).toContain('/textbook/agents/scientific-discovery/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders TTC selection, upstream search details, cost derivation and source figures', async () => {
    const html = await readFile(join(distDir,'textbook/reasoning/test-time-compute/index.html'),'utf8');
    for (const figure of ['snell-prm-search-methods.png','opro-loop.png','self-discover-stages.png',
      'self-consistency.png','m01-p42-sample-scaling.png','usc-workflow.png',
      'snell-revisions-parallel.png','s1-budget-forcing.png','tot-search.png','tot-game24.png',
      'snell-budget-difficulty.png','snell-sequential-parallel-ratio.png'])
      expect(html,figure).toContain(figure);
    for (const mechanism of ['weighted_vote','expand_frontier','inference_multiplier',
      'get_value','n_generate_sample','21.001','Universal Self-Consistency',
      '8050e67d0e3a0fddc424d7fa5801538722a4c4cc','38%'])
      expect(html,mechanism).toContain(mechanism);
    for (const anchor of ['декомпозиция-и-выбор-reasoning-модулей',
      'budget-forcing-заставить-траекторию-продолжиться','поиск-по-промежуточным-состояниям'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).toContain('/textbook/post-training/reasoning-distillation/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('does not swallow textbook prose into malformed math blocks', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) =>
        section.pages.map((page) => page.route).filter((route) => route.startsWith('textbook/'))
    );
    const broken: string[] = [];
    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      for (const locale of ['', 'en']) {
        const html = await readFile(join(distDir, locale, outputRoute, 'index.html'), 'utf8');
        if (html.includes('class="katex-error"')) broken.push(`${locale}/${outputRoute}`);
      }
    }
    expect(broken).toEqual([]);
  });

  it('includes previous and next links on an interior handbook page', async () => {
    const html = await readFile(
      join(distDir, 'textbook', 'transformer', 'self-attention', 'index.html'),
      'utf8'
    );
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
  });

  it('retains accessible formulas but excludes duplicate MathML and TeX from search', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/models/gpt-3/index.html'), 'utf8');
      const mathml = [...html.matchAll(/<span[^>]*class="katex-mathml"[^>]*>/g)];
      expect(mathml.length).toBeGreaterThan(10);
      for (const span of mathml) expect(span[0]).toContain('data-pagefind-ignore="all"');
      expect(html).toContain('<annotation encoding="application/x-tex">');
      expect(html).toContain('class="katex-html"');
    }
  });

  it('delivers complete lecture PDFs from official origins without removing their course pages', async () => {
    const cases = [
      ['stanford-cs336-spring-2026', 'Lectures/repository/lecture_09.pdf',
        'https://raw.githubusercontent.com/stanford-cs336/lectures/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf'],
      ['berkeley-advanced-llm-agents-spring-2025', 'Lectures/meeting-04-slides.pdf',
        'https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf']
    ];
    for (const [course, pdf, official] of cases) {
      for (const locale of ['', 'en/']) {
        const html = await readFile(join(distDir, locale, 'sources/courses', course, 'index.html'), 'utf8');
        expect(html).toContain(`href="${official}"`);
      }
      await expect(access(join(distDir, 'sources/courses', course, pdf))).rejects.toMatchObject({ code: 'ENOENT' });
      await expect(access(join(rootDir, '05 Источники/Courses', course === 'stanford-cs336-spring-2026'
        ? 'Stanford CS336 Spring 2026' : 'Berkeley Advanced LLM Agents Spring 2025', pdf))).resolves.toBeUndefined();
    }
  });

  it('keeps the full course original and an EN alias that retains old section addresses', async () => {
    const route = 'sources/courses/harvard-ml-systems/tinytorch/19-benchmarking';
    const original = await readFile(join(distDir, route, 'index.html'), 'utf8');
    const alias = await readFile(join(distDir, 'en', route, 'index.html'), 'utf8');
    expect(original).toContain('# jupyter:');
    expect(original).toContain('id="complete-python-source"');
    expect(alias).toContain(`0;url=/${route}/`);
    expect(alias).toContain(`href="/${route}/"`);
    expect(alias).toContain('id="complete-python-source"');
    expect(alias).toContain('location.search + location.hash');
    expect(alias).toContain('noindex');
    expect(alias).not.toContain('# jupyter:');
  });

  it('writes a Russian Pagefind index', async () => {
    await expect(access(join(pagefindDir, 'pagefind.js'))).resolves.toBeUndefined();
    const entry = await readFile(join(pagefindDir, 'pagefind-entry.json'), 'utf8');
    expect(entry).toContain('ru');
  });

  it('finds the attention chapter using its indexed Transformer terminology', async () => {
    const search = await pagefind.search('Transformer');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/transformer/self-attention/')))
      .toBe(true);
  });

  it('stems Russian inflections in the generated index', async () => {
    const search = await pagefind.search('нейронов');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/neural-networks/backpropagation/')))
      .toBe(true);
  });
});
