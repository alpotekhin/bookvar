import { readFileSync, readdirSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import config from '../astro.config.mjs';
import { normalizeArchiveMath } from '../../publishing/adapter/math';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validateConfig } from '../node_modules/astro/dist/core/config/validate.js';
import { markdownContentEntryType } from '../node_modules/astro/dist/vite-plugin-markdown/content-entry-type.js';

function configuredProcessor() {
  const processor = config.markdown?.processor;
  if (!processor) throw new Error('The site must configure a Markdown processor');
  return processor;
}

const cases = [
  ['harvard-ml-systems/labs/vol1/lab-10-model-compress', 'Harvard ML Systems/labs/vol1/lab_10_model_compress.md', 1],
  ['lf-recommenders/diversity-metrics', 'Linux Foundation Recommenders/examples/03_evaluate/als_movielens_diversity_metrics.ipynb.md', 3],
  ['hse-ml-course/ml1-2026-spring/seminars/sem09-trees-ipynb', 'HSE ML course/ml1-2026-spring/seminars/sem09-trees.ipynb.md', 2],
  ['hse-ml-course/ml1-2026-spring/homework-practice/homework-practice-05-trees/homework-practice-05-trees-ipynb', 'HSE ML course/ml1-2026-spring/homework-practice/homework-practice-05-trees/homework-practice-05-trees.ipynb.md', 1],
  ['hse-ml-course/ml2-2026-spring/homeworks-practice/homework-practice-11-random-features/homework-practice-11-random-features-ipynb', 'HSE ML course/ml2-2026-spring/homeworks-practice/homework-practice-11-random-features/homework-practice-11-random-features.ipynb.md', 2],
  ['scikit-learn-mooc/python-scripts/dev-features-importance', 'Scikit-learn MOOC/python_scripts/dev_features_importance.py.md', 1]
] as const;

describe('source archive math compatibility', () => {
  test.each([
    'lf-recommenders/diversity-metrics',
    'scikit-learn-mooc/python-scripts/dev-features-importance',
    'harvard-ml-systems/slides/vol1/00-course-overview'
  ])('uses Astro entry-info/render pipeline with the real generated file: %s', async route => {
    const fileUrl = new URL(`../src/content/docs/generated/sources/courses/${route}.md`, import.meta.url);
    const contents = readFileSync(fileUrl, 'utf8');
    const entry = await markdownContentEntryType.getEntryInfo({ contents, fileUrl });
    const resolved = await validateConfig(config, fileURLToPath(new URL('../', import.meta.url)), 'build');
    if (!markdownContentEntryType.getRenderFunction) throw new Error('Astro Markdown renderer is required');
    const render = await markdownContentEntryType.getRenderFunction(resolved);
    const result = await render({ id: route, data: entry.data, body: entry.body, filePath: fileURLToPath(fileUrl), digest: 'source-math-test' });
    expect((result?.html.match(/class="katex-error"/g) ?? []).length).toBe(0);
  });

  test('includes the normalizer module hash in Astro-serialized processor options', async () => {
    const source = readFileSync(new URL('../../publishing/adapter/math.ts', import.meta.url));
    const expected = createHash('sha256').update(source).digest('hex');
    const resolved = await validateConfig(config, fileURLToPath(new URL('../', import.meta.url)), 'build');
    expect(JSON.stringify(resolved.markdown.processor.options).includes(expected)).toBe(true);
    expect(createHash('sha256').update(source).update('// changed implementation').digest('hex')).not.toBe(expected);
  });
  const slidesRoot = new URL('../../05 Источники/Courses/Harvard ML Systems/slides/', import.meta.url);
  const decks = ['vol1', 'vol2'].flatMap(volume => readdirSync(new URL(volume, slidesRoot)).filter(name => name.endsWith('.md')).map(name => [volume, name]));
  test.each(decks)('restores exact retained TeX in %s/%s', async (volume, name) => {
    const body = readFileSync(new URL(`${volume}/${name}`, slidesRoot), 'utf8').replace(/^---\n[\s\S]*?\n---\n/, '');
    const url = new URL(`../src/content/docs/generated/sources/courses/harvard-ml-systems/slides/${volume}/${name.replaceAll('_', '-')}`, import.meta.url);
    const normalized = normalizeArchiveMath(body, url);
    expect(normalized === body).toBe(false);
    expect(normalized.slice(normalized.indexOf('## Complete original Beamer source')) === body.slice(body.indexOf('## Complete original Beamer source'))).toBe(true);
    expect(normalizeArchiveMath(normalized, url) === normalized).toBe(true);
    const canonicalUrl = new URL('../src/content/docs/generated/textbook/control.md', import.meta.url);
    expect(normalizeArchiveMath(body, canonicalUrl) === body).toBe(true);
    const headings = (text: string) => text.split('\n').filter(line => /^#{1,6} /.test(line));
    expect(headings(normalized)).toEqual(headings(body));
    const links = (text: string) => [...text.matchAll(/\]\((https?:\/\/[^)]+)\)/g)].map(match => match[1]);
    expect(links(normalized)).toEqual(links(body));
    expect(normalizeArchiveMath(body.replace('## Readable slide sequence\n\n', '## Readable slide sequence\n\nEditorial change\n\n'), url)).toBe(body.replace('## Readable slide sequence\n\n', '## Readable slide sequence\n\nEditorial change\n\n'));
    const renderer = await configuredProcessor().createRenderer({});
    const result = await renderer.render(body, { fileURL: url });
    const errors = [...result.code.matchAll(/<span class="katex-error"[^>]*title="([^"]*)"[^>]*>([\s\S]*?)<\/span>/g)].map(match => ({ error: match[1], sample: match[2].slice(0, 200) }));
    expect(errors).toEqual([]);
  });
  test.each(cases)('renders the complete frozen %s body without consuming prose', async (route, source, originalErrors) => {
    const body = readFileSync(new URL(`../../05 Источники/Courses/${source}`, import.meta.url), 'utf8')
      .replace(/^---\n[\s\S]*?\n---\n/, '') + '\n\nARCHIVE_TAIL_SENTINEL\n';
    const renderer = await configuredProcessor().createRenderer({});
    const before = await renderer.render(body, { fileURL: new URL(`../src/content/docs/generated/textbook/control.md`, import.meta.url) });
    expect((before.code.match(/class="katex-error"/g) ?? []).length).toBe(originalErrors);
    for (const locale of ['', 'en/']) {
      const result = await renderer.render(body, { fileURL: new URL(`../src/content/docs/generated/${locale}sources/courses/${route}.md`, import.meta.url) });
      expect((result.code.match(/class="katex-error"/g) ?? []).length).toBe(0);
      expect(result.code).toContain('class="katex"');
      expect(result.code.includes('<p>ARCHIVE_TAIL_SENTINEL</p>')).toBe(true);
      const normalized = normalizeArchiveMath(body, new URL(`../src/content/docs/generated/${locale}sources/courses/${route}.md`, import.meta.url));
      const fences = (text: string) => [...text.matchAll(/^```[^\n]*\n[\s\S]*?^```[ \t]*$/gm)].map(match => match[0]);
      expect(fences(normalized)).toEqual(fences(body));
      expect((result.code.match(/<pre/g) ?? []).length).toBe(fences(body).length);
      expect(normalizeArchiveMath(normalized, new URL(`../src/content/docs/generated/${locale}sources/courses/${route}.md`, import.meta.url))).toBe(normalized);
      if (source.startsWith('Harvard')) {
        const changed = body.replace('## Readable lab narrative\n\n', '## Readable lab narrative\n\nEditorial change\n\n');
        expect(normalizeArchiveMath(changed, new URL(`../src/content/docs/generated/${locale}sources/courses/${route}.md`, import.meta.url))).toBe(changed);
      }
    }
  });

  test('recovers mathematical operators from retained TeX, not from guessed repairs', () => {
    const body = readFileSync(new URL('vol2/05_distributed_training.md', slidesRoot), 'utf8');
    const url = new URL('../src/content/docs/generated/sources/courses/harvard-ml-systems/slides/vol2/05-distributed-training.md', import.meta.url);
    const readable = normalizeArchiveMath(body, url).split('## Complete original Beamer source')[0];
    expect(readable.includes(String.raw`g_k = \frac{1}{|B_k|} \sum_{x_i \in B_k} \nabla_\theta L(\theta, x_i)`)).toBe(true);
    expect(readable.includes(String.raw`g_k = {1}{|B_k|} _{x_i  B_k} _ L(, x_i)`)).toBe(false);
    const boundary = body.indexOf('## Complete original Beamer source');
    const mutated = body.slice(0, boundary) + body.slice(boundary).replaceAll(String.raw`\sum`, String.raw`\prod`);
    expect(normalizeArchiveMath(mutated, url) === mutated).toBe(true);
  });

  test('all retained Harvard payloads match the independently frozen importer manifest', () => {
    const manifest = JSON.parse(readFileSync(new URL('../../05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json', import.meta.url), 'utf8')) as Array<{ kind: string; route: string; output: string; sha256: string }>;
    const sources = manifest.filter(entry => entry.kind === 'slides' || entry.route.endsWith('/lab-10-model-compress'));
    expect(sources.length).toBe(36);
    for (const entry of sources) {
      const body = readFileSync(new URL(`../../${entry.output}`, import.meta.url), 'utf8');
      const original = body.match(/## Complete original (?:Beamer|Marimo) source\n\n```(?:tex|python)\n([\s\S]*?)\n```/)?.[1];
      expect(createHash('sha256').update(original + '\n').digest('hex')).toBe(entry.sha256);
    }
  });

  test('does not normalize canonical pages, unrelated archives, missing URL or ungenerated paths', () => {
    const body = String.raw`$\textrm{reco_df}$`;
    for (const path of [
      '../src/content/docs/generated/reference/inference/quantization.md',
      '../src/content/docs/generated/en/textbook/example.md',
      '../src/content/docs/generated/sources/courses/harvard-ml-systems/slides/vol1/00-course-overview.md',
      '../src/content/docs/generated/sources/courses/lf-recommenders/diversity-metrics-extra.md',
      '../src/content/docs/sources/courses/lf-recommenders/diversity-metrics.md'
    ]) expect(normalizeArchiveMath(body, new URL(path, import.meta.url))).toBe(body);
    expect(normalizeArchiveMath(body)).toBe(body);
  });

  test.each([
    '`$\\textrm{reco_df}$`',
    '``literal ` $\\textrm{reco_df}$``',
    '```tex\n$\\textrm{reco_df}$\n```',
    '~~~~tex\n```\n$\\textrm{reco_df}$\n```\n~~~~',
    '````tex\n```\n$\\textrm{reco_df}$\n```\n````',
    '```tex\n$\\textrm{reco_df}$'
  ])('preserves literal source code: %s', body => {
    const url = new URL('../src/content/docs/generated/sources/courses/lf-recommenders/diversity-metrics.md', import.meta.url);
    expect(normalizeArchiveMath(body, url)).toBe(body);
  });
});
