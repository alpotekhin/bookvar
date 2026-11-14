import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { expect } from 'vitest';

export async function verifyCourseLibraryOriginals(root: string, dist: string): Promise<void> {
  const route = 'sources/courses/';
  const groups = {
    'lf-recommenders': ['als-operationalization', 'sar-deep-dive', 'ncf-deep-dive'],
    'machine-learning-visualized': ['overview', 'complete-book', 'interactive-linear-regression',
      'interactive-logistic-regression', 'interactive-perceptron'],
    'stanford-cs230-cheatsheets': ['overview', 'cnn', 'rnn', 'tips-and-tricks', 'complete'],
  };
  const evidence = JSON.parse(await readFile(join(root, 'docs/audit/2026-11-13-course-library-media-additions.json'), 'utf8')) as {
    sources: Record<string, { objects: [string, string, string, string, number?][] }>;
    previews: [string, number, string][];
  };
  const hashes = new Map(Object.values(evidence.sources).flatMap(group =>
    group.objects.map(row => [row[1].split('/').at(-1)!, row[3]] as const)));
  const mediaByPage: Record<string, string[]> = {
    'als-operationalization': ['linux-foundation-recommenders-architecture.png'],
    'sar-deep-dive': ['sar-schema.svg'],
    'ncf-deep-dive': ['NCF.svg'],
    'machine-learning-visualized/overview': ['home.gif', 'marimo.gif', 'latex.gif'],
    'complete-book': ['main.pdf'],
    cnn: ['cheatsheet-convolutional-neural-networks.pdf'],
    rnn: ['cheatsheet-recurrent-neural-networks.pdf'],
    'tips-and-tricks': ['cheatsheet-deep-learning-tips-tricks.pdf'],
    complete: ['super-cheatsheet-deep-learning.pdf'],
  };
  const marimoByPage: Record<string, string> = {
    'interactive-linear-regression': '9hsfob',
    'interactive-logistic-regression': 'cs19cb',
    'interactive-perceptron': '50ccg0',
  };
  for (const [group, pages] of Object.entries(groups)) for (const slug of pages) {
    const path = route + group + '/' + slug + '/';
    const html = await readFile(join(dist, path, 'index.html'), 'utf8');
    const alias = await readFile(join(dist, 'en', path, 'index.html'), 'utf8');
    expect(alias).toContain('/' + path);
    expect(html).not.toMatch(/<(?:img|iframe|video|source)\b[^>]*\bsrc="(?:https?:)?\/\//i);
    const sources = [...html.matchAll(/<(img|iframe)\b[^>]*\bsrc="([^"]+)"[^>]*>/g)];
    for (const file of mediaByPage[group + '/' + slug] ?? mediaByPage[slug] ?? []) {
      if (file.endsWith('.pdf')) {
        const preview = evidence.previews.find(([path]) => path.split('/').at(-1)!.startsWith(file.slice(0, -4) + '-page-'));
        expect(preview, path).toBeDefined();
        const [png, page, hash] = preview!;
        const previewUrl = '/' + png.replace(/^site\/public\//, '');
        const pdfUrl = previewUrl.replace(/-page-\d+\.png$/, '.pdf');
        expect(html).toContain(`href="${pdfUrl}">Open the original PDF at full size`);
        expect(html).toContain(`href="${pdfUrl}#page=${page}"`);
        expect(html).toContain('Extracted document text');
        expect(html).not.toContain('<iframe');
        expect(sources.find(match => match[2] === previewUrl)?.[0]).toMatch(/\balt="[^"]+"/);
        expect(createHash('sha256').update(await readFile(join(dist, previewUrl))).digest('hex')).toBe(hash);
        expect(createHash('sha256').update(await readFile(join(dist, pdfUrl))).digest('hex')).toBe(hashes.get(file));
        continue;
      }
      const asset = sources.find(match => match[2].endsWith('/' + file));
      expect(asset, path).toBeDefined();
      const [tag, , url] = asset!;
      expect(createHash('sha256').update(await readFile(join(dist, decodeURIComponent(url)))).digest('hex'))
        .toBe(hashes.get(file));
      expect(tag).toMatch(/\balt="[^"]+"/);
    }
    if (slug in marimoByPage) {
      expect(html).toContain(`href="https://marimo.app/l/${marimoByPage[slug]}"`);
      expect(html).not.toContain('<iframe');
    }
    if (group === 'stanford-cs230-cheatsheets' && slug === 'overview') {
      for (const sheet of pages.slice(1)) expect(html).toContain(`href="/${route}${group}/${sheet}/"`);
    }
  }
}
