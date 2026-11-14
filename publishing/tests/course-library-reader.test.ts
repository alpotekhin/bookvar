import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { courseLibraryReaderMedia } from '../adapter/course-library-reader.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path));
const hash = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');
const journalName = '2026-11-13-course-library-media-additions.json';
const evidence = JSON.parse(read('docs/audit/' + journalName).toString()) as {
  archives: Record<string, string>;
  files: { source: string; before_sha256: string }[];
  sources: Record<string, {
    notice: string; licenseSha256: string;
    objects: [upstreamPath: string, localPath: string, blob: string, sha256: string, pages?: number][];
  }>;
  previews: [localPath: string, page: number, sha256: string][];
};

describe('source library original media', () => {
  it('matches the pinned source bytes and license notices', () => {
    for (const { notice, licenseSha256, objects } of Object.values(evidence.sources)) {
      expect(hash(read(notice)), notice).toBe(licenseSha256);
      for (const [upstream, local, blob, sha256, pages] of objects) {
        const bytes = read(local);
        expect(hash(bytes), upstream).toBe(sha256);
        expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'), upstream)
          .toBe(blob);
        if (pages) expect(bytes.subarray(0, 5).toString()).toBe('%PDF-');
      }
    }
    for (const [local, page, expected] of evidence.previews) {
      expect(local).toMatch(new RegExp('-page-' + page + '\\.png$'));
      expect(hash(read(local)), local).toBe(expected);
    }
  });

  it('keeps the archive bytes and prior adapter revision', () => {
    for (const [name, expected] of Object.entries(evidence.archives)) {
      const archive = '05 Источники/Courses/' + name;
      const digest = createHash('sha256');
      for (const path of readdirSync(resolve(root, archive), { recursive: true }).filter(path => path.endsWith('.md')).sort()) {
        digest.update(path + '\0').update(read(archive + '/' + path)).update('\0');
      }
      expect(digest.digest('hex'), name).toBe(expected);
    }
    for (const file of evidence.files) {
      expect(hash(beforeTopicConsolidation(read(file.source).toString(), file.source, root,
        '2026-11-13-gpt-generation-reader-comparison-additions.json'))).toBe(file.before_sha256);
    }
  });

  it('only rewrites the notebook media on the matching reader route, outside code', () => {
    const route = 'sources/courses/lf-recommenders/ncf-deep-dive';
    const source = '05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb.md';
    const html = '<img src="https://raw.githubusercontent.com/recommenders-team/resources/main/images/NCF.svg?sanitize=true">';
    const code = '\x60\x60\x60html\n' + html + '\n\x60\x60\x60';
    const body = html + '\n\n' + code;
    const reader = courseLibraryReaderMedia(route, source, body);
    expect(reader).toContain('![[Assets/Sources/Linux Foundation Recommenders/NCF.svg|NeuMF');
    expect(reader).toContain('1708.05031v2#S3.F3');
    expect(reader.endsWith(code)).toBe(true);
    expect(courseLibraryReaderMedia(route + '-other', source, body)).toBe(body);
    expect(courseLibraryReaderMedia(route, 'unknown.md', body)).toBe(body);
  });

  it('keeps the full extracted PDF text verbatim below the local document', () => {
    const marker = '## Extracted document text\n';
    for (const [collection, relative, slug] of [
      ['Machine Learning Visualized', 'book/main.pdf.md', 'machine-learning-visualized/complete-book'],
      ['Stanford CS230 Cheatsheets', 'en/cheatsheet-recurrent-neural-networks.pdf.md', 'stanford-cs230-cheatsheets/rnn'],
    ]) {
      const source = '05 Источники/Courses/' + collection + '/' + relative;
      const original = read(source).toString();
      const rendered = courseLibraryReaderMedia('sources/courses/' + slug, source, original);
      expect(rendered.split(marker)).toHaveLength(2);
      expect(rendered.split(marker)[1]).toBe(original.split(marker)[1]);
      expect(rendered).not.toContain('<iframe');
      expect(rendered).toMatch(/\]\(\/source-documents\/[^)]+-page-[15]\.png\)\]\(\/source-documents\/[^)]+\.pdf#page=[15]\)/);
      expect(rendered).toContain('[Open the original PDF at full size]');
    }
  });
});
