import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { auditHarvardMedia } from '../tools/check-harvard-media.js';

const course = 'Sources/Harvard ML Systems/';
const chapter = course + 'vol1/serving/queue.svg';
const slide = course + 'slides/vol1/serving/images/steps.png';
const retired = course + 'vol1/serving/old-photo.png';
const external = course + 'slides/vol1/serving/images/remote-photo.png';
const route = 'sources/courses/harvard-ml-systems/serving/index.html';
const bytes = Buffer.from('<svg/>');
const sha256 = createHash('sha256').update(bytes).digest('hex');
const gitBlob = createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');
const scratch: string[] = [];

afterEach(() => {
  for (const directory of scratch.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function fixture(basePath = '') {
  const dist = mkdtempSync(join(tmpdir(), 'bookvar-harvard-media-'));
  scratch.push(dist);
  const write = (file: string, content: string | Buffer) => {
    mkdirSync(dirname(join(dist, file)), { recursive: true });
    writeFileSync(join(dist, file), content);
  };
  const href = (file: string) => basePath + '/assets/' + file.split('/').map(encodeURIComponent).join('/');
  const main = '<main><img data-src="ignored" src="' + href(chapter) +
    '" alt="queue"><img src="' + href(slide) + '">' +
    '<pre><code><img src="' + href(retired) + '"></code></pre>' +
    '<a href="https://mlsysbook.ai/photo.png">External link</a></main>';
  write(route, main);
  write('assets/' + chapter, bytes);
  write('assets/' + slide, bytes);
  const args = {
    dist, basePath,
    chapters: [{ images: [chapter, retired].map(file => ({
      file, sha256, source_url: 'https://mlsysbook.ai/' + file.slice(course.length),
    })) }],
    slides: [{
      state: 'local-original' as const, local_path: 'Assets/' + slide, sha256,
      source: slide.slice(course.length), source_commit: '45ecc8d82fcae70c149cdce550d3b3d3411df913',
      git_blob_sha1: gitBlob,
    }, {
      state: 'external-source-link' as const, source: external.slice(course.length),
      source_commit: '45ecc8d82fcae70c149cdce550d3b3d3411df913', git_blob_sha1: gitBlob,
    }],
    retired: ['Assets/' + retired],
  };
  return { write, href, args, main };
}

describe('Harvard media in the built site', () => {
  it.each(['', '/bookvar'])('checks exact chapter and slide bytes with base "%s"', base => {
    const { args } = fixture(base);
    expect(auditHarvardMedia(args)).toEqual({
      coursePages: 1, visibleCourseMedia: 2, checkedDistributionPaths: 2,
      matching: { chapters: 1, slides: 1 }, upstreamMatching: null,
      retired: 1, externalOnlySlides: 1, errors: [],
    });
  });

  it('detects missing bytes, corrupted bytes and an unregistered embed', () => {
    const { write, href, args, main } = fixture();
    write(route, main.replace('</main>', '<img src="' + href(course + 'invented.svg') + '"></main>'));
    write('assets/' + slide, 'different');
    rmSync(join(args.dist, 'assets', chapter));
    expect(auditHarvardMedia(args).errors).toEqual(expect.arrayContaining([
      'Missing rendered image: /assets/' + chapter,
      'SHA-256 mismatch: /assets/' + slide,
      expect.stringContaining('image missing from both course manifests:'),
    ]));
  });

  it('rejects an excluded copy in the textbook even if the original source page is clean', () => {
    const { write, href, args } = fixture();
    write('textbook/example/index.html', '<main><img src="' + href(retired) + '"></main>');
    write('assets/' + retired, bytes);
    write('assets/Assets/' + external, bytes);
    const errors = auditHarvardMedia(args).errors;
    expect(errors).toContain('/textbook/example/: embeds excluded ' + retired);
    expect(errors).toContain('Excluded asset is distributed: assets/' + retired);
    expect(errors).toContain('Excluded asset is distributed: assets/Assets/' + external);
  });

  it('checks source Git blobs separately from the manifest SHA-256', () => {
    const { args } = fixture();
    const upstream = {
      chapters: new Map([['vol1/serving/queue.svg', '0'.repeat(40)]]),
      slides: new Map([['slides/vol1/serving/images/steps.png', gitBlob]]),
    };
    const result = auditHarvardMedia({ ...args, upstream });
    expect(result.matching).toEqual({ chapters: 1, slides: 1 });
    expect(result.upstreamMatching).toEqual({ chapters: 0, slides: 1 });
    expect(result.errors).toEqual([expect.stringContaining('Git blob mismatch or missing source: vol1/serving/queue.svg')]);
  });
});
