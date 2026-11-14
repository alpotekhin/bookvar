import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { edlsReaderMedia } from '../adapter/edls-course-media.js';

const read = (path: string) => readFileSync(new URL('../../' + path, import.meta.url));
const archive = '05 Источники/Courses/Efficient DL Systems/';
const policy = JSON.parse(read('publishing/adapter/edls-course-figures.json').toString()) as {
  revision: string;
  license: { path: string; blob: string; sha256: string };
  courseTraces: Record<string, { blob: string; sha256: string }>;
  duplicateTraces: Record<string, string>;
};

function verify(path: string, entry: { blob: string; sha256: string }) {
  const bytes = read(path);
  expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex')).toBe(entry.blob);
  expect(createHash('sha256').update(bytes).digest('hex')).toBe(entry.sha256);
}

describe('Efficient DL Systems reader media', () => {
  it('retains the archived code and replaces the duplicate trace with the verified script link', () => {
    const source = read(archive + 'week06_dl_arithmetic/seminar/practice.ipynb.md').toString();
    const converted = edlsReaderMedia(source);
    const codeBlocks = (value: string) => value.match(/^```[^\n]*\n[\s\S]*?^```$/gm);
    expect(codeBlocks(converted)).toEqual(codeBlocks(source));
    expect(converted).toContain('the same image for the compiled run');
    expect(converted).toContain('/blob/' + policy.revision + '/week06_dl_arithmetic/seminar/cross_entropy/profile_compiled_ce.py');
    expect(converted).not.toContain('![[Assets/Sources/Efficient DL Systems/repository-figures/compiled_ce.png');
    expect(converted).toContain('https://horace.io/brrr_intro.html');
    expect(converted).not.toContain('<p float="left">');
    expect(converted).toContain('https://arxiv.org/html/2410.10989v3#S3.F1');
  });

  it('rewrites links in reader prose but not in an example', () => {
    const url = 'https://i.imgur.com/a1QhKF7.png';
    const markup = '![img](' + url + ')';
    const source = [markup, '', '`' + markup + '`', '', '```md', markup, '```', ''].join('\n');
    const transformed = edlsReaderMedia(source);
    expect(transformed).toBe(source.replace(markup, '[Example generation (the generated code is blue)](' + url + ')'));
  });

  it('ships the exact upstream MIT notice and byte-identical authored traces', () => {
    verify(policy.license.path, policy.license);
    for (const [file, row] of Object.entries(policy.courseTraces)) {
      verify('Assets/Sources/Efficient DL Systems/repository-figures/' + file, row);
    }
    expect(policy.duplicateTraces['compiled_ce.png']).toBe(policy.courseTraces['vanilla_ce.png'].blob);
  });
});
