import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { checkHseReleaseBoundary } from '../tools/check-hse-release-boundary.mjs';

const archiveText = '05 Источники/Courses/HSE ML course/ml1-2026-spring/task.ipynb.md';
const archiveImage = 'Assets/Sources/HSE ML course/notes/plot.png';
const sourceBytes = 'archived notebook content\n';
const assetBytes = Buffer.from([0, 127, 255, 25]);
const script = resolve(import.meta.dirname, '../tools/check-hse-release-boundary.mjs');
let tmp: string, repo: string, dist: string, archiveRef: string;

function git(...args: string[]): string {
  return execFileSync('git', args, { cwd: repo, encoding: 'utf8' }).trim();
}
function write(root: string, path: string, body: string | Buffer): void {
  const full = join(root, path);
  mkdirSync(dirname(full), { recursive: true });
  writeFileSync(full, body);
}
function commit(): string {
  git('add', '.');
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
      '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture');
  return git('rev-parse', 'HEAD');
}

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), 'hse-release-'));
  repo = join(tmp, 'repository');
  dist = join(tmp, 'site');
  mkdirSync(repo);
  git('init', '-q', '--object-format=sha1');
  write(repo, archiveText, sourceBytes);
  write(repo, archiveImage, assetBytes);
  archiveRef = commit();
  write(dist, 'index.html', '<a href="https://github.com/esokolov/ml-course-hse">Source</a>');
});
afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

describe('HSE release boundary', () => {
  it('reports the tracked archive independently from a site containing only catalogue links', async () => {
    write(repo, 'draft.txt', sourceBytes);
    const result = await checkHseReleaseBoundary({ repo, archiveRef, candidate: 'HEAD', dist });
    expect(result.archive).toEqual({ commit: archiveRef, files: 2, bytes: Buffer.byteLength(sourceBytes) + 4 });
    expect(result.candidate).toEqual({ commit: archiveRef, matching: [
      { path: archiveText, inArchivePath: true, exactCopy: true },
      { path: archiveImage, inArchivePath: true, exactCopy: true },
    ] });
    expect(result.artifact.files).toBe(1);
    expect(result.artifact.matching).toEqual([]);
  });

  it('finds exact copies after both original paths leave the candidate, in Git and on the site', async () => {
    rmSync(join(repo, archiveText));
    rmSync(join(repo, archiveImage));
    write(repo, 'renamed/source.txt', sourceBytes);
    const candidate = commit();
    write(dist, 'renamed/plot.png', assetBytes);
    write(dist, 'renamed/same-size.bin', Buffer.from([1, 127, 255, 25]));
    write(dist, 'assets/' + archiveImage, 'different bytes at archive path');
    write(dist, 'assets/' + archiveImage.slice('Assets/'.length), 'changed image on the public path');
    const result = await checkHseReleaseBoundary({ repo, archiveRef, candidate, dist });
    expect(result.candidate.matching).toEqual([
      { path: 'renamed/source.txt', inArchivePath: false, exactCopy: true },
    ]);
    expect(result.artifact.matching).toEqual([
      { path: 'assets/' + archiveImage, inArchivePath: true, exactCopy: false },
      { path: 'assets/' + archiveImage.slice('Assets/'.length), inArchivePath: true, exactCopy: false },
      { path: 'renamed/plot.png', inArchivePath: false, exactCopy: true },
    ]);
  });

  it('accepts a clean candidate with a catalogue and rejects empty baselines and incomplete sites', async () => {
    rmSync(join(repo, archiveText));
    rmSync(join(repo, archiveImage));
    write(repo, 'course-index.md', 'Link to the source course.');
    const candidate = commit();
    const opts = { repo, archiveRef, candidate, dist };
    expect((await checkHseReleaseBoundary(opts)).candidate.matching).toEqual([]);
    await expect(checkHseReleaseBoundary({ ...opts, archiveRef: candidate })).rejects.toThrow('no known archive blobs');
    symlinkSync(join(dist, 'index.html'), join(dist, 'linked.html'));
    await expect(checkHseReleaseBoundary(opts)).rejects.toThrow('Unexpected symlink');
    rmSync(join(dist, 'index.html'));
    await expect(checkHseReleaseBoundary(opts)).rejects.toThrow('index.html');
  });

  it('requires a reference revision at the command line', () => {
    const result = spawnSync(process.execPath, [script], { encoding: 'utf8' });
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('Pass --archive-ref');
  });
});
