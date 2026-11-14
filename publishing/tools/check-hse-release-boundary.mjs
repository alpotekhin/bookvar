import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { lstat, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const archiveRoots = [
  '05 Источники/Courses/HSE ML course/',
  'Assets/Sources/HSE ML course/',
];
const archivePath = path => archiveRoots.some(root => path.startsWith(root));
const artifactArchivePath = path => archivePath(path) ||
  (path.startsWith('assets/') && (archivePath(path.slice(7)) || archivePath('Assets/' + path.slice(7))));
const defaultRoot = fileURLToPath(new URL('../..', import.meta.url));

function git(repo, args) {
  return execFileSync('git', args, { cwd: repo, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
}

function tree(repo, ref) {
  const commit = git(repo, ['rev-parse', '--verify', '--end-of-options', ref + '^{commit}']).trim();
  const entries = git(repo, ['ls-tree', '-r', '-l', '-z', '--full-tree', commit])
    .split('\0').filter(Boolean).map(entry => {
      const tab = entry.indexOf('\t');
      const [, type, hash, size] = entry.slice(0, tab).split(/\s+/);
      return { type, hash, bytes: size === '-' ? null : Number(size), path: entry.slice(tab + 1) };
    });
  return { commit, entries };
}

async function inspectArtifact(dist, bySize, algorithm) {
  const matching = [];
  let files = 0;
  async function walk(directory, prefix = '') {
    for (const item of await readdir(directory, { withFileTypes: true })) {
      const path = prefix + item.name;
      const full = resolve(directory, item.name);
      if (item.isSymbolicLink()) throw new Error('Unexpected symlink in artifact: ' + path);
      if (item.isDirectory()) {
        await walk(full, path + '/');
      } else if (item.isFile()) {
        files += 1;
        const bytes = (await lstat(full)).size;
        const hashes = bySize.get(bytes);
        const inArchivePath = artifactArchivePath(path);
        let exactCopy = false;
        if (hashes) {
          const hash = createHash(algorithm).update('blob ' + bytes + '\0');
          for await (const part of createReadStream(full)) hash.update(part);
          exactCopy = hashes.has(hash.digest('hex'));
        }
        if (inArchivePath || exactCopy) matching.push({ path, inArchivePath, exactCopy });
      }
    }
  }
  await walk(dist);
  if (files === 0) throw new Error('Artifact has no files: ' + dist);
  return { files, matching: matching.sort((a, b) => a.path.localeCompare(b.path)) };
}

/** Match a candidate tree and built site against a revision containing the known archive. */
export async function checkHseReleaseBoundary({ repo = defaultRoot, candidate, archiveRef, dist }) {
  const archive = tree(repo, archiveRef);
  const originals = archive.entries.filter(entry => entry.type === 'blob' && archivePath(entry.path));
  if (originals.length === 0) throw new Error('Archive reference contains no known archive blobs: ' + archiveRef);

  const hashes = new Set(originals.map(entry => entry.hash));
  const bySize = new Map();
  for (const entry of originals) {
    if (!bySize.has(entry.bytes)) bySize.set(entry.bytes, new Set());
    bySize.get(entry.bytes).add(entry.hash);
  }
  const target = tree(repo, candidate);
  const gitMatches = [];
  for (const entry of target.entries) {
    const inArchivePath = archivePath(entry.path);
    const exactCopy = entry.type === 'blob' && hashes.has(entry.hash);
    if (inArchivePath || exactCopy) gitMatches.push({ path: entry.path, inArchivePath, exactCopy });
  }
  const algorithm = git(repo, ['rev-parse', '--show-object-format']).trim();
  if (!(await lstat(dist)).isDirectory()) throw new Error('Artifact must be a regular directory: ' + dist);
  const index = resolve(dist, 'index.html');
  if (!(await lstat(index)).isFile()) throw new Error('Artifact has no regular index.html: ' + dist);
  const artifact = await inspectArtifact(dist, bySize, algorithm);
  return {
    archive: { commit: archive.commit, files: originals.length, bytes: originals.reduce((sum, entry) => sum + entry.bytes, 0) },
    candidate: { commit: target.commit, matching: gitMatches },
    artifact: { directory: dist, ...artifact },
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: {
      ref: { type: 'string', default: 'HEAD' },
      'archive-ref': { type: 'string' },
      dist: { type: 'string', default: 'site/dist' },
    } });
    if (!values['archive-ref']) throw new Error('Pass --archive-ref with the known archive revision.');
    const result = await checkHseReleaseBoundary({
      candidate: values.ref, archiveRef: values['archive-ref'], dist: resolve(defaultRoot, values.dist),
    });
    const { archive, candidate, artifact } = result;
    for (const [kind, data] of [['candidate', candidate], ['artifact', artifact]]) {
      const { matching, ...rest } = data;
      const inArchivePath = matching.filter(row => row.inArchivePath).length;
      const exactCopiesOutsideArchive = matching.filter(row => row.exactCopy && !row.inArchivePath).length;
      console.log(JSON.stringify({ kind, ...rest, inArchivePath, exactCopiesOutsideArchive, examples: matching.slice(0, 5) }));
    }
    console.log(JSON.stringify({ kind: 'archive-reference', ...archive }));
    if (candidate.matching.length || artifact.matching.length) process.exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 2;
  }
}
