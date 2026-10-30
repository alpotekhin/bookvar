import { createHash } from 'node:crypto';
import { lstatSync, readFileSync } from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';

export interface PublicAssetRecord {
  asset: string;
  sha256?: string;
  bytes?: number;
  public_license_copy?: string;
}

function localFile(root: string, path: string): string {
  const file = resolve(root, path);
  const offset = relative(root, file);
  if (isAbsolute(path) || offset === '..' || offset.startsWith(`..${sep}`) || path.split(/[\\/]/).includes('..')) {
    throw new Error(`Public asset path escapes the repository: ${path}`);
  }
  if (!lstatSync(file).isFile()) throw new Error(`Public asset must be a regular file: ${path}`);
  return file;
}

/** Check the distributed bytes. Upstream/source identity is a separate evidence check. */
export function verifyPublicAssets(rootDir: string, assets: PublicAssetRecord[], legacyUnhashed: ReadonlySet<string>) {
  const root = resolve(rootDir);
  if (assets.length === 0) throw new Error('Public asset registry is empty');
  let hashed = 0, legacy = 0;
  const notices = new Set<string>();
  for (const row of assets) {
    const bytes = readFileSync(localFile(root, row.asset));
    if (row.sha256) {
      const actual = createHash('sha256').update(bytes).digest('hex');
      if (!/^[a-f0-9]{64}$/.test(row.sha256) || actual !== row.sha256) {
        throw new Error(`Public asset SHA-256 mismatch: ${row.asset}`);
      }
      if (row.bytes !== undefined && row.bytes !== bytes.length) throw new Error(`Public asset size mismatch: ${row.asset}`);
      hashed++;
    } else {
      if (!legacyUnhashed.has(row.asset)) throw new Error(`New public asset needs SHA-256: ${row.asset}`);
      legacy++;
    }
    if (row.public_license_copy) {
      if (!row.public_license_copy.startsWith('site/public/licenses/')) throw new Error(`License notice is outside the public license directory: ${row.asset}`);
      const notice = readFileSync(localFile(root, row.public_license_copy), 'utf8');
      if (!notice.trim()) throw new Error(`Public license notice is empty: ${row.asset}`);
      notices.add(row.public_license_copy);
    }
  }
  return { assets: assets.length, hashed, legacyUnhashed: legacy, publicNotices: notices.size };
}
