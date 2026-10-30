import { readdir, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Use decimal GB so the artifact also fits the stricter interpretation of Pages' limit.
export const PAGES_MAX_BYTES = 1_000_000_000;

export async function measurePublicationSize(dir: string): Promise<{ bytes: number; files: number }> {
  let bytes = 0;
  let files = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = resolve(dir, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Publication artifact contains a symlink: ${path}`);
    if (entry.isDirectory()) {
      const nested = await measurePublicationSize(path);
      bytes += nested.bytes;
      files += nested.files;
    } else if (entry.isFile()) {
      bytes += (await stat(path)).size;
      files += 1;
    }
  }
  return { bytes, files };
}

export function assertPublicationSize(bytes: number): void {
  if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > PAGES_MAX_BYTES) {
    throw new Error(`Pages artifact exceeds the ${PAGES_MAX_BYTES}-byte limit or has an invalid size: ${bytes}`);
  }
}

const modulePath = fileURLToPath(import.meta.url);
if (process.argv[1] && resolve(process.argv[1]) === modulePath) {
  const size = await measurePublicationSize(resolve(dirname(modulePath), '../../site/dist'));
  assertPublicationSize(size.bytes);
  console.log(JSON.stringify({ ...size, limit: PAGES_MAX_BYTES, headroom: PAGES_MAX_BYTES - size.bytes }));
}
