import { readFile } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import YAML from 'yaml';

/** Deliver full lecture PDFs from their recorded official URLs, not the Pages quota. */
export async function loadCoursePdfDownloads(rootDir: string, sourceRoots: readonly string[]): Promise<Map<string, string>> {
  const downloads = new Map<string, string>();
  for (const sourceRoot of sourceRoots) {
    let raw: string;
    try {
      raw = await readFile(join(rootDir, sourceRoot, 'source-manifest.yml'), 'utf8');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
      throw error;
    }
    const manifest = YAML.parse(raw) as { objects?: Array<Record<string, unknown>> };
    for (const item of manifest.objects ?? []) {
      if (item.kind !== 'pdf' || item.source_authority !== 'official-course' || item.verification_status !== 'verified') continue;
      // Archive-only paths still resolve old links, but require no public copy.
      const localPath = item.local_path ?? item.local_archive_path;
      if (typeof localPath !== 'string' || typeof item.canonical_url !== 'string') {
        throw new Error(`Missing course PDF path or official URL: ${sourceRoot}`);
      }
      const offset = relative(resolve(rootDir, sourceRoot), resolve(rootDir, sourceRoot, localPath));
      if (isAbsolute(localPath) || offset === '..' || offset.startsWith(`..${sep}`)) {
        throw new Error(`Course PDF path escapes its source root: ${localPath}`);
      }
      const url = new URL(item.canonical_url);
      if (url.protocol !== 'https:' || url.username || url.password) {
        throw new Error(`Invalid official course PDF URL: ${sourceRoot}/${localPath}`);
      }
      const githubFile = url.pathname.match(/^\/([^/]+)\/([^/]+)\/blob\/([a-f0-9]{40})\/(.+\.pdf)$/i);
      if (url.hostname === 'github.com' && githubFile) {
        url.hostname = 'raw.githubusercontent.com';
        url.pathname = `/${githubFile[1]}/${githubFile[2]}/${githubFile[3]}/${githubFile[4]}`;
      }
      const source = join(sourceRoot, offset).split(sep).join('/');
      if (downloads.has(source)) throw new Error(`Duplicate course PDF source: ${source}`);
      downloads.set(source, url.href);
    }
  }
  return downloads;
}
