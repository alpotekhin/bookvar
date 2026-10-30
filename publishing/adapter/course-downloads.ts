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
      if (typeof item.local_path !== 'string' || typeof item.canonical_url !== 'string') {
        throw new Error(`Missing course PDF path or official URL: ${sourceRoot}`);
      }
      const offset = relative(resolve(rootDir, sourceRoot), resolve(rootDir, sourceRoot, item.local_path));
      if (isAbsolute(item.local_path) || offset === '..' || offset.startsWith(`..${sep}`)) {
        throw new Error(`Course PDF path escapes its source root: ${item.local_path}`);
      }
      const url = new URL(item.canonical_url);
      if (url.protocol !== 'https:' || url.username || url.password) {
        throw new Error(`Invalid official course PDF URL: ${sourceRoot}/${item.local_path}`);
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
