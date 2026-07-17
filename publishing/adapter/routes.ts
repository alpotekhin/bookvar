import type { PageRecord, RouteRegistry } from './types.js';

function normalizeWikiTarget(target: string): string {
  return target
    .replaceAll('\\', '/')
    .replace(/^02 Areas\/ML & DL\//, '')
    .replace(/\.md$/, '');
}

export function createRouteRegistry(
  records: PageRecord[],
  options: { allowAmbiguousBasenames?: boolean } = {}
): RouteRegistry {
  const routes = new Map<string, string>();
  const basenames = new Set<string>();
  const ambiguousBasenames = new Set<string>();

  for (const record of records) {
    const sourcePath = normalizeWikiTarget(record.sourcePath);
    const basename = sourcePath.slice(sourcePath.lastIndexOf('/') + 1);

    if (basenames.has(basename)) {
      if (!options.allowAmbiguousBasenames) {
        throw new Error(`Ambiguous basename: ${basename}`);
      }
      routes.delete(basename);
      ambiguousBasenames.add(basename);
    }

    basenames.add(basename);
    routes.set(sourcePath, record.route);
    if (!ambiguousBasenames.has(basename)) routes.set(basename, record.route);
  }

  return {
    routeForWikiTarget(target: string): string | undefined {
      return routes.get(normalizeWikiTarget(target));
    }
  };
}
