import type { PageRecord, RouteRegistry } from './types.js';

function normalizeWikiTarget(target: string): string {
  return target.replaceAll('\\', '/').replace(/\.md$/, '');
}

export function createRouteRegistry(records: PageRecord[]): RouteRegistry {
  const routes = new Map<string, string>();
  const basenames = new Set<string>();

  for (const record of records) {
    const sourcePath = normalizeWikiTarget(record.sourcePath);
    const basename = sourcePath.slice(sourcePath.lastIndexOf('/') + 1);

    if (basenames.has(basename)) {
      throw new Error(`Ambiguous basename: ${basename}`);
    }

    basenames.add(basename);
    routes.set(sourcePath, record.route);
    routes.set(basename, record.route);
  }

  return {
    routeForWikiTarget(target: string): string | undefined {
      return routes.get(normalizeWikiTarget(target));
    }
  };
}
