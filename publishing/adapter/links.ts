import type { RouteRegistry } from './types.js';

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;

function basename(target: string): string {
  const name = target.slice(target.lastIndexOf('/') + 1);
  return name.replace(/\.md$/, '');
}

export function wikiHeadingSlug(heading: string): string {
  return heading
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\p{L}\p{N}_-]/gu, '');
}

function convertLink(
  expression: string,
  registry: RouteRegistry,
  unresolved: string[],
  allowlist: ReadonlyMap<string, string>,
  allowed: Array<{ target: string; reason: string }>
): string {
  const [targetWithHeading, alias] = expression.split('|', 2);
  const headingAt = targetWithHeading.indexOf('#');
  const target = headingAt === -1
    ? targetWithHeading
    : targetWithHeading.slice(0, headingAt);
  const heading = headingAt === -1
    ? undefined
    : targetWithHeading.slice(headingAt + 1);
  const label = alias ?? (target === '' ? heading ?? '' : basename(target));
  if (target === '' && heading !== undefined) {
    return `[${label}](#wiki-${wikiHeadingSlug(heading)})`;
  }
  const route = registry.routeForWikiTarget(target);

  if (route === undefined) {
    const reason = allowlist.get(target);
    if (reason === undefined) unresolved.push(target);
    else allowed.push({ target, reason });
    return label;
  }

  const anchor = heading === undefined ? '' : `#wiki-${wikiHeadingSlug(heading)}`;
  return `[${label}](${route}${anchor})`;
}

function convertEmbed(
  expression: string,
  registry: RouteRegistry,
  unresolved: string[],
  allowlist: ReadonlyMap<string, string>,
  allowed: Array<{ target: string; reason: string }>
): string {
  if (!IMAGE_EXTENSION.test(expression)) {
    return convertLink(expression, registry, unresolved, allowlist, allowed);
  }

  const assetPath = expression.replace(/^Assets\//, '');
  const filename = basename(assetPath);
  const alt = filename.replace(IMAGE_EXTENSION, '');
  return `![${alt}](../../assets/${assetPath})`;
}

export function convertWikiSyntax(
  markdown: string,
  registry: RouteRegistry,
  allowlist: ReadonlyMap<string, string> = new Map()
): {
  markdown: string;
  unresolved: string[];
  allowlisted: Array<{ target: string; reason: string }>;
} {
  const unresolved: string[] = [];
  const allowlisted: Array<{ target: string; reason: string }> = [];
  const withoutEmbeds = markdown.replace(
    /!\[\[([^\]\n]+)\]\]/g,
    (match, expression: string) => expression.trim() === ''
      ? match
      : convertEmbed(expression, registry, unresolved, allowlist, allowlisted)
  );
  const converted = withoutEmbeds.replace(
    /\[\[([^\]\n]+)\]\]/g,
    (match, expression: string) => expression.trim() === ''
      ? match
      : convertLink(expression, registry, unresolved, allowlist, allowlisted)
  );

  return { markdown: converted, unresolved, allowlisted };
}
