import type { RouteRegistry } from './types.js';

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;

function basename(target: string): string {
  const name = target.slice(target.lastIndexOf('/') + 1);
  return name.replace(/\.md$/, '');
}

function headingSlug(heading: string): string {
  return heading
    .toLocaleLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\p{L}\p{N}_-]/gu, '');
}

function convertLink(
  expression: string,
  registry: RouteRegistry,
  unresolved: string[]
): string {
  const [targetWithHeading, alias] = expression.split('|', 2);
  const headingAt = targetWithHeading.indexOf('#');
  const target = headingAt === -1
    ? targetWithHeading
    : targetWithHeading.slice(0, headingAt);
  const heading = headingAt === -1
    ? undefined
    : targetWithHeading.slice(headingAt + 1);
  const label = alias ?? basename(target);
  const route = registry.routeForWikiTarget(target);

  if (route === undefined) {
    unresolved.push(target);
    return label;
  }

  const anchor = heading === undefined ? '' : `#${headingSlug(heading)}`;
  return `[${label}](${route}${anchor})`;
}

function convertEmbed(
  expression: string,
  registry: RouteRegistry,
  unresolved: string[]
): string {
  if (!IMAGE_EXTENSION.test(expression)) {
    return convertLink(expression, registry, unresolved);
  }

  const assetPath = expression.replace(/^Assets\//, '');
  const filename = basename(assetPath);
  const alt = filename.replace(IMAGE_EXTENSION, '');
  return `![${alt}](../../assets/${assetPath})`;
}

export function convertWikiSyntax(
  markdown: string,
  registry: RouteRegistry
): { markdown: string; unresolved: string[] } {
  const unresolved: string[] = [];
  const withoutEmbeds = markdown.replace(
    /!\[\[([^\]\n]+)\]\]/g,
    (_match, expression: string) => convertEmbed(expression, registry, unresolved)
  );
  const converted = withoutEmbeds.replace(
    /\[\[([^\]\n]+)\]\]/g,
    (_match, expression: string) => convertLink(expression, registry, unresolved)
  );

  return { markdown: converted, unresolved };
}
