import type { RouteRegistry } from './types.js';
import { mapMarkdownProse } from './markdown-prose.js';

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
  const route = target === '' ? '' : registry.routeForWikiTarget(target);

  if (target !== '' && route === undefined) {
    const reason = allowlist.get(target);
    if (reason === undefined) unresolved.push(target);
    else allowed.push({ target, reason });
    return label;
  }

  const canonicalHref = registry.canonicalHrefForWikiTarget?.(target, heading);
  if (canonicalHref !== undefined) return `[${label}](${canonicalHref})`;

  let anchor = '';
  if (heading !== undefined) {
    const fragment = registry.fragmentForWikiTarget?.(target, heading);
    if (registry.fragmentForWikiTarget && fragment === undefined) {
      const fullTarget = `${target}#${heading}`;
      const reason = allowlist.get(fullTarget);
      if (reason === undefined) unresolved.push(fullTarget);
      else allowed.push({ target: fullTarget, reason });
      return label;
    }
    anchor = `#${fragment ?? wikiHeadingSlug(heading)}`;
  }
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
  const assetHref = `../../assets/${assetPath}`;
  return `[![${alt}](${assetHref})](${assetHref})`;
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
  // Transform prose in place. Nested temporary markers could survive a single
  // restoration pass and replace entire source blocks in the rendered book.
  const restored = mapMarkdownProse(markdown, prose => prose.replace(
    /!\[\[([^\]\n]+)\]\]/g,
    (match, expression: string) => expression.trim() === ''
      ? match
      : convertEmbed(expression, registry, unresolved, allowlist, allowlisted)
  ).replace(
    /\[\[([^\]\n]+)\]\]/g,
    (match, expression: string) => expression.trim() === ''
      ? match
      : convertLink(expression, registry, unresolved, allowlist, allowlisted)
  ));

  return { markdown: restored, unresolved, allowlisted };
}
