import type { RouteRegistry } from './types.js';

const IMAGE_EXTENSION = /\.(?:png|jpe?g|webp|svg|gif)$/i;
const PROTECTED_TOKEN_PREFIX = 'BOOKVARPROTECTEDCODE';

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
  const protectedSegments: string[] = [];
  const protect = (segment: string): string => {
    const token = `${PROTECTED_TOKEN_PREFIX}${protectedSegments.length}TOKEN`;
    protectedSegments.push(segment);
    return token;
  };
  // Obsidian syntax is meaningful in prose, not inside literal source code.
  // Protect fenced blocks first, then inline code spans. The fence expression
  // deliberately accepts both backticks and tildes and preserves the source
  // byte-for-byte when the tokens are restored below.
  const withoutFencedCode = markdown.replace(
    /^( {0,3})(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\2[ \t]*$/gm,
    protect
  );
  const withoutCode = withoutFencedCode.replace(
    /(`+)([\s\S]*?)\1/g,
    protect
  );
  const withoutEmbeds = withoutCode.replace(
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
  const restored = converted.replace(
    new RegExp(`${PROTECTED_TOKEN_PREFIX}(\\d+)TOKEN`, 'g'),
    (_match, index: string) => protectedSegments[Number(index)]
  );

  return { markdown: restored, unresolved, allowlisted };
}
