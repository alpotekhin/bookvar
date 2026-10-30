import { mapMarkdownProse } from './markdown-prose.js';

/** Rewrite real root-relative links, leaving fenced, inline and HTML code intact. */
export function rewritePublicationLinks(markdown: string, href: (path: string) => string): string {
  const transform = (text: string) => text
    .replace(/(\]\(\s*<?)(\/(?!\/)[^\s)>]*)(>?)/g, (_match, lead, path, end) => lead + href(path) + end)
    .replace(/^( {0,3}\[[^\]\n]+\]:\s*<?)(\/(?!\/)[^\s>]*)(>?)/gm,
      (_match, lead, path, end) => lead + href(path) + end)
    .replace(/\b((?:href|src)\s*=\s*["'])(\/(?!\/)[^"']*)/gi,
      (_match, lead, path) => lead + href(path));
  return mapMarkdownProse(markdown, prose => {
    // Imported notebooks can use literal HTML code blocks instead of fences.
    const code = /<(pre|code|script|style)\b[^>]*>[\s\S]*?<\/\1>/gi;
    let end = 0, result = '';
    for (const match of prose.matchAll(code)) {
      result += transform(prose.slice(end, match.index)) + match[0];
      end = match.index! + match[0].length;
    }
    return result + transform(prose.slice(end));
  });
}

/** Add the project prefix to real root-relative links, never to code examples. */
export function prefixPublicationLinks(markdown: string, base: string): string {
  if (!base || base === '/') return markdown;
  const prefix = '/' + base.replace(/^\/+|\/+$/g, '');
  return rewritePublicationLinks(markdown, path => path === prefix || path.startsWith(prefix + '/')
    ? path : prefix + path);
}
