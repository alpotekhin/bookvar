import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';

const siteRoot = fileURLToPath(new URL('..', import.meta.url));

async function read(relativePath: string) {
  return readFile(new URL(relativePath, `file://${siteRoot}/`), 'utf8');
}

describe('handbook landing output', () => {
  test('renders exactly one h1 and removes prototype copy', async () => {
    const html = await read('dist/index.html');
    expect(html.match(/<h1(?:\s|>)/g)).toHaveLength(1);
    expect(html).not.toContain('Публикационный прототип');
    expect(html).not.toContain('Содержательные страницы будут сгенерированы');
  });

  test('renders every expected navigation card', async () => {
    const html = await read('dist/index.html');
    const cards = [...html.matchAll(/<a class="entry-card" href="([^"]+)">([\s\S]*?)<\/a>/g)]
      .map((match) => ({ href: match[1], label: match[2].match(/<strong>(.*?)<\/strong>/)?.[1] }));
    expect(cards).toEqual([
      { href: './textbook/', label: 'Учебник' },
      { href: './reference/', label: 'Справочник' },
      { href: './models/', label: 'Атлас моделей' },
      { href: './sources/', label: 'Источники' },
      { href: './practice/causal-self-attention/', label: 'Практика' },
      { href: './questions/llm/', label: 'Вопросы' },
    ]);
  });
});

describe('diagram presentation', () => {
  test('keeps Mermaid diagrams readable and horizontally scrollable on mobile', async () => {
    const css = await read('src/styles/custom.css');
    expect(css).toMatch(/\.mermaid\s*\{[^}]*overflow-x:\s*auto/s);
    expect(css).toMatch(/\.mermaid\s+svg\s*\{[^}]*min-width:/s);
    expect(css).toMatch(/\.mermaid\s+svg\s*\{[^}]*margin-inline:\s*0/s);
    expect(css).toMatch(/@media\s*\(max-width:\s*30rem\)/);
  });
});
