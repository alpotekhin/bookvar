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

  test('renders every expected navigation card in its section', async () => {
    const html = await read('dist/index.html');
    const sections = [...html.matchAll(/<nav class="entry-grid" aria-labelledby="([^"]+)">([\s\S]*?)<\/nav>/g)]
      .map((section) => ({
        labelledBy: section[1],
        cards: [...section[2].matchAll(/<a class="entry-card" href="([^"]+)">([\s\S]*?)<\/a>/g)]
          .map((match) => ({ href: match[1], label: match[2].match(/<strong>(.*?)<\/strong>/)?.[1] }))
      }));
    expect(sections).toEqual([
      {
        labelledBy: 'choose-a-route',
        cards: [
          { href: './textbook/', label: 'Учебник' },
          { href: './reference/', label: 'Справочник' },
          { href: './models/', label: 'Атлас моделей' },
          { href: './sources/', label: 'Источники' },
          { href: './practice/causal-self-attention/', label: 'Практика' },
          { href: './questions/llm/', label: 'Вопросы' },
        ]
      },
      {
        labelledBy: 'university-courses',
        cards: [
          { href: './sources/courses/stanford-cs336-spring-2026/', label: 'Stanford CS336 — Language Modeling from Scratch' },
          { href: './sources/courses/berkeley-advanced-llm-agents-spring-2025/', label: 'Berkeley Advanced LLM Agents' },
        ]
      }
    ]);
  });
});

describe('diagram presentation', () => {
  test.each([
    '2f10e92057738ef1e4a50e524fddf5389d61d795.svg',
    '3b022a3ffa6f028fe2c2c3165c292a16a3a9e0ff.svg',
  ])('keeps the original Harvard figure %s readable in dark mode', async (filename) => {
    const css = await read('src/styles/custom.css');
    const rule = css.match(/\.sl-markdown-content img\[src\*='\/Assets\/Figures\/'\]([^{}]*)\{([^}]+)\}/);
    expect(rule).not.toBeNull();
    expect(rule?.[1]).toContain(`.sl-markdown-content img[src$='/${filename}']`);
    expect(rule?.[2]).toMatch(/background-color:\s*#fff;/);
    expect(rule?.[2]).not.toMatch(/(?:filter|width|height|object-fit):/);
    expect(css).not.toContain("img[src*='/Assets/Sources/']");
  });

  test('scales the MHA/MLA projection figure responsively without cropping', async () => {
    const css = await read('src/styles/custom.css');
    const rule = css.match(/\.sl-markdown-content img\[src\$='\/Assets\/Figures\/curated\/attention-review-2026-09\/mha-mla-projections\.svg'\]\s*\{([^}]+)\}/);
    expect(rule).not.toBeNull();
    expect(rule?.[1]).toMatch(/width:\s*min\(100%,\s*48rem\);/);
    expect(rule?.[1]).toMatch(/height:\s*auto;/);
    expect(rule?.[1]).not.toMatch(/object-fit:\s*cover/);
  });

  test('scales the Geva FFN figure responsively without cropping', async () => {
    const css = await read('src/styles/custom.css');
    const rule = css.match(/\.sl-markdown-content img\[src\$='\/Assets\/Figures\/curated\/ffn-review-2026-09\/geva-ffn-memory\.svg'\]\s*\{([^}]+)\}/);
    expect(rule).not.toBeNull();
    expect(rule?.[1]).toMatch(/width:\s*min\(100%,\s*36rem\);/);
    expect(rule?.[1]).toMatch(/height:\s*auto;/);
    expect(rule?.[1]).not.toMatch(/object-fit:\s*cover/);
  });

  test('keeps Mermaid diagrams readable and horizontally scrollable on mobile', async () => {
    const css = await read('src/styles/custom.css');
    expect(css).toMatch(/\.mermaid\s*\{[^}]*overflow-x:\s*auto/s);
    expect(css).toMatch(/\.mermaid\s+svg\s*\{[^}]*min-width:/s);
    expect(css).toMatch(/\.mermaid\s+svg\s*\{[^}]*margin-inline:\s*0/s);
    expect(css).toMatch(/@media\s*\(max-width:\s*30rem\)/);
  });
});
