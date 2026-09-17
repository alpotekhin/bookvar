import { describe, expect, test } from 'vitest';
import config from '../astro.config.mjs';

describe('configured textbook Markdown rendering', () => {
  test.each([
    String.raw`r = \text{награда}`,
    String.raw`x − y`,
    String.raw`π_θ(y \mid x)`
  ])('keeps the article and renders valid Unicode math: %s', async (formula) => {
    const processor = config.markdown?.processor;
    if (!processor) throw new Error('The site must configure a Markdown processor');
    const renderer = await processor.createRenderer({});
    const result = await renderer.render(`Before formula.\n\n$${formula}$\n\nAfter formula.`, {
      fileURL: new URL('../src/content/docs/generated/textbook/unicode-fixture.md', import.meta.url)
    });
    expect(result.code).toContain('<p>Before formula.</p>');
    expect(result.code).toContain('class="katex"');
    expect(result.code).not.toContain('katex-error');
    expect(result.code).toContain('<p>After formula.</p>');
  });
});
