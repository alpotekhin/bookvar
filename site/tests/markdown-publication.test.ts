import { describe, expect, it } from 'vitest';
import config from '../astro.config.mjs';
import { archiveCodeDisplay } from '../src/lib/markdown-publication';

describe('publication-only Markdown presentation', () => {
  it('retains MathML and original TeX while excluding their duplicate search text', async () => {
    const renderer = await config.markdown!.processor!.createRenderer({});
    const result = await renderer.render('Inline $x^2 + y$.\n\n$$\n\\frac{a}{b}\n$$\n');
    const mathml = [...result.code.matchAll(/<span[^>]*class="katex-mathml"[^>]*>/g)];
    expect(mathml).toHaveLength(2);
    for (const span of mathml) expect(span[0]).toContain('data-pagefind-ignore="all"');
    expect(result.code).toContain('<math');
    expect(result.code).toContain('<annotation encoding="application/x-tex">');
    expect(result.code).toContain('\\frac{a}{b}');
    expect((result.code.match(/class="katex-html"/g) ?? []).length).toBe(2);
  });

  it('changes only the highlighting language of long source listings, never their code', () => {
    for (const locale of ['', 'en/']) {
      const codeBlock = {
        code: '# Original source\n'.repeat(2000), language: 'python',
        parentDocument: { sourceFilePath: `/repo/site/src/content/docs/generated/${locale}sources/courses/course/source.md` }
      };
      const original = codeBlock.code;
      archiveCodeDisplay.hooks.preprocessLanguage({ codeBlock });
      expect(codeBlock.language).toBe('plaintext');
      expect(codeBlock.code).toBe(original);
    }
  });

  it('preserves highlighting in textbook examples, short archives and unknown contexts', () => {
    for (const sourceFilePath of [undefined, '/repo/generated/textbook/example.md', '/repo/src/sources/courses/example.md']) {
      const codeBlock = { code: 'x = 1\n'.repeat(4000), language: 'python', parentDocument: { sourceFilePath } };
      archiveCodeDisplay.hooks.preprocessLanguage({ codeBlock });
      expect(codeBlock.language).toBe('python');
    }
    const short = { code: 'x = 1', language: 'python', parentDocument: { sourceFilePath: '/repo/generated/sources/courses/example.md' } };
    archiveCodeDisplay.hooks.preprocessLanguage({ codeBlock: short });
    expect(short.language).toBe('python');
  });

  it('uses the exact long-listing boundary without truncating either listing', () => {
    for (const length of [4999, 5000]) {
      const codeBlock = { code: 'x'.repeat(length), language: 'python', parentDocument: { sourceFilePath: '/repo/generated/sources/courses/example.md' } };
      archiveCodeDisplay.hooks.preprocessLanguage({ codeBlock });
      expect(codeBlock.language).toBe(length < 5000 ? 'python' : 'plaintext');
      expect(codeBlock.code).toHaveLength(length);
    }
  });
});
