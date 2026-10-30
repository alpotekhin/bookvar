import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import sidebar from './generated-sidebar.mjs';
import { normalizeArchiveMath } from '../publishing/adapter/math.ts';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { rehypeMathSearch, archiveCodeDisplay } from './src/lib/markdown-publication.ts';

const processor = unified({
  remarkPlugins: [remarkMath],
  rehypePlugins: [rehypeKatex, rehypeMathSearch]
});
// Astro's content digest serializes processor options, not imported function
// dependencies. Invalidate cached HTML whenever render normalization changes.
Object.assign(processor.options, {
  bookvarMathSourceSha256: createHash('sha256')
    .update(readFileSync(new URL('../publishing/adapter/math.ts', import.meta.url)))
    .digest('hex'),
  bookvarPublicationPresentationSha256: createHash('sha256')
    .update(readFileSync(new URL('./src/lib/markdown-publication.ts', import.meta.url)))
    .digest('hex')
});
const createRenderer = processor.createRenderer.bind(processor);
const publicationBase = process.env.PUBLICATION_BASE_PATH || '/';
const causalSelfAttentionHref = `${publicationBase.replace(/\/+$/, '')}/practice/causal-self-attention`;
processor.createRenderer = async (shared) => {
  const renderer = await createRenderer(shared);
  return {
    ...renderer,
    render(content, options) {
      const normalized = normalizeArchiveMath(content, options?.fileURL);
      return renderer.render(normalized, options);
    }
  };
};

export default defineConfig({
  site: 'https://alpotekhin.github.io',
  base: publicationBase,
  output: 'static',
  redirects: {
    '/en/practice/causal-self-attention': causalSelfAttentionHref,
    '/practice/causal-self-attention-notes': causalSelfAttentionHref,
    '/en/practice/causal-self-attention-notes': causalSelfAttentionHref
  },
  markdown: {
    processor
  },
  integrations: [
    starlight({
      title: 'Bookvar',
      defaultLocale: 'root',
      locales: {
        root: { label: 'Русский', lang: 'ru' },
        en: { label: 'English', lang: 'en' }
      },
      customCss: ['./src/styles/custom.css'],
      expressiveCode: { plugins: [archiveCodeDisplay] },
      components: {
        Footer: './src/components/BookFooter.astro',
        PageTitle: './src/components/BookPageTitle.astro',
        FallbackContentNotice: './src/components/BookFallbackContentNotice.astro'
      },
      sidebar
    })
  ]
});
