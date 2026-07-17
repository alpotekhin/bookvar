import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import sidebar from './generated-sidebar.mjs';

export default defineConfig({
  output: 'static',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex]
    })
  },
  integrations: [
    mermaid({ enableLog: false }),
    starlight({
      title: 'ML & DL Handbook',
      defaultLocale: 'root',
      locales: { root: { label: 'Русский', lang: 'ru' } },
      customCss: ['./src/styles/custom.css'],
      sidebar
    })
  ]
});
