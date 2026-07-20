import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import sidebar from './generated-sidebar.mjs';

function rejectNonAsciiMath() {
  return (tree, file) => {
    const visit = (node) => {
      if ((node.type === 'math' || node.type === 'inlineMath') && /[^\x00-\x7F]/.test(node.value)) {
        throw new Error(`Non-ASCII text in LaTeX at ${file.path}: ${node.value}`);
      }
      if (Array.isArray(node.children)) node.children.forEach(visit);
    };
    visit(tree);
  };
}

export default defineConfig({
  site: 'https://alpotekhin.github.io',
  base: process.env.PUBLICATION_BASE_PATH || '/',
  output: 'static',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, rejectNonAsciiMath],
      rehypePlugins: [rehypeKatex]
    })
  },
  integrations: [
    starlight({
      title: 'Bookvar',
      defaultLocale: 'root',
      locales: { root: { label: 'Русский', lang: 'ru' } },
      customCss: ['./src/styles/custom.css'],
      sidebar
    })
  ]
});
