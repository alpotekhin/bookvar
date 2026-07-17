import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  output: 'static',
  integrations: [
    starlight({
      title: 'ML & DL Handbook',
      defaultLocale: 'root',
      locales: { root: { label: 'Русский', lang: 'ru' } },
      sidebar: [{ label: 'Начало', items: [{ label: 'О проекте', slug: 'index' }] }]
    })
  ]
});
