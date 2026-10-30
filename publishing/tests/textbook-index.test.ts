import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadManifest, type PublicationSidebarItem } from '../adapter/manifest.js';
import { readingPaths } from '../adapter/reading-paths.js';

const root = resolve(import.meta.dirname, '../..');
const textbook = loadManifest(resolve(root, 'publishing/navigation.yml')).sections
  .find((section) => section.id === 'textbook')!;
const ru = readFileSync(resolve(root, '00 Учебник/_index.md'), 'utf8');
const en = readFileSync(resolve(root, 'en/00 Textbook/_index.md'), 'utf8');
const normalize = (path: string) => path.replace(/^02 Areas\/ML & DL\//, '').replace(/\.md$/, '');
const routes = new Map(textbook.pages.flatMap((page) => [page.source, page.sourceEn]
  .filter((source): source is string => Boolean(source))
  .map((source) => [normalize(source), page.route] as const)));

function chapterLinks(text: string): string[] {
  return [...text.matchAll(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g)]
    .map((match) => normalize(match[1]))
    .filter((source) => source.startsWith('00 Учебник/') || source.startsWith('en/00 Textbook/'))
    .map((source) => {
      expect(routes.has(source), `Unpublished textbook index target: ${source}`).toBe(true);
      return routes.get(source)!;
    });
}

function flatten(items: PublicationSidebarItem[]): Array<Extract<PublicationSidebarItem, { route: string }>> {
  return items.flatMap((item) => 'route' in item ? [item] : flatten(item.items));
}

describe('reader-facing textbook order', () => {
  it('introduces SFT, reward modeling and PPO before the worked Constitutional AI recipe', () => {
    const expected = ['module-map', 'sft-instruction-data', 'fine-tuning-peft',
      'preference-data', 'reward-modeling', 'ppo', 'alignment-goals', 'dpo',
      'rlvr-verifiers', 'grpo-deepseek-r1', 'reasoning-distillation'];
    const order = readingPaths(textbook).find(path => path.id === 'main')!.routes
      .filter(route => route.startsWith('textbook/post-training/'))
      .map(route => route.slice('textbook/post-training/'.length));
    expect(order).toEqual(expected);
    for (const contents of [ru, en]) {
      expect(chapterLinks(contents).filter(route => route.startsWith('textbook/post-training/'))
        .map(route => route.slice('textbook/post-training/'.length))).toEqual(expected);
    }
  });

  it('sends the post-training introduction to SFT instead of a later preference calculation', () => {
    const intro = readFileSync(resolve(root,
      '00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники.md'), 'utf8');
    expect(intro).toContain('next: "[[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]"');
    const choices = [...intro.matchAll(/^\d+\. (\[\[[^\n]+\]\])[^\n]*$/gm)]
      .flatMap(match => chapterLinks(match[1]));
    const expected = readingPaths(textbook).find(path => path.id === 'main')!.routes
      .filter(route => route.startsWith('textbook/post-training/')
        && !route.endsWith('/module-map'));
    expect(choices).toEqual(expected);
  });

  it('does not leak source-file ordinals into chapter link labels', () => {
    for (const match of ru.matchAll(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g)) {
      const label = match[2] ?? match[1].split('/').at(-1)!;
      expect(label, match[1]).not.toMatch(/^\d+[a-z]?\s/);
    }
  });

  it('renders the systems sequence as separate Markdown list items', () => {
    const systems = ru.split('Вычислительные основы ML-систем\n')[1].split(/^### /m)[0];
    expect(systems.match(/^- \[\[/gm)).toHaveLength(8);
    expect(systems).not.toMatch(/^S\d+\./m);
  });

  it('preserves every chapter exactly once across core and independent modules', () => {
    const contents = ru.split('## Основной курс')[1].split('## Справочные разделы')[0];
    const routes = chapterLinks(contents);
    expect(routes).toEqual(flatten(textbook.sidebar ?? []).slice(2).map(page => page.route));
    expect([...routes].sort()).toEqual(textbook.pages.slice(2).map(page => page.route).sort());
    expect(new Set(routes).size).toBe(routes.length);
    expect(chapterLinks(ru.split('## Основной курс')[0])).toContain('textbook/how-to-use');
    const paths = readingPaths(textbook);
    expect(chapterLinks(ru.split('## Основной курс')[1].split('## Углублённые модули')[0]))
      .toEqual(paths.find(path => path.id === 'main')!.routes);
    expect(paths.find(path => path.id === 'main')!.routes).not.toContain('textbook/pretraining/semi-supervised-fixmatch');
  });

  it('uses the same unnumbered chapter labels in the full contents and sidebar', () => {
    const labels=new Map(flatten(textbook.sidebar ?? []).map(item=>[item.route,item.label]));
    const fullRoute=ru.split('## Основной курс')[1].split('## Справочные разделы')[0];
    const entries=[...fullRoute.matchAll(/^- (\[\[[^|]+\|([^\]]+)\]\])$/gm)];
    expect(entries).toHaveLength(textbook.pages.length-2);
    for(const match of entries) {
      const [route]=chapterLinks(match[1]);
      expect(labels.get(route),route).toBe(match[2]);
    }
  });

  it('preserves the complete core and specialist routes in the English contents', () => {
    const contents = en.split('## Core course')[1].split('## Reference sections')[0];
    const routes = chapterLinks(contents);
    expect(routes).toEqual(flatten(textbook.sidebar ?? []).slice(2).map(page => page.route));
    expect(new Set(routes).size).toBe(routes.length);
    expect(chapterLinks(en.split('## Core course')[1].split('## Specialist modules')[0]))
      .toEqual(readingPaths(textbook).find(path => path.id === 'main')!.routes);
    expect(en).toContain('the English edition of the whole book is not');
    expect(en).toContain('<a id="углублённые-модули"></a>\n\n## Specialist modules');
    expect(en).toContain('<a id="контрастивное-и-маскированное-предобучение"></a>\n\n### Contrastive and masked pre-training');
    const notice = readFileSync(resolve(root, 'site/src/components/BookPageTitle.astro'), 'utf8');
    expect(notice).toContain('Partial English edition');
    expect(notice).toContain('Canonical-language chapter');
    expect(notice).toContain('Read the canonical chapter');
  });

  it('lists every authored English chapter once and does not label a fallback as available', () => {
    const available = en.split('\n').filter(line => line.endsWith(' · **EN**'));
    expect(chapterLinks(available.join('\n')).sort()).toEqual(
      textbook.pages.slice(1).filter((page) => page.sourceEn).map((page) => page.route).sort());
  });
});
