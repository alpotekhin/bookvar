import { mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';

function fixture(yaml: string, sources: string[] = []): string {
  const root = mkdtempSync(join(tmpdir(), 'publication-manifest-'));
  const publishing = join(root, 'publishing');
  mkdirSync(publishing);

  for (const source of sources) {
    const absolute = join(root, source);
    mkdirSync(join(absolute, '..'), { recursive: true });
    writeFileSync(absolute, '# Page\n');
  }

  const path = join(publishing, 'navigation.yml');
  writeFileSync(path, yaml);
  return path;
}

const validYaml = `
site_title: ML & DL Handbook
sections:
  - id: textbook
    title: Учебник
    pages:
      - source: 00 Учебник/_index.md
        route: textbook/index
  - id: practice
    title: Практика
    pages:
      - source: 06 Практика/lab.md
        route: practice/attention-lab
`;

describe('loadManifest', () => {
  it('keeps the published textbook index on the same 1 through 68 primary route', () => {
    const root = join(import.meta.dirname, '..', '..');
    const index = readFileSync(join(root, '00 Учебник', '_index.md'), 'utf8');
    const primaryNumbers = [...index.matchAll(/^(\d+)\. \[\[/gm)]
      .map((match) => Number(match[1]));
    expect(primaryNumbers).toEqual(Array.from({ length: 68 }, (_, index) => index + 1));
    expect(index).toContain('S1. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы]]');
    expect(index).toContain('S7. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки]]');
    expect(index).toContain('Вводная: [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура');
  });

  it('keeps all 68 curriculum routes in the matrix and publication manifest in the same order', () => {
    const root = join(import.meta.dirname, '..', '..');
    const matrix = readFileSync(join(root, '00 Учебник', 'Редакционная матрица Bookvar.md'), 'utf8');
    const matrixRows = [...matrix.matchAll(/^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|$/gm)]
      .map((match) => ({ number: Number(match[1]), route: match[2] }));
    expect(matrixRows.map(({ number }) => number)).toEqual(
      Array.from({ length: 68 }, (_, index) => index + 1)
    );

    const manifest = loadManifest(join(root, 'publishing', 'navigation.yml'));
    const publishedRoutes = new Set(
      manifest.sections.flatMap((section) => section.pages.map((page) => page.route))
    );
    for (const { route } of matrixRows) expect(publishedRoutes.has(route), route).toBe(true);

    const textbookOrder = manifest.sections
      .find((section) => section.id === 'textbook')
      ?.pages.map((page) => page.route) ?? [];
    const curriculumOrder = textbookOrder.filter((route) =>
      matrixRows.some((row) => row.route === route)
    );
    expect(curriculumOrder).toEqual(matrixRows.map(({ route }) => route));
  });

  it('maps every curriculum topic from 1 through 68 to compared source material', () => {
    const root = join(import.meta.dirname, '..', '..');
    const maps = ['Темы 01–18.md', 'Темы 19–40.md', 'Темы 41–68.md']
      .map((name) => readFileSync(join(root, '05 Источники', 'Source maps', name), 'utf8'))
      .join('\n');
    const topicNumbers = [...maps.matchAll(/^##\s+(\d+)\./gm)].map((match) => Number(match[1]));
    expect(topicNumbers).toEqual(Array.from({ length: 68 }, (_, index) => index + 1));
  });

  it('loads the representative handbook manifest with at least eight real pages', () => {
    const path = join(import.meta.dirname, '..', 'navigation.yml');
    const manifest = loadManifest(path);

    expect(manifest.sections.map((section) => section.id)).toEqual([
      'textbook', 'models', 'sources', 'questions', 'practice'
    ]);
    expect(manifest.sections.flatMap((section) => section.pages).length).toBeGreaterThanOrEqual(8);
  });

  it('returns ordered sections and pages from the exact manifest schema', () => {
    const manifest = loadManifest(fixture(validYaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]));

    expect(manifest.siteTitle).toBe('ML & DL Handbook');
    expect(manifest.sections.map((section) => section.id)).toEqual(['textbook', 'practice']);
    expect(manifest.sections[0]?.pages).toEqual([{
      source: '00 Учебник/_index.md',
      route: 'textbook/index'
    }]);
  });

  it('loads optional English sources and navigation labels without duplicating routes', () => {
    const yaml = validYaml
      .replace('title: Учебник', 'title: Учебник\n    title_en: Textbook')
      .replace(
        '        route: textbook/index',
        '        route: textbook/index\n        source_en: en/textbook/index.md'
      );
    const manifest = loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      'en/textbook/index.md',
      '06 Практика/lab.md'
    ]));

    expect(manifest.sections[0]).toMatchObject({
      title: 'Учебник',
      titleEn: 'Textbook',
      pages: [{
        source: '00 Учебник/_index.md',
        sourceEn: 'en/textbook/index.md',
        route: 'textbook/index'
      }]
    });
  });

  it('loads nested sidebar groups while keeping section pages flat', () => {
    const yaml = validYaml.replace(
      '        route: textbook/index',
      [
        '        route: textbook/index',
        '    sidebar:',
        '      - label: I. Основы',
        '        items:',
        '          - label: 1. Первая тема',
        '            route: textbook/index'
      ].join('\n')
    );

    const manifest = loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]));

    expect(manifest.sections[0]?.sidebar).toEqual([{
      label: 'I. Основы',
      items: [{ label: '1. Первая тема', route: 'textbook/index' }]
    }]);
    expect(manifest.sections[0]?.pages).toEqual([{
      source: '00 Учебник/_index.md',
      route: 'textbook/index'
    }]);
  });

  it('loads optional English labels on nested sidebar items', () => {
    const yaml = validYaml.replace(
      '        route: textbook/index',
      [
        '        route: textbook/index',
        '    sidebar:',
        '      - label: I. Основы',
        '        label_en: I. Foundations',
        '        items:',
        '          - label: 1. Первая тема',
        '            label_en: 1. First topic',
        '            route: textbook/index'
      ].join('\n')
    );

    const manifest = loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]));

    expect(manifest.sections[0]?.sidebar).toEqual([{
      label: 'I. Основы',
      labelEn: 'I. Foundations',
      items: [{
        label: '1. Первая тема',
        labelEn: '1. First topic',
        route: 'textbook/index'
      }]
    }]);
  });

  it('defines fourteen readable textbook modules without changing flat routes', () => {
    const root = join(import.meta.dirname, '..', '..');
    const textbook = loadManifest(join(root, 'publishing', 'navigation.yml')).sections
      .find((section) => section.id === 'textbook')!;
    const modules = textbook.sidebar?.filter((item) => 'items' in item) ?? [];
    expect(modules).toHaveLength(14);
    expect(modules.map((module) => module.label)).toEqual([
      'I. Математические и ML-основания',
      'II. Классическое машинное обучение',
      'III. Рекомендательные системы',
      'IV. Нейронные сети',
      'V. Текст до Transformer',
      'VI. Transformer, BERT и GPT',
      'VII. Анатомия современной LLM',
      'VIII. Вычислительные основы ML-систем',
      'IX. Обучение LLM',
      'X. Инференс и serving',
      'XI. Эксплуатация и оценивание',
      'XII. Retrieval и RAG',
      'XIII. Мультимодальные модели',
      'XIV. Инструменты и агенты'
    ]);

    const flattenSidebar = (items: NonNullable<typeof textbook.sidebar>): typeof items =>
      items.flatMap((item) => 'route' in item ? [item] : flattenSidebar(item.items));
    const moduleItems = flattenSidebar(modules);
    const primaryNumbers = moduleItems
      .map((item) => item.label.match(/^(\d+)\. /))
      .filter((match): match is RegExpMatchArray => match !== null)
      .map((match) => Number(match[1]));
    expect(primaryNumbers).toEqual(
      Array.from({ length: 68 }, (_, index) => index + 1)
    );
    expect(moduleItems.filter((item) => /^\d+\.\d+ /.test(item.label)).map((item) => item.label))
      .toEqual(expect.arrayContaining([
        expect.stringMatching(/^55\.1 /),
        expect.stringMatching(/^55\.2 /),
        expect.stringMatching(/^55\.3 /),
        expect.stringMatching(/^58\.1 /),
        expect.stringMatching(/^58\.2 /)
      ]));
    const numberedLabels = moduleItems
      .map((item) => item.label.match(/^(?:S\d+|\d+(?:\.\d+)?)\.?/i)?.[0])
      .filter((label): label is string => label !== undefined);
    expect(numberedLabels).toHaveLength(new Set(numberedLabels).size);
    const sidebarRoutes = flattenSidebar(textbook.sidebar ?? []).map((item) => item.route);
    expect(sidebarRoutes).toEqual(textbook.pages.map((page) => page.route));
  });

  it('rejects sidebar routes absent from the flat page manifest', () => {
    const yaml = validYaml.replace(
      '        route: textbook/index',
      '        route: textbook/index\n    sidebar:\n      - label: Missing\n        route: textbook/missing'
    );
    expect(() => loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]))).toThrow(/Unknown sidebar route.*textbook\/missing/);
  });

  it('rejects a missing source file', () => {
    expect(() => loadManifest(fixture(validYaml, ['00 Учебник/_index.md'])))
      .toThrow(/Missing source.*06 Практика\/lab\.md/);
  });

  it('rejects manifest sources under the ignored raw tree', () => {
    const yaml = validYaml.replace('00 Учебник/_index.md', 'raw/papers/example.md');
    const path = fixture(yaml, [
      'raw/papers/example.md',
      '06 Практика/lab.md'
    ]);

    expect(() => loadManifest(path)).toThrow(/Manifest source must not reference ignored raw content.*raw\/papers/);
  });

  it.each([
    './raw/papers/example.md',
    'raw/../raw/papers/example.md',
    'notes/../raw/papers/example.md',
    '.\\raw\\papers\\example.md'
  ])('rejects normalized raw source variant: %s', (source) => {
    const yaml = validYaml.replace('00 Учебник/_index.md', source);
    const path = fixture(yaml, [
      'raw/papers/example.md',
      '06 Практика/lab.md'
    ]);

    expect(() => loadManifest(path)).toThrow(/Manifest source must not reference ignored raw content/);
  });

  it('rejects a source symlink that resolves under the raw tree', () => {
    const path = fixture(validYaml, [
      'raw/papers/example.md',
      '06 Практика/lab.md'
    ]);
    const root = join(path, '..', '..');
    mkdirSync(join(root, '00 Учебник'), { recursive: true });
    symlinkSync(join(root, 'raw/papers/example.md'), join(root, '00 Учебник/_index.md'));

    expect(() => loadManifest(path)).toThrow(/Manifest source must not reference ignored raw content/);
  });

  it('does not confuse an ordinary draw directory with raw', () => {
    const yaml = validYaml.replace('00 Учебник/_index.md', 'draw/example.md');
    const manifest = loadManifest(fixture(yaml, [
      'draw/example.md',
      '06 Практика/lab.md'
    ]));

    expect(manifest.sections[0]?.pages[0]?.source).toBe('draw/example.md');
  });

  it('rejects duplicate routes', () => {
    const yaml = validYaml.replace('practice/attention-lab', 'textbook/index');
    expect(() => loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]))).toThrow(/Duplicate route.*textbook\/index/);
  });

  it('rejects duplicate sources', () => {
    const yaml = validYaml.replace('06 Практика/lab.md', '00 Учебник/_index.md');
    expect(() => loadManifest(fixture(yaml, ['00 Учебник/_index.md'])))
      .toThrow(/Duplicate source.*00 Учебник\/_index\.md/);
  });

  it('rejects an unknown section ID', () => {
    const yaml = validYaml.replace('id: practice', 'id: archive');
    expect(() => loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]))).toThrow(/Unknown section ID.*archive/);
  });

  it('rejects a route beginning with a slash', () => {
    const yaml = validYaml.replace('route: textbook/index', 'route: /textbook/index');
    expect(() => loadManifest(fixture(yaml, [
      '00 Учебник/_index.md',
      '06 Практика/lab.md'
    ]))).toThrow(/Route must not begin with.*\//);
  });

  it.each(['../outside.md', '/tmp/outside.md'])(
    'rejects a source path outside the publication root: %s',
    (source) => {
      const path = fixture(validYaml.replace('00 Учебник/_index.md', source), [
        '06 Практика/lab.md'
      ]);
      expect(() => loadManifest(path)).toThrow(/Source path escapes publication root/);
    }
  );
});
