import { mkdtempSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import matter from 'gray-matter';
import { describe, expect, it } from 'vitest';
import {
  buildPublication,
  insertFragmentAliases,
  parseMarkdownHeadings,
  publicationHref
} from '../adapter/build.js';

const configuredBase = process.env.PUBLICATION_BASE_PATH?.replace(/^\/+|\/+$/g, '') ?? '';
const publicationBase = configuredBase === '' ? '' : `/${configuredBase}`;

function write(path: string, contents: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, contents);
}

function fixture(
  body = 'См. [[Page B]] и [[Missing]].\n\n![[Assets/Figures/chart.svg]]',
  targetBody = '> [!warning] Check\n> Body'
): {
  rootDir: string;
  manifestPath: string;
  outputDir: string;
  reportPath: string;
  sourcePath: string;
} {
  const root = mkdtempSync(join(tmpdir(), 'publication-build-'));
  const sourcePath = join(root, 'Notes', 'Page A.md');
  write(sourcePath, [
    '---',
    'title: Page A',
    'type: concept',
    'status: stable',
    'last_updated: 2026-07-17',
    '---',
    body
  ].join('\n'));
  write(join(root, 'Notes', 'Page B.md'), [
    '---',
    'title: Page B',
    'type: concept',
    'status: stable',
    '---',
    targetBody
  ].join('\n'));
  write(join(root, 'Assets', 'Figures', 'chart.svg'), '<svg>fixture</svg>');

  const manifestPath = join(root, 'publishing', 'navigation.yml');
  write(manifestPath, [
    'site_title: Fixture',
    'sections:',
    '  - id: textbook',
    '    title: Textbook',
    '    pages:',
    '      - source: Notes/Page A.md',
    '        route: nested/page-a',
    '      - source: Notes/Page B.md',
    '        route: page-b'
  ].join('\n'));
  write(join(root, 'publishing', 'link-allowlist.json'), JSON.stringify({
    entries: [{ target: 'Missing', reason: 'test fixture target' }]
  }));

  return {
    rootDir: root,
    manifestPath,
    outputDir: join(root, 'site', 'src', 'content', 'docs', 'generated'),
    reportPath: join(root, 'publishing-report.json'),
    sourcePath
  };
}

describe('buildPublication', () => {
  it('uses the emitted directory URL for section index routes', () => {
    expect(publicationHref('textbook/index')).toBe(`${publicationBase}/textbook/`);
    expect(publicationHref('textbook/chapter')).toBe(`${publicationBase}/textbook/chapter/`);
  });
  it('keeps compatibility redirects out of the search index', async () => {
    const options = fixture('Compatibility page.');
    const source = readFileSync(options.sourcePath, 'utf8')
      .replace('status: stable', 'status: redirect');
    write(options.sourcePath, source);

    await buildPublication(options);

    const generated = matter(readFileSync(
      join(options.outputDir, 'nested', 'page-a.md'),
      'utf8'
    ));
    expect(generated.data.pagefind).toBe(false);
  });
  it('rejects incomplete source fragment attribution metadata', async () => {
    const options = fixture('Original excerpt.');
    const source = readFileSync(options.sourcePath, 'utf8').replace(
      'last_updated: 2026-07-17',
      [
        'last_updated: 2026-07-17',
        'source_fragment:',
        '  author: Course Author',
        '  course: Systems Course',
        '  source_url: https://github.com/example/course/blob/0123456789abcdef0123456789abcdef01234567/lecture.md',
        '  language: en',
        '  transformation: original excerpt'
      ].join('\n')
    );
    write(options.sourcePath, source);

    await expect(buildPublication(options)).rejects.toThrow(
      'source_fragment.license must be a non-empty string'
    );
  });

  it('renders complete source fragment attribution without changing excerpt language', async () => {
    const options = fixture('Original excerpt stays in English.');
    const source = readFileSync(options.sourcePath, 'utf8').replace(
      'last_updated: 2026-07-17',
      [
        'last_updated: 2026-07-17',
        'source_fragment:',
        '  author: Course Author',
        '  course: Systems Course',
        '  source_url: https://github.com/example/course/blob/0123456789abcdef0123456789abcdef01234567/lecture.md',
        '  license: CC BY 4.0',
        '  language: en',
        '  transformation: original excerpt'
      ].join('\n')
    );
    write(options.sourcePath, source);

    await buildPublication(options);

    const generated = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8'));
    expect(generated.content).toContain('<aside class="source-attribution"');
    expect(generated.content).toContain('Original course material');
    expect(generated.content).toContain('<strong>Systems Course</strong>');
    expect(generated.content).toContain('Course Author');
    expect(generated.content).toContain('Language: en');
    expect(generated.content).toContain('License: CC BY 4.0');
    expect(generated.content).toContain('Modification: original excerpt');
    expect(generated.content).toContain('href="https://github.com/example/course/blob/0123456789abcdef0123456789abcdef01234567/lecture.md"');
    expect(generated.content).toContain('Original excerpt stays in English.');
  });
  it('generates nested pages, converted Markdown, copied assets, and a warning report without mutating sources', async () => {
    const options = fixture();
    const before = readFileSync(options.sourcePath);

    await buildPublication(options);

    const generated = readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8');
    expect(generated).toContain('title: Page A');
    expect(generated).toContain('description: Page A');
    const metadata = matter(generated).data;
    expect(metadata).not.toHaveProperty('editUrl');
    expect(generated).not.toMatch(/file:|\/Users\//);
    expect(metadata.lastUpdated).toBeInstanceOf(Date);
    expect(metadata.slug).toBe('nested/page-a');
    expect(generated).toContain(`[Page B](${publicationBase}/page-b/)`);
    expect(generated).toContain('Missing');
    expect(generated).toContain(
      `[![chart](${publicationBase}/assets/Figures/chart.svg)](${publicationBase}/assets/Figures/chart.svg)`
    );
    expect(readFileSync(join(options.outputDir, 'page-b.md'), 'utf8'))
      .toContain(':::caution[Check]\nBody\n:::');
    expect(JSON.parse(readFileSync(options.reportPath, 'utf8'))).toMatchObject({
      pages: [{
        source: 'Notes/Page A.md',
        route: 'nested/page-a',
        unresolved: [],
        allowlisted: [{ target: 'Missing', reason: 'test fixture target' }]
      }]
    });
    expect(readFileSync(join(options.rootDir, 'site', 'public', 'assets', 'Figures', 'chart.svg'), 'utf8'))
      .toBe('<svg>fixture</svg>');
    expect(readFileSync(options.sourcePath)).toEqual(before);
    expect(readFileSync(join(options.rootDir, 'site', 'generated-sidebar.mjs'), 'utf8')).toBe([
      '// Generated from publishing/navigation.yml. Do not edit.',
      'export default [',
      '  {',
      '    "label": "Textbook",',
      '    "items": [',
      '      {',
      '        "label": "Page A",',
      '        "slug": "nested/page-a"',
      '      },',
      '      {',
      '        "label": "Page B",',
      '        "slug": "page-b"',
      '      }',
      '    ]',
      '  }',
      '];',
      ''
    ].join('\n'));
  });

  it('emits English translations under en and lets Starlight fall back for untranslated pages', async () => {
    const options = fixture('Русская страница A', '## Section\n\nРусская страница B');
    write(join(options.rootDir, 'English', 'Page A.md'), [
      '---',
      'title: Page A in English',
      'type: concept',
      'status: stable',
      '---',
      'English page linking to [[Page B]] and [[Page B#Section]].'
    ].join('\n'));
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: textbook',
      '    title: Учебник',
      '    title_en: Textbook',
      '    pages:',
      '      - source: Notes/Page A.md',
      '        source_en: English/Page A.md',
      '        route: nested/page-a',
      '      - source: Notes/Page B.md',
      '        route: page-b',
      '    sidebar:',
      '      - label: 28. Первая страница',
      '        route: nested/page-a',
      '      - label: Вторая страница',
      '        route: page-b'
    ].join('\n'));

    await buildPublication(options);

    const translated = readFileSync(join(options.outputDir, 'en', 'nested', 'page-a.md'), 'utf8');
    expect(translated).toContain('title: Page A in English');
    expect(matter(translated).data.slug).toBe('en/nested/page-a');
    expect(translated).toContain(`[Page B](${publicationBase}/en/page-b/)`);
    expect(translated).toContain(`[Page B](${publicationBase}/en/page-b/#section)`);
    expect(() => readFileSync(join(options.outputDir, 'en', 'page-b.md'), 'utf8')).toThrow();

    const sidebarSource = readFileSync(
      join(options.rootDir, 'site', 'generated-sidebar.mjs'),
      'utf8'
    );
    const sidebar = JSON.parse(sidebarSource
      .replace('// Generated from publishing/navigation.yml. Do not edit.\nexport default ', '')
      .replace(/;\n$/, ''));
    expect(sidebar).toEqual([{
      label: 'Учебник',
      translations: { en: 'Textbook' },
      items: [
        { label: '28. Первая страница', translations: { en: '28. Page A in English' }, slug: 'nested/page-a' },
        { label: 'Вторая страница', slug: 'page-b' }
      ]
    }]);
  });

  it('keeps deep source archives out of textbook search without removing their pages', async () => {
    const options = fixture('Course archive', 'Textbook chapter');
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: textbook',
      '    title: Textbook',
      '    pages:',
      '      - source: Notes/Page B.md',
      '        route: textbook/page-b',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      '      - source: Notes/Page A.md',
      '        route: sources/courses/example/lecture'
    ].join('\n'));

    await buildPublication(options);

    const chapter = matter(readFileSync(
      join(options.outputDir, 'textbook', 'page-b.md'),
      'utf8'
    )).data;
    const archive = matter(readFileSync(
      join(options.outputDir, 'sources', 'courses', 'example', 'lecture.md'),
      'utf8'
    )).data;
    expect(chapter).not.toHaveProperty('pagefind');
    expect(archive.pagefind).toBe(false);
  });

  it('makes protocol-relative links in imported HTML explicitly external', async () => {
    const options = fixture(
      '<a href="//commons.wikimedia.org/wiki/User:Chire">Chire</a>\n' +
      '<img src="//upload.wikimedia.org/example.png" alt="Example">'
    );

    await buildPublication(options);

    const generated = readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8');
    expect(generated).toContain('href="https://commons.wikimedia.org/wiki/User:Chire"');
    expect(generated).toContain('src="https://upload.wikimedia.org/example.png"');
  });

  it('resolves Russian canonical paths from English pages when the target is translated', async () => {
    const options = fixture('Русская страница A', 'Русская страница B');
    write(join(options.rootDir, 'English', 'Page A.md'), [
      '---', 'title: Page A', 'type: concept', 'status: stable', '---',
      'See [[Notes/Page B|translated B]].'
    ].join('\n'));
    write(join(options.rootDir, 'English', 'Page B.md'), [
      '---', 'title: Page B', 'type: concept', 'status: stable', '---', 'English B.'
    ].join('\n'));
    write(options.manifestPath, [
      'site_title: Fixture', 'sections:', '  - id: textbook', '    title: Textbook', '    pages:',
      '      - source: Notes/Page A.md', '        source_en: English/Page A.md', '        route: page-a',
      '      - source: Notes/Page B.md', '        source_en: English/Page B.md', '        route: page-b'
    ].join('\n'));

    await buildPublication(options);

    expect(readFileSync(join(options.outputDir, 'en', 'page-a.md'), 'utf8'))
      .toContain(`[translated B](${publicationBase}/en/page-b/)`);
  });

  it('percent-encodes copied asset paths so Markdown renders them as images', async () => {
    const options = fixture('![[Assets/Figures With Spaces/chart one.svg]]');
    write(join(options.rootDir, 'Assets', 'Figures With Spaces', 'chart one.svg'), '<svg/>');

    await buildPublication(options);

    const generated = readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8');
    const assetHref = `${publicationBase}/assets/Figures%20With%20Spaces/chart%20one.svg`;
    expect(generated).toContain(`[![chart one](${assetHref})](${assetHref})`);
  });

  it('preserves manifest-defined nested textbook groups, labels, and slugs', async () => {
    const options = fixture('Body');
    const pages = [
      ['Index', '00 Учебник/_index.md', 'textbook/index'],
      ['How to use', '00 Учебник/00 Как пользоваться учебником.md', 'textbook/how-to-use'],
      ['Tensors', '00 Учебник/00 Математические и ML-основания/01 Tensors.md', 'textbook/foundations/tensors'],
      ['Gradients', '00 Учебник/00 Математические и ML-основания/02 Gradients.md', 'textbook/foundations/gradients'],
      ['Decoding', '00 Учебник/14 Inference и оптимизация/54 Decoding.md', 'textbook/inference/decoding']
    ] as const;
    for (const [title, source] of pages) {
      write(join(options.rootDir, source), [
        '---',
        `title: ${title}`,
        'type: textbook-chapter',
        'status: stable',
        '---',
        'Body'
      ].join('\n'));
    }
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: textbook',
      '    title: Учебник',
      '    pages:',
      ...pages.flatMap(([, source, route]) => [
        `      - source: ${source}`,
        `        route: ${route}`
      ]),
      '    sidebar:',
      '      - label: Index',
      '        route: textbook/index',
      '      - label: How to use',
      '        route: textbook/how-to-use',
      '      - label: I. Foundations',
      '        items:',
      '          - label: Core mathematics',
      '            items:',
      '              - label: 1. Tensors',
      '                route: textbook/foundations/tensors',
      '              - label: 2. Gradients',
      '                route: textbook/foundations/gradients',
      '      - label: VII. Serving',
      '        items:',
      '          - label: 68. Decoding',
      '            route: textbook/inference/decoding'
    ].join('\n'));

    await buildPublication(options);

    const sidebarSource = readFileSync(
      join(options.rootDir, 'site', 'generated-sidebar.mjs'),
      'utf8'
    );
    const sidebar = JSON.parse(sidebarSource
      .replace('// Generated from publishing/navigation.yml. Do not edit.\nexport default ', '')
      .replace(/;\n$/, ''));
    expect(sidebar).toEqual([{
      label: 'Учебник',
      items: [
        { label: 'Index', slug: 'textbook/index' },
        { label: 'How to use', slug: 'textbook/how-to-use' },
        {
          label: 'I. Foundations',
          collapsed: true,
          items: [
            {
              label: 'Core mathematics',
              collapsed: true,
              items: [
                { label: '1. Tensors', slug: 'textbook/foundations/tensors' },
                { label: '2. Gradients', slug: 'textbook/foundations/gradients' }
              ]
            }
          ]
        },
        {
          label: 'VII. Serving',
          collapsed: true,
          items: [{ label: '68. Decoding', slug: 'textbook/inference/decoding' }]
        }
      ]
    }]);
  });

  it('shows only source index pages in the source sidebar while building hidden pages', async () => {
    const options = fixture('Body');
    const pages = [
      ['Reference', 'reference/index'],
      ['Research', 'research/index'],
      ['Sources', 'sources/index'],
      ['Legacy', 'reference/legacy/old-page'],
      ['Legacy status', 'reference/old-concept'],
      ['Redirect', 'reference/redirect-old'],
      ['Paper', 'sources/papers/paper'],
      ['Course', 'sources/courses/course']
    ] as const;
    for (const [title, route] of pages) {
      write(join(options.rootDir, 'Notes', `${title}.md`), [
        '---',
        `title: ${title}`,
        'type: concept',
        `status: ${title === 'Redirect' ? 'redirect' : title === 'Legacy status' ? 'legacy' : 'stable'}`,
        '---',
        'Body'
      ].join('\n'));
    }
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Source fixture',
      '    pages:',
      ...pages.flatMap(([title, route]) => [
        `      - source: Notes/${title}.md`,
        `        route: ${route}`
      ])
    ].join('\n'));

    await buildPublication(options);

    const sidebarSource = readFileSync(
      join(options.rootDir, 'site', 'generated-sidebar.mjs'),
      'utf8'
    );
    const sidebar = JSON.parse(sidebarSource
      .replace('// Generated from publishing/navigation.yml. Do not edit.\nexport default ', '')
      .replace(/;\n$/, ''));
    expect(sidebar).toEqual([{
      label: 'Source fixture',
      items: [{ label: 'Sources', slug: 'sources/index' }]
    }]);
    for (const [, route] of pages) {
      expect(readFileSync(join(options.outputDir, `${route}.md`), 'utf8')).toContain('Body');
    }
  });

  it('combines mechanisms, models, research overviews and timeline in the reference sidebar', async () => {
    const options = fixture('Body');
    const pages = [
      ['Models', 'models/index'],
      ['Timeline', 'models/timeline'],
      ['Reference', 'reference/index'],
      ['Overview', 'research/index'],
      ['Sources', 'sources/index']
    ] as const;
    for (const [title] of pages) {
      write(join(options.rootDir, 'Notes', `${title}.md`), [
        '---',
        `title: ${title}`,
        'type: concept',
        'status: stable',
        '---',
        'Body'
      ].join('\n'));
    }
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: models',
      '    title: Справочник',
      '    pages:',
      ...pages.slice(0, 2).flatMap(([title, route]) => [
        `      - source: Notes/${title}.md`,
        `        route: ${route}`
      ]),
      '  - id: sources',
      '    title: Источники',
      '    pages:',
      ...pages.slice(2).flatMap(([title, route]) => [
        `      - source: Notes/${title}.md`,
        `        route: ${route}`
      ])
    ].join('\n'));

    await buildPublication(options);

    const sidebarSource = readFileSync(
      join(options.rootDir, 'site', 'generated-sidebar.mjs'),
      'utf8'
    );
    const sidebar = JSON.parse(sidebarSource
      .replace('// Generated from publishing/navigation.yml. Do not edit.\nexport default ', '')
      .replace(/;\n$/, ''));
    expect(sidebar).toEqual([
      {
        label: 'Справочник',
        items: [
          { label: 'Механизмы', translations: { en: 'Mechanisms' }, items: [{ label: 'Reference', slug: 'reference/index' }] },
          { label: 'Модели и семейства', translations: { en: 'Models and families' }, items: [{ label: 'Models', slug: 'models/index' }] },
          { label: 'Обзоры направлений', translations: { en: 'Research overviews' }, items: [{ label: 'Overview', slug: 'research/index' }] },
          { label: 'Хронология', translations: { en: 'Timeline' }, items: [{ label: 'Timeline', slug: 'models/timeline' }] }
        ]
      },
      { label: 'Источники', items: [{ label: 'Sources', slug: 'sources/index' }] }
    ]);
  });

  it('omits exactly one leading source H1 even when it differs from the page title', async () => {
    const options = fixture('\n# Short source heading\n\n# Second heading\n\nBody');

    await buildPublication(options);

    const generated = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8'));
    expect(generated.content).toBe('\n# Second heading\n\nBody\n');
  });

  it('preserves content that has no leading H1', async () => {
    const options = fixture('\n## First subsection\n\nBody');

    await buildPublication(options);

    const generated = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8'));
    expect(generated.content).toBe('\n## First subsection\n\nBody\n');
  });

  it('fails when a wiki target is neither published nor explicitly allowlisted', async () => {
    const options = fixture('[[Unknown target]]');
    await expect(buildPublication(options)).rejects.toThrow(
      'Unexplained unresolved wiki links:\n- Notes/Page A.md: Unknown target'
    );
  });

  it('reports unresolved links from hidden legacy pages without failing canonical publication', async () => {
    const options = fixture('Body');
    write(join(options.rootDir, 'Notes', 'Legacy.md'), [
      '---',
      'title: Legacy',
      'type: concept',
      'status: legacy',
      '---',
      'See [[Removed old note]].'
    ].join('\n'));
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      '      - source: Notes/Legacy.md',
      '        route: reference/old-concept'
    ].join('\n'));

    await expect(buildPublication(options)).resolves.toBeUndefined();
    const report = JSON.parse(readFileSync(options.reportPath, 'utf8'));
    expect(report.pages).toEqual([{
      source: 'Notes/Legacy.md',
      route: 'reference/old-concept',
      unresolved: [],
      allowlisted: [{
        target: 'Removed old note',
        reason: 'legacy page preserved for old links; not part of the canonical publication'
      }]
    }]);
  });

  it('places fragment aliases directly before their matching headings', async () => {
    const options = fixture(
      '[[Page B#First Heading|first]] and [[Page B#Chat template|chat]]',
      'Intro\n\n## First Heading\n\nOne\n\n## Chat template\n\nTwo'
    );
    await buildPublication(options);

    const source = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8')).content;
    const target = matter(readFileSync(join(options.outputDir, 'page-b.md'), 'utf8')).content;
    expect(source).toContain(`[first](${publicationBase}/page-b/#first-heading)`);
    expect(source).toContain(`[chat](${publicationBase}/page-b/#chat-template)`);
    expect(target).toContain('## First Heading');
    expect(target).toContain('## Chat template');
    expect(target).not.toContain('aria-hidden="true"');
  });

  it('ignores mixed fence delimiters and headings inside tilde fences', () => {
    const markdown = [
      '~~~text',
      '```',
      '# Fake heading',
      '  ```',
      '~~~~',
      '# Real heading'
    ].join('\n');

    expect(parseMarkdownHeadings(markdown).map(({ text }) => text)).toEqual(['Real heading']);
    const inserted = insertFragmentAliases(markdown, new Map([
      ['Fake heading', 'wiki-fake-heading'],
      ['Real heading', 'wiki-real-heading']
    ]));
    expect(inserted.missing).toEqual(['Fake heading']);
    expect(inserted.markdown).not.toContain('wiki-fake-heading');
    expect(inserted.markdown).toContain(
      '<span id="wiki-real-heading" aria-hidden="true"></span>\n# Real heading'
    );
  });

  it('requires a matching fence character and a closing delimiter at least as long as its opener', () => {
    const markdown = [
      '````js',
      '```',
      '# Still fenced',
      '~~~~',
      '    ````',
      '# Also fenced',
      '`````',
      '# Real after longer close',
      '```',
      '# Hidden by short opener',
      '````',
      '# Real after valid longer close'
    ].join('\n');

    expect(parseMarkdownHeadings(markdown).map(({ text }) => text)).toEqual([
      'Real after longer close',
      'Real after valid longer close'
    ]);
  });

  it('rejects backtick fence openers whose info string contains a backtick', () => {
    const markdown = ['```lang`variant', '# Parsed heading'].join('\n');

    expect(parseMarkdownHeadings(markdown).map(({ text }) => text)).toEqual(['Parsed heading']);
  });

  it('allows backticks in a tilde fence info string', () => {
    const markdown = [
      '~~~lang`variant',
      '# Fenced heading',
      '~~~',
      '# Parsed heading'
    ].join('\n');

    expect(parseMarkdownHeadings(markdown).map(({ text }) => text)).toEqual(['Parsed heading']);
  });

  it('preserves an H1 that appears after body content', async () => {
    const options = fixture('\nIntroduction.\n\n# Later heading\n\nBody');

    await buildPublication(options);

    const generated = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8'));
    expect(generated.content).toBe('\nIntroduction.\n\n# Later heading\n\nBody\n');
  });

  it('removes stale generated files but preserves siblings outside the known output root', async () => {
    const options = fixture('Body');
    write(join(options.outputDir, 'stale.md'), 'stale');
    const sibling = join(dirname(options.outputDir), 'keep.md');
    write(sibling, 'keep');

    await buildPublication(options);

    expect(() => readFileSync(join(options.outputDir, 'stale.md'), 'utf8')).toThrow();
    expect(readFileSync(sibling, 'utf8')).toBe('keep');
  });

  it('removes stale publication assets after successful preflight', async () => {
    const options = fixture('Body');
    const stale = join(options.rootDir, 'site', 'public', 'assets', 'private', 'stale.png');
    write(stale, 'must be removed');

    await buildPublication(options);

    expect(() => readFileSync(stale, 'utf8')).toThrow();
  });

  it.each(['../outside.svg', '/tmp/outside.svg'])(
    'rejects an asset path outside rootDir: %s',
    async (asset) => {
      const options = fixture(`![[${asset}]]`);
      await expect(buildPublication(options)).rejects.toThrow(/Asset path escapes rootDir/);
    }
  );

  it('fails when a referenced local asset is missing', async () => {
    const options = fixture('![[Assets/Figures/missing.svg]]');
    const stale = join(options.outputDir, 'stale.md');
    const staleAsset = join(options.rootDir, 'site', 'public', 'assets', 'stale.png');
    write(stale, 'must survive failed preflight');
    write(staleAsset, 'must survive failed preflight');

    await expect(buildPublication(options)).rejects.toThrow(/Missing asset.*missing\.svg/);
    expect(readFileSync(stale, 'utf8')).toBe('must survive failed preflight');
    expect(readFileSync(staleAsset, 'utf8')).toBe('must survive failed preflight');
  });

  it('rejects distinct source assets that normalize to the same public path before cleanup', async () => {
    const options = fixture([
      '![[Assets/Figures/chart.svg]]',
      '![[Figures/chart.svg]]'
    ].join('\n'));
    write(join(options.rootDir, 'Figures', 'chart.svg'), '<svg>different</svg>');
    const stale = join(options.outputDir, 'stale.md');
    write(stale, 'must survive failed preflight');

    await expect(buildPublication(options)).rejects.toThrow(
      /Asset destination collision.*Figures\/chart\.svg/
    );
    expect(readFileSync(stale, 'utf8')).toBe('must survive failed preflight');
  });

  it('deduplicates repeated references to the same source and public path', async () => {
    const options = fixture([
      '![[Assets/Figures/chart.svg]]',
      '![[Assets/Figures/chart.svg]]'
    ].join('\n'));

    await buildPublication(options);

    expect(readFileSync(
      join(options.rootDir, 'site', 'public', 'assets', 'Figures', 'chart.svg'),
      'utf8'
    )).toBe('<svg>fixture</svg>');
  });

  it('rewrites a standard Markdown link to a registered publication page', async () => {
    const options = fixture('[Page B](Page B.md)');

    await buildPublication(options);

    const generated = readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8');
    expect(generated).toContain(`[Page B](${publicationBase}/page-b/)`);
  });

  it('rewrites registered course notes to routes and copies only linked source artifacts route-relatively', async () => {
    const options = fixture('unused');
    const courseSource = '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025';
    const courseRoute = 'sources/courses/berkeley-advanced-llm-agents-spring-2025';
    const hub = join(options.rootDir, courseSource, '_index.md');
    const reading = join(options.rootDir, courseSource, 'Readings', 'meeting-01-reading-01.md');
    const linkedArtifacts = [
      ['Lectures/deck.pdf', 'pdf'],
      ['Lectures/lecture.py', 'python'],
      ['audit.json', 'json'],
      ['coverage.yml', 'yaml'],
      ['page-index.txt', 'text']
    ] as const;
    write(hub, [
      '---',
      'title: Course hub',
      'type: source-note',
      'status: verified',
      '---',
      '[Reading](Readings/meeting-01-reading-01.md)',
      ...linkedArtifacts.map(([path]) => `[Artifact](${path})`)
    ].join('\n'));
    write(reading, [
      '# Reading',
      'Source-native reading.'
    ].join('\n'));
    for (const [path, contents] of linkedArtifacts) {
      write(join(options.rootDir, courseSource, path), contents);
    }
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      `      - source: ${courseSource}/_index.md`,
      `        route: ${courseRoute}/index`,
      `      - source: ${courseSource}/Readings/meeting-01-reading-01.md`,
      `        route: ${courseRoute}/readings/meeting-01-reading-01`
    ].join('\n'));

    await buildPublication(options);

    const generated = readFileSync(join(options.outputDir, courseRoute, 'index.md'), 'utf8');
    expect(generated).toContain(
      `[Reading](${publicationBase}/${courseRoute}/readings/meeting-01-reading-01/)`
    );
    for (const [path, contents] of linkedArtifacts) {
      expect(generated).toContain(
        `[Artifact](${publicationBase}/${courseRoute}/${path.split('/').map(encodeURIComponent).join('/')})`
      );
      expect(readFileSync(
        join(options.rootDir, 'site', 'public', courseRoute, path),
        'utf8'
      )).toBe(contents);
    }
  });

  it('rejects a linked course artifact that escapes the publication root before cleanup', async () => {
    const options = fixture('unused');
    const courseSource = '05 Источники/Courses/Stanford CS336 Spring 2026';
    const hub = join(options.rootDir, courseSource, '_index.md');
    write(hub, [
      '---',
      'title: Course hub',
      'type: source-note',
      'status: verified',
      '---',
      '[Escape](../../../../outside.pdf)'
    ].join('\n'));
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      `      - source: ${courseSource}/_index.md`,
      '        route: sources/courses/stanford-cs336-spring-2026/index'
    ].join('\n'));
    const stale = join(options.outputDir, 'stale.md');
    write(stale, 'must survive failed preflight');

    await expect(buildPublication(options)).rejects.toThrow(/Linked source artifact path escapes rootDir/);
    expect(readFileSync(stale, 'utf8')).toBe('must survive failed preflight');
  });

  it.each([
    [
      '05 Источники/Courses/Stanford CS336 Spring 2026',
      'sources/courses/stanford-cs336-spring-2026'
    ],
    [
      '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025',
      'sources/courses/berkeley-advanced-llm-agents-spring-2025'
    ]
  ])(
    'rejects a %s artifact symlink that resolves outside its exact course root before cleanup',
    async (courseSource, courseRoute) => {
      const options = fixture('unused');
      const hub = join(options.rootDir, courseSource, '_index.md');
      const secret = join(options.rootDir, 'private', 'secret.pdf');
      const leak = join(options.rootDir, courseSource, 'Lectures', 'leak.pdf');
      write(secret, 'PRIVATE_VAULT_CONTENT');
      mkdirSync(dirname(leak), { recursive: true });
      symlinkSync(secret, leak);
      write(hub, [
        '---',
        'title: Course hub',
        'type: source-note',
        'status: verified',
        '---',
        '[Leak](Lectures/leak.pdf)'
      ].join('\n'));
      write(options.manifestPath, [
        'site_title: Fixture',
        'sections:',
        '  - id: sources',
        '    title: Sources',
        '    pages:',
        `      - source: ${courseSource}/_index.md`,
        `        route: ${courseRoute}/index`
      ].join('\n'));
      const staleOutput = join(options.outputDir, 'stale.md');
      const staleArtifact = join(options.rootDir, 'site', 'public', courseRoute, 'stale.pdf');
      write(staleOutput, 'must survive failed preflight');
      write(staleArtifact, 'must survive failed preflight');

      await expect(buildPublication(options)).rejects.toThrow(/escapes course root/);
      expect(readFileSync(staleOutput, 'utf8')).toBe('must survive failed preflight');
      expect(readFileSync(staleArtifact, 'utf8')).toBe('must survive failed preflight');
    }
  );

  it.each([
    [
      '05 Источники/Courses/Stanford CS336 Spring 2026',
      'sources/courses/stanford-cs336-spring-2026'
    ],
    [
      '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025',
      'sources/courses/berkeley-advanced-llm-agents-spring-2025'
    ]
  ])('removes stale %s artifacts when a later manifest fully unpublishes the course', async (courseSource, courseRoute) => {
    const options = fixture('unused');
    const hub = join(options.rootDir, courseSource, '_index.md');
    const artifact = join(options.rootDir, courseSource, 'Lectures', 'deck.pdf');
    write(hub, [
      '---',
      'title: Course hub',
      'type: source-note',
      'status: verified',
      '---',
      '[Deck](Lectures/deck.pdf)'
    ].join('\n'));
    write(artifact, 'published once');
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      `      - source: ${courseSource}/_index.md`,
      `        route: ${courseRoute}/index`
    ].join('\n'));

    await buildPublication(options);
    const published = join(options.rootDir, 'site', 'public', courseRoute, 'Lectures', 'deck.pdf');
    expect(readFileSync(published, 'utf8')).toBe('published once');

    write(options.sourcePath, [
      '---',
      'title: Page A',
      'type: concept',
      'status: stable',
      '---',
      'Textbook only.'
    ].join('\n'));
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: textbook',
      '    title: Textbook',
      '    pages:',
      '      - source: Notes/Page A.md',
      '        route: page-a'
    ].join('\n'));

    await buildPublication(options);

    expect(() => readFileSync(published, 'utf8')).toThrow();
  });

  it('rejects a pattern-shaped Berkeley reading outside the exact audited set', async () => {
    const options = fixture('unused');
    const courseSource = '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025';
    const courseRoute = 'sources/courses/berkeley-advanced-llm-agents-spring-2025';
    const readingSource = `${courseSource}/Readings/meeting-99-reading-99.md`;
    write(join(options.rootDir, readingSource), '# Unaudited reading\n\nBody.');
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      `      - source: ${readingSource}`,
      `        route: ${courseRoute}/readings/meeting-99-reading-99`
    ].join('\n'));

    await expect(buildPublication(options)).rejects.toThrow(/not in the audited Berkeley reading set/);
  });

  it('rejects an unapproved raw Markdown artifact below a course root', async () => {
    const options = fixture('unused');
    const courseSource = '05 Источники/Courses/Stanford CS336 Spring 2026';
    const courseRoute = 'sources/courses/stanford-cs336-spring-2026';
    const hub = join(options.rootDir, courseSource, '_index.md');
    const rawReadme = 'Assignments/unreviewed/README.md';
    write(hub, [
      '---',
      'title: Course hub',
      'type: source-note',
      'status: verified',
      '---',
      `[Unreviewed](${rawReadme})`
    ].join('\n'));
    write(join(options.rootDir, courseSource, rawReadme), 'unreviewed');
    write(options.manifestPath, [
      'site_title: Fixture',
      'sections:',
      '  - id: sources',
      '    title: Sources',
      '    pages:',
      `      - source: ${courseSource}/_index.md`,
      `        route: ${courseRoute}/index`
    ].join('\n'));

    await expect(buildPublication(options)).rejects.toThrow(/not an approved raw Markdown course artifact/);
  });

  it('refuses to clean rootDir itself as the generated output directory', async () => {
    const options = fixture('Body');
    options.outputDir = options.rootDir;

    await expect(buildPublication(options)).rejects.toThrow(/Output directory must be below rootDir/);
    expect(readFileSync(options.sourcePath, 'utf8')).toContain('title: Page A');
  });
});
