import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
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
    expect(publicationHref('textbook/index')).toBe('/textbook/');
    expect(publicationHref('textbook/chapter')).toBe('/textbook/chapter/');
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
    expect(generated).toContain('[Page B](/page-b/)');
    expect(generated).toContain('Missing');
    expect(generated).toContain('![chart](/assets/Figures/chart.svg)');
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

  it('places fragment aliases directly before their matching headings', async () => {
    const options = fixture(
      '[[Page B#First Heading|first]] and [[Page B#Chat template|chat]]',
      'Intro\n\n## First Heading\n\nOne\n\n## Chat template\n\nTwo'
    );
    await buildPublication(options);

    const source = matter(readFileSync(join(options.outputDir, 'nested', 'page-a.md'), 'utf8')).content;
    const target = matter(readFileSync(join(options.outputDir, 'page-b.md'), 'utf8')).content;
    expect(source).toContain('[first](/page-b/#wiki-first-heading)');
    expect(source).toContain('[chat](/page-b/#wiki-chat-template)');
    expect(target).toContain(
      '<span id="wiki-first-heading" aria-hidden="true"></span>\n## First Heading'
    );
    expect(target).toContain(
      '<span id="wiki-chat-template" aria-hidden="true"></span>\n## Chat template'
    );
    expect(target).not.toMatch(/^<span id="wiki-/);
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

  it('refuses to clean rootDir itself as the generated output directory', async () => {
    const options = fixture('Body');
    options.outputDir = options.rootDir;

    await expect(buildPublication(options)).rejects.toThrow(/Output directory must be below rootDir/);
    expect(readFileSync(options.sourcePath, 'utf8')).toContain('title: Page A');
  });
});
