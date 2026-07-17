import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
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
  it('loads the representative handbook manifest with at least eight real pages', () => {
    const path = join(import.meta.dirname, '..', 'navigation.yml');
    const manifest = loadManifest(path);

    expect(manifest.sections.map((section) => section.id)).toEqual([
      'textbook', 'models', 'sources', 'questions', 'practice'
    ]);
    expect(manifest.sections.flatMap((section) => section.pages)).toHaveLength(10);
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

  it('rejects a missing source file', () => {
    expect(() => loadManifest(fixture(validYaml, ['00 Учебник/_index.md'])))
      .toThrow(/Missing source.*06 Практика\/lab\.md/);
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
});
