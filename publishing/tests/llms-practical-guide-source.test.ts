import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const sourceRoot = resolve(root, '05 Источники/Courses/LLMs Practical Guide');
const commit = 'c4a39847f5455b8383dffee55c4fe9e5e16966a4';

type FileRecord = {
  source_path: string;
  source_url: string;
  output: string;
  sha256: string;
};

type PageRecord = FileRecord & {
  route: string;
  source_sha256: string;
  source_bytes: number;
  source_lines: number;
  source_words: number;
};

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function manifest(): {
  repository: string;
  commit: string;
  license_status: string;
  page_count: number;
  route_count: number;
  media_count: number;
  editable_source_count: number;
  pages: PageRecord[];
  media: FileRecord[];
  editable_sources: FileRecord[];
} {
  return JSON.parse(readFileSync(resolve(sourceRoot, 'import-manifest.json'), 'utf8'));
}

describe('LLMs Practical Guide source-native archive', () => {
  it('pins the complete substantive corpus and records the absent formal license', () => {
    const data = manifest();
    expect(data).toMatchObject({
      repository: 'https://github.com/Mooler0410/LLMsPracticalGuide',
      commit,
      license_status: 'no-repository-license-found',
      page_count: 4,
      route_count: 5,
      media_count: 10,
      editable_source_count: 2
    });
    expect(data.pages.map(({ source_path }) => source_path)).toEqual([
      'README.md',
      'awesome_examples/social_game_werewolf.md',
      'awesome_examples/tableQA.md',
      'source/README.md'
    ]);
    expect(data.pages.map(({ source_bytes, source_lines }) => ({ source_bytes, source_lines })))
      .toEqual([
        { source_bytes: 33062, source_lines: 583 },
        { source_bytes: 20858, source_lines: 174 },
        { source_bytes: 5584, source_lines: 190 },
        { source_bytes: 198, source_lines: 5 }
      ]);
    expect(data.pages.reduce((sum, page) => sum + page.source_words, 0))
      .toBeGreaterThan(4500);
  });

  it('preserves every source page in its original language without truncation', () => {
    const data = manifest();
    for (const page of data.pages) {
      const output = resolve(root, page.output);
      expect(existsSync(output), `Missing page: ${page.output}`).toBe(true);
      const markdown = readFileSync(output, 'utf8');
      expect(markdown).toContain(`source_commit: ${commit}`);
      expect(markdown).toContain(`source_sha256: ${page.source_sha256}`);
      expect(markdown).toContain('Complete original source');
      expect(markdown).toContain('license_status: no-repository-license-found');
      expect(markdown).toContain(page.source_url);
    }
    expect(readFileSync(resolve(sourceRoot, 'README.md'), 'utf8'))
      .toContain('## Usage and Restrictions');
    expect(readFileSync(resolve(sourceRoot, 'awesome_examples/tableQA.md'), 'utf8'))
      .toContain('# Example 2 (2022/04/29)');
    expect(readFileSync(resolve(sourceRoot, 'awesome_examples/social_game_werewolf.md'), 'utf8'))
      .toContain('## GPT-4 Example 1');
  });

  it('localizes every original image and editable slide source with exact provenance', () => {
    const data = manifest();
    for (const record of [...data.media, ...data.editable_sources]) {
      const output = resolve(root, record.output);
      expect(existsSync(output), `Missing binary source: ${record.output}`).toBe(true);
      expect(sha256(output), `Binary drift: ${record.output}`).toBe(record.sha256);
      expect(record.source_url).toBe(
        `https://github.com/Mooler0410/LLMsPracticalGuide/blob/${commit}/${record.source_path}`
      );
    }
    const index = readFileSync(
      resolve(root, '05 Источники/Courses/LLMs Practical Guide.md'),
      'utf8'
    );
    for (const image of data.media) {
      expect(index).toContain(`![[${image.output}|700]]`);
      expect(index).toContain(image.source_url);
      expect(index).toContain(image.sha256);
    }
  });

  it('publishes the index and all four complete pages on five unique routes', () => {
    const data = manifest();
    expect(new Set(data.pages.map(({ route }) => route)).size).toBe(4);
    const navigation = readFileSync(resolve(root, 'publishing/navigation.yml'), 'utf8');
    const routes = navigation.match(
      /^\s+route: sources\/courses\/llms-practical-guide(?:\/[^\s]+)?$/gm
    ) ?? [];
    expect(routes).toHaveLength(5);
    expect(navigation).toContain(
      'source: 05 Источники/Courses/LLMs Practical Guide.md'
    );
    for (const page of data.pages) {
      expect(navigation).toContain(`source: ${page.output}`);
      expect(navigation).toContain(`route: ${page.route}`);
    }
  });
});
