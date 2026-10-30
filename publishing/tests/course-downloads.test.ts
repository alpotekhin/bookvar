import { existsSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadCoursePdfDownloads } from '../adapter/course-downloads.js';
import { buildPublication } from '../adapter/build.js';

const course = '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025';
function fixture(overrides: Record<string, unknown> = {}) {
  const rootDir = mkdtempSync(join(tmpdir(), 'bookvar-downloads-'));
  const write = (path: string, contents: string) => {
    mkdirSync(dirname(join(rootDir, path)), { recursive: true });
    writeFileSync(join(rootDir, path), contents);
  };
  write(`${course}/source-manifest.yml`, JSON.stringify({ objects: [{
    kind: 'pdf', source_authority: 'official-course', verification_status: 'verified',
    local_path: 'Lectures/deck.pdf', canonical_url: 'https://example.edu/deck.pdf', ...overrides
  }] }));
  return { rootDir, write };
}

describe('official course PDF delivery', () => {
  it('uses only verified official PDF records and leaves unknown files under the existing copy policy', async () => {
    const { rootDir } = fixture();
    const downloads = await loadCoursePdfDownloads(rootDir, [course, 'Absent course']);
    expect([...downloads]).toEqual([[`${course}/Lectures/deck.pdf`, 'https://example.edu/deck.pdf']]);
    for (const overrides of [{ kind: 'html' }, { source_authority: 'community' }, { verification_status: 'unverified' }]) {
      expect((await loadCoursePdfDownloads(fixture(overrides).rootDir, [course])).size).toBe(0);
    }
  });

  it('rejects path escape, credentials and non-HTTPS delivery', async () => {
    for (const overrides of [{ local_path: '../private.pdf' }, { local_path: '/private.pdf' },
      { canonical_url: 'http://example.edu/deck.pdf' }, { canonical_url: 'https://name:secret@example.edu/deck.pdf' }]) {
      await expect(loadCoursePdfDownloads(fixture(overrides).rootDir, [course])).rejects.toThrow();
    }
  });

  it('opens the full PDF bytes for a commit-pinned official GitHub file', async () => {
    const revision = 'a'.repeat(40);
    const { rootDir } = fixture({ canonical_url: `https://github.com/org/lectures/blob/${revision}/lecture_09.pdf` });
    expect((await loadCoursePdfDownloads(rootDir, [course])).get(`${course}/Lectures/deck.pdf`))
      .toBe(`https://raw.githubusercontent.com/org/lectures/${revision}/lecture_09.pdf`);
  });

  it('keeps the local original and page fragment, while not duplicating the PDF into Pages', async () => {
    const { rootDir, write } = fixture();
    const route = 'sources/courses/berkeley-advanced-llm-agents-spring-2025';
    write(`${course}/_index.md`, '---\ntitle: Course\ntype: source-note\nstatus: verified\n---\n[Full PDF](Lectures/deck.pdf#page=7)\n[Code](Lectures/example.py)\n');
    write(`${course}/Lectures/deck.pdf`, 'unchanged original fixture');
    write(`${course}/Lectures/example.py`, 'print(1)\n');
    write('publishing/navigation.yml', `site_title: Fixture\nsections:\n  - id: sources\n    title: Sources\n    pages:\n      - source: ${course}/_index.md\n        route: ${route}/index\n`);
    write('publishing/link-allowlist.json', '{"entries":[]}');
    const outputDir = join(rootDir, 'site/src/content/docs/generated');
    await buildPublication({ rootDir, outputDir, manifestPath: join(rootDir, 'publishing/navigation.yml'), reportPath: join(rootDir, 'report.json') });
    const generated = readFileSync(join(outputDir, route, 'index.md'), 'utf8');
    expect(generated).toContain('[Full PDF](https://example.edu/deck.pdf#page=7)');
    expect(readFileSync(join(rootDir, course, 'Lectures/deck.pdf'), 'utf8')).toBe('unchanged original fixture');
    expect(existsSync(join(rootDir, 'site/public', route, 'Lectures/deck.pdf'))).toBe(false);
    expect(readFileSync(join(rootDir, 'site/public', route, 'Lectures/example.py'), 'utf8')).toBe('print(1)\n');
  });

  it('builds a recorded external PDF link without requiring a local verification copy', async () => {
    const { rootDir, write } = fixture();
    const route = 'sources/courses/berkeley-advanced-llm-agents-spring-2025';
    write(`${course}/_index.md`, '---\ntitle: Course\ntype: source-note\nstatus: verified\n---\n[Full PDF](Lectures/deck.pdf#page=7)\n');
    write('publishing/navigation.yml', `site_title: Fixture\nsections:\n  - id: sources\n    title: Sources\n    pages:\n      - source: ${course}/_index.md\n        route: ${route}/index\n`);
    write('publishing/link-allowlist.json', '{"entries":[]}');
    const outputDir = join(rootDir, 'generated');
    await buildPublication({ rootDir, outputDir, manifestPath: join(rootDir, 'publishing/navigation.yml'), reportPath: join(rootDir, 'report.json') });
    expect(readFileSync(join(outputDir, route, 'index.md'), 'utf8'))
      .toContain('[Full PDF](https://example.edu/deck.pdf#page=7)');
    expect(existsSync(join(rootDir, course, 'Lectures/deck.pdf'))).toBe(false);
  });
});
