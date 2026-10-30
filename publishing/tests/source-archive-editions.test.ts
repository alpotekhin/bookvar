import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { buildPublication } from '../adapter/build.js';

function fixture(route = 'sources/courses/example/topic', sourceEn = false) {
  const rootDir = mkdtempSync(join(tmpdir(), 'bookvar-source-editions-'));
  const write = (path: string, body: string) => {
    mkdirSync(dirname(join(rootDir, path)), { recursive: true });
    writeFileSync(join(rootDir, path), body);
  };
  write('Notes/source.md', `---\ntitle: Original\ntype: source-note\nstatus: verified\nsource_language: en\n---\n## Part\n\nFULL ORIGINAL EXPLANATION\n\n![[Assets/chart.svg]]\n\n[Section](/en/${route.replace(/\/index$/, '')}/#part)\n\n\x60\x60\x60python\nurl = "/en/${route}/"\n\x60\x60\x60\n`);
  write('Notes/english.md', '---\ntitle: English edition\ntype: source-note\nstatus: verified\n---\nSEPARATE EDITION\n');
  write('Assets/chart.svg', '<svg xmlns="http://www.w3.org/2000/svg"/>');
  write('publishing/link-allowlist.json', '{"entries":[]}');
  write('publishing/navigation.yml', `site_title: Fixture\nsections:\n  - id: sources\n    title: Sources\n    pages:\n      - source: Notes/source.md\n        route: ${route}\n${sourceEn ? '        source_en: Notes/english.md\n' : ''}`);
  return { rootDir, route, options: {
    rootDir, manifestPath: join(rootDir, 'publishing/navigation.yml'),
    outputDir: join(rootDir, 'generated'), reportPath: join(rootDir, 'report.json')
  } };
}

describe('one complete source-language course original', () => {
  it('keeps all prose, code and images in the original and makes EN a fragment-preserving direct alias', async () => {
    const previous = process.env.PUBLICATION_BASE_PATH;
    try {
      for (const base of ['', '/bookvar']) {
        process.env.PUBLICATION_BASE_PATH = base;
        const { rootDir, route, options } = fixture();
        await buildPublication(options);
        const original = readFileSync(join(options.outputDir, `${route}.md`), 'utf8');
        const alias = readFileSync(join(options.outputDir, `en/${route}.md`), 'utf8');
        expect(original).toContain('FULL ORIGINAL EXPLANATION');
        expect(original).toContain(`url = "/en/${route}/"`);
        expect(original).toContain(`[Section](${base}/${route}/#part)`);
        expect(original).toContain(`${base}/assets/chart.svg`);
        expect(readFileSync(join(rootDir, 'site/public/assets/chart.svg'), 'utf8')).toContain('<svg');
        expect(alias).toContain(`0;url=${base}/${route}/`);
        expect(alias).toContain('template: splash');
        expect(alias).toContain('pagefind: false');
        expect(alias).toContain('id="part"');
        expect(alias).toContain('location.search + location.hash');
        expect(alias).not.toContain('FULL ORIGINAL EXPLANATION');
        expect(alias).not.toContain('```python');
      }
    } finally {
      if (previous === undefined) delete process.env.PUBLICATION_BASE_PATH;
      else process.env.PUBLICATION_BASE_PATH = previous;
    }
  });

  it('does not deduplicate real English editions or localized course hubs', async () => {
    for (const data of [fixture('sources/courses/example/topic', true), fixture('sources/courses/example/index')]) {
      await buildPublication(data.options);
      const en = readFileSync(join(data.options.outputDir, `en/${data.route}.md`), 'utf8');
      expect(en).not.toContain('Read the complete course original');
      expect(en).toContain(data.route.endsWith('/index') ? 'FULL ORIGINAL EXPLANATION' : 'SEPARATE EDITION');
    }
  });
});
