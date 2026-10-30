import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import publication from '../vitest.publication.config';
import evidence from '../vitest.evidence.config';

const root = resolve(import.meta.dirname, '..');
const catalog = JSON.parse(readFileSync(resolve(root, 'tests/source-evidence-suites.json'), 'utf8'));

describe('explicit publication and source-evidence scopes', () => {
  it('retains the complete unit command and a separate strict evidence command', () => {
    const scripts = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8')).scripts;
    expect(scripts['test:unit']).toBe('vitest run --exclude tests/site-output.test.ts');
    expect(scripts['test:evidence']).toBe('vitest run --config vitest.evidence.config.ts');
    expect(scripts['test:publication']).toBe('vitest run --config vitest.publication.config.ts');
  });
  it('accounts for every test file without deletion or presence-based skipping', () => {
    const files = readdirSync(resolve(root, 'tests')).filter(name => name.endsWith('.test.ts') && name !== 'site-output.test.ts').map(name => 'tests/' + name);
    const sourceFiles: string[] = catalog.suites.map((row: { file: string; requires: string }) => {
      expect(row.requires).toBe('local-source-snapshots');
      expect(files).toContain(row.file);
      return row.file;
    });
    expect(new Set(sourceFiles).size).toBe(sourceFiles.length);
    expect(evidence.test?.include).toEqual(sourceFiles);
    for (const file of sourceFiles) expect(publication.test?.exclude).toContain(file);
    const publicFiles = files.filter(file => !sourceFiles.includes(file));
    expect(publicFiles.length).toBeGreaterThan(50);
    for (const file of ['build','manifest','links','routes','reading-paths','publication-policy',
      'topic-consolidation','english-reconciliation','reference-index-sync','source-archive-editions',
      'public-assets','check-built-links','check-built-size','pages-workflow']) {
      expect(publicFiles).toContain(`tests/${file}.test.ts`);
    }
    expect(publicFiles.length + sourceFiles.length).toBe(files.length);
  });
});
