import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');

describe('publication workflow policy', () => {
  it('separates unit tests from built-output tests and runs them around the build', () => {
    const publishing = JSON.parse(readFileSync(resolve(root, 'publishing/package.json'), 'utf8'));
    const workflow = readFileSync(resolve(root, '.github/workflows/verify.yml'), 'utf8');

    expect(publishing.scripts['test:unit']).toBe('vitest run --exclude tests/site-output.test.ts');
    expect(publishing.scripts['test:output']).toBe('vitest run tests/site-output.test.ts');
    expect(workflow.indexOf('pnpm --dir publishing test:unit'))
      .toBeLessThan(workflow.indexOf('pnpm --dir site build'));
    expect(workflow.indexOf('pnpm --dir site build'))
      .toBeLessThan(workflow.indexOf('pnpm --dir publishing test:output'));
  });

  it('does not track a second navigation source and generates the ignored sidebar before consumers', () => {
    const ignore = readFileSync(resolve(root, '.gitignore'), 'utf8');
    const site = JSON.parse(readFileSync(resolve(root, 'site/package.json'), 'utf8'));

    expect(ignore).toContain('site/generated-sidebar.mjs');
    expect(site.scripts.precheck).toBe('pnpm content:build');
    expect(site.scripts.prebuild).toBe('pnpm content:build');
  });

  it('pins every direct dependency to an exact version', () => {
    for (const manifest of ['publishing/package.json', 'site/package.json']) {
      const pkg = JSON.parse(readFileSync(resolve(root, manifest), 'utf8'));
      for (const [name, spec] of Object.entries({ ...pkg.dependencies, ...pkg.devDependencies })) {
        expect(spec, `${manifest}: ${name}`).toMatch(/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
      }
    }
  });
});
