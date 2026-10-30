import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';

const root = resolve(import.meta.dirname, '../..');
type Step = { run?: string; uses?: string; env?: Record<string, string>; with?: Record<string, string> };
const workflow = YAML.parse(readFileSync(resolve(root, '.github/workflows/pages.yml'), 'utf8'));
const build = workflow.jobs.build as { env?: Record<string, string>; steps: Step[] };
const baseFor = (step: Step) => step.env?.PUBLICATION_BASE_PATH ?? build.env?.PUBLICATION_BASE_PATH ?? '';

describe('GitHub Pages release checks', () => {
  it('installs PDF verification tools before the public suite in both workflows', () => {
    for (const file of ['pages.yml', 'verify.yml']) {
      const config = YAML.parse(readFileSync(resolve(root, '.github/workflows', file), 'utf8'));
      const steps = (config.jobs.build ?? config.jobs.test).steps as Step[];
      const install = steps.findIndex(step => step.run?.includes('apt-get install') && step.run.includes('poppler-utils'));
      const suite = steps.findIndex(step => step.run === 'pnpm --dir publishing test:publication');
      expect(install, file).toBeGreaterThanOrEqual(0);
      expect(suite, file).toBeGreaterThan(install);
    }
  });

  it('runs root-only semantic output assertions on a real root build', () => {
    const output = build.steps.findIndex(step => step.run === 'pnpm --dir publishing test:output');
    const rootBuild = build.steps.findIndex(step => step.run === 'pnpm --dir site build' && baseFor(step) === '');
    expect(rootBuild).toBeGreaterThanOrEqual(0);
    expect(output).toBeGreaterThan(rootBuild);
    expect(baseFor(build.steps[output])).toBe('');
  });

  it('rebuilds and checks the actual prefixed artifact before uploading it', () => {
    const output = build.steps.findIndex(step => step.run === 'pnpm --dir publishing test:output');
    const pagesBuild = build.steps.findIndex(step => step.run === 'pnpm --dir site build' && baseFor(step) === '/bookvar');
    const pagesLinks = build.steps.findIndex(step => step.run === 'pnpm --dir publishing check:links' && baseFor(step) === '/bookvar');
    const upload = build.steps.findIndex(step => step.uses?.startsWith('actions/upload-pages-artifact@'));
    expect(pagesBuild).toBeGreaterThan(output);
    expect(pagesLinks).toBeGreaterThan(pagesBuild);
    expect(upload).toBeGreaterThan(pagesLinks);
    expect(build.steps[upload].with?.path).toBe('site/dist');
  });

  it('rejects oversized Pages output before upload', () => {
    const pagesBuild = build.steps.findIndex(step => step.run === 'pnpm --dir site build' && baseFor(step) === '/bookvar');
    const size = build.steps.findIndex(step => step.run === 'pnpm --dir publishing check:size');
    const upload = build.steps.findIndex(step => step.uses?.startsWith('actions/upload-pages-artifact@'));
    expect(size).toBeGreaterThan(pagesBuild);
    expect(upload).toBeGreaterThan(size);
  });

  it('documents the distinction without suggesting root-only tests accept prefixed output', () => {
    const docs = readFileSync(resolve(root, 'docs/PUBLISHING.md'), 'utf8');
    expect(docs).not.toContain('PUBLICATION_BASE_PATH=/bookvar pnpm --dir publishing test:output');
    expect(docs).toContain('PUBLICATION_BASE_PATH=/bookvar pnpm --dir site build');
    expect(docs).toContain('PUBLICATION_BASE_PATH=/bookvar pnpm --dir publishing check:links');
  });
});
