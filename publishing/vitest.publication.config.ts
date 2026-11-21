import { readFileSync } from 'node:fs';
import { configDefaults, defineConfig, mergeConfig } from 'vitest/config';
import base, { sourceArchiveTag } from './vitest.config';

const catalog = JSON.parse(readFileSync(new URL('./tests/source-evidence-suites.json', import.meta.url), 'utf8'));
export default mergeConfig(base, defineConfig({
  test: {
    // Historical revision checks reconstruct chapters from many journals on cold CI runners.
    testTimeout: 30_000,
    include: ['tests/**/*.test.ts'],
    exclude: [...configDefaults.exclude, 'tests/site-output.test.ts', ...catalog.suites.map((row: { file: string }) => row.file)],
    tagsFilter: ['!' + sourceArchiveTag]
  }
}));
