import { readFileSync } from 'node:fs';
import { configDefaults, defineConfig } from 'vitest/config';

const catalog = JSON.parse(readFileSync(new URL('./tests/source-evidence-suites.json', import.meta.url), 'utf8'));
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    exclude: [...configDefaults.exclude, 'tests/site-output.test.ts', ...catalog.suites.map((row: { file: string }) => row.file)]
  }
});
