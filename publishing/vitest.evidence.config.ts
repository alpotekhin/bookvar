import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

const catalog = JSON.parse(readFileSync(new URL('./tests/source-evidence-suites.json', import.meta.url), 'utf8'));
export default defineConfig({
  test: { include: catalog.suites.map((row: { file: string }) => row.file) }
});
