import { defineConfig } from 'vitest/config';

export const sourceArchiveTag = 'source-archive';

export default defineConfig({
  test: {
    tags: [{ name: sourceArchiveTag, description: 'Compares with a retained local source snapshot' }]
  }
});
