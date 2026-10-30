import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('explicit editorial disposition of existing course residuals', () => {
  it('accounts for retained course material without calling it newly integrated knowledge', () => {
    const root = resolve(import.meta.dirname, '../..');
    const report = JSON.parse(execFileSync(process.execPath, ['publishing/tools/check-course-residuals.mjs'], {
      cwd: root, encoding: 'utf8', timeout: 15000,
    }));
    expect(report.records).toBe(206);
    expect(report.counts.sourceDisposition).toEqual({ 'contract-only': 21, 'source-only': 185 });
    expect(report.counts.decision['gap-filled-with-component-map']).toBe(1);
    expect(report.counts.decision['retain-recording']).toBe(12);
    expect(report.counts.unpublishedNotes).toBe(31);
  });
});
