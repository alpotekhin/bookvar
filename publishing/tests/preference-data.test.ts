import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('preference data chapter examples', () => {
  it('executes every published Python fence and the negative data-conversion cases', () => {
    const root = resolve(import.meta.dirname, '../..');
    const output = execFileSync('python3', ['publishing/tools/check_preference_data_examples.py'], {
      cwd: root, encoding: 'utf8', timeout: 15000
    });
    const report = JSON.parse(output.slice(0, output.indexOf('\nPASS ')));
    expect(report.published_python_fences).toBe(5);
    expect(report.ranking_pairs).toBe(6);
    expect(report.tied_ranking_strict_pairs).toBe(5);
    expect(report.train_pairs).toBe(12);
    expect(report.validation_pairs).toBe(6);
    expect(report.independent_cycle_probability).toBeCloseTo(0.09453439083199781, 12);
    expect(report.pair_weighted_loss).toBeCloseTo(0.6941176470588236, 12);
    expect(report.group_weighted_loss).toBe(0.5);
    expect(report.majority_cycle_votes).toEqual([2, 2, 2]);
    expect(report.example_stdout).toEqual(['0.094534']);
    expect(report.chapter_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(output).toContain('PASS actual preference-data examples');
  });
});
