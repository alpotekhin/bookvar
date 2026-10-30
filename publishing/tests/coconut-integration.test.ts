import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (path: string) => createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex');
const chapter = '00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md';

describe('Coconut integrated in the existing reasoning explanation', () => {
  it('runs every current TTC Python block and independent feedback/label controls', () => {
    const output = execFileSync('uv', ['run', '--offline', '--with', 'sympy==1.14.0', 'python',
      'publishing/tools/check_test_time_compute_examples.py'], { cwd: root, encoding: 'utf8', timeout: 30000 });
    const report = JSON.parse(output.slice(0, output.indexOf('\nPASS ')));
    expect(report.python_fences).toBe(7);
    expect(report.coconut_feedback_gradient_cases).toBe(41);
    expect(report.coconut_curriculum_cases).toBe(16);
    expect(report.coconut_rejected_inputs).toBe(4);
    expect(report.chapter_sha256).toBe(hash(chapter));
  });
  it('preserves the complete preceding chapter outside the recorded addition', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-coconut-additions.json')).files[0];
    let restored = beforeTopicConsolidation(read(chapter), chapter, root,
      '2026-10-29-coconut-additions.json');
    expect(createHash('sha256').update(restored).digest('hex')).toBe(journal.after_sha256);
    for (const change of [...journal.changes].reverse()) {
      expect(restored.split(change.new)).toHaveLength(2);
      restored = restored.replace(change.new, () => change.old);
    }
    expect(restored).toBe(read('docs/audit/snapshots/2026-10-29-before-final-closure/test-time-compute.md'));
    const text = read(chapter);
    for (const boundary of ['not a Transformer', 'every epoch', 'no explicit BFS']) {
      expect(text).toContain(boundary);
    }
    expect(text).toContain('id="coconut-continuous-thought"');
    for (const index of ['01 Справочник/_index.md', 'en/01 Reference/_index.md'])
      expect(read(index)).toContain(chapter.replace(/\.md$/, '') + '#coconut-continuous-thought');
  });
  it('keeps the complete source image and the original MIT notice byte-identical', () => {
    const spec = JSON.parse(read('00 Учебник/Assets/Figures/curated/coconut-2026-10-29/assets.json'))[0];
    expect(spec.commit).toBe('27273cb8cca4bb763c041a63b036d0c3b7cbbb48');
    expect(hash(spec.asset)).toBe(spec.sha256);
    expect(hash(spec.source_asset)).toBe(spec.sha256);
    expect(hash(spec.license_notice)).toBe(hash(spec.license_evidence));
    expect(spec.dimensions_px).toEqual([2896, 796]);
    expect(spec.used_in).toEqual([chapter]);
  });
});
