import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const source = '02 Атлас моделей/Сравнения/Карта архитектурных линий.md';
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string) => createHash('sha256').update(text).digest('hex');

describe('architecture-map GQA example', () => {
  it('keeps Falcon 2 sizes and Falcon 3 per-checkpoint training budgets separate', () => {
    const text = read('02 Атлас моделей/Семейства/Falcon.md');
    expect(text).toContain('| Falcon 2 (2024) | [Falcon2-11B]');
    expect(text).toContain('| Falcon 3 (2024) | Семейство 1B–10B');
    expect(text).toContain('7B обучена на 14T токенов');
    expect(text).toContain('продолжением обучения на 2T токенов');
    expect(text).not.toContain('| Falcon 2 / Falcon 3 (2024) |');
    const journal = JSON.parse(read('docs/audit/2026-10-30-atlas-release-label-additions.json'));
    const row = journal.files[0];
    expect(sha(text)).toBe(row.after_sha256);
    const restored = beforeTopicConsolidation(text, row.source, root,
      '2026-10-30-atlas-gqa-label-additions.json');
    expect(sha(restored)).toBe(row.before_sha256);
    expect(text.match(/^!\[\[.*\]\]$/gm)).toEqual(restored.match(/^!\[\[.*\]\]$/gm));
    expect(() => beforeTopicConsolidation(text + '\nUnreviewed change\n', row.source, root,
      '2026-10-30-atlas-gqa-label-additions.json')).toThrow();
  });

  it('names a released GQA variant rather than generalizing across Llama 2 sizes', () => {
    const text = read(source);
    expect(text).toContain('Llama 2 70B / Llama 3: GQA');
    expect(text).not.toContain('Llama 2/3: GQA');
    expect(read('02 Атлас моделей/Семейства/Llama.md')).toContain('GQA только у 34B/70B');
  });

  it('preserves all other comparison content and rejects an unrecorded change', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-30-atlas-gqa-label-additions.json'));
    const row = journal.files[0];
    const current = read(source);
    expect(sha(current)).toBe(row.after_sha256);
    const restored = beforeTopicConsolidation(current, source, root,
      '2026-10-30-en-scheduler-navigation-additions.json');
    expect(sha(restored)).toBe(row.before_sha256);
    expect(current.match(/^!\[\[.*\]\]$/gm)).toEqual(restored.match(/^!\[\[.*\]\]$/gm));
    expect(() => beforeTopicConsolidation(current + '\nUnreviewed change\n', source, root,
      '2026-10-30-en-scheduler-navigation-additions.json')).toThrow();
  });
});
