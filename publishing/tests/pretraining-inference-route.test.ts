import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const canonical = '00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена.md';
const english = 'en/00 Textbook/14 Inference and optimization/54 Decoding and next-token selection.md';
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const matter = (text: string) => parse(text.split('---')[1]);

describe('pretraining to decoding prerequisites', () => {
  it('requires the next-token objective rather than the optional synthetic-data module', () => {
    for (const source of [canonical, english]) {
      const prerequisites = matter(read(source)).prerequisites;
      expect(prerequisites).toHaveLength(1);
      expect(prerequisites[0]).toContain('42 Next-token prediction');
      expect(prerequisites[0]).not.toContain('53 Синтетические данные');
    }
  });

  it('pins the same canonical revision without creating or replacing a translation', () => {
    const data = matter(read(english));
    expect(data.canonical_revision).toBe(hash(read(canonical)));
    expect(data.translation_status).toBe('current');
    expect(data.translation_reviewed).toBe('2026-10-30');
  });

  it('changes only frontmatter in both full chapters and rejects unknown edits', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-30-decoding-prerequisites-additions.json'));
    expect(journal.files.map((r: { source: string }) => r.source)).toEqual([canonical, english]);
    for (const row of journal.files) {
      const current = read(row.source);
      const before = beforeTopicConsolidation(current, row.source, root,
        '2026-10-30-reader-provenance-additions.json');
      expect(hash(current), row.source).toBe(row.after_sha256);
      expect(hash(before), row.source).toBe(row.before_sha256);
      expect(current.split('\n---\n')[1]).toBe(before.split('\n---\n')[1]);
      expect(() => beforeTopicConsolidation(current + '\nUnreviewed paragraph\n', row.source, root,
        '2026-10-30-reader-provenance-additions.json')).toThrow('Unreviewed topic consolidation revision');
    }
  });
});
