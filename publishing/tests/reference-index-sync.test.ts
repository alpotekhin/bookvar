import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import matter from 'gray-matter';
import { describe, expect, it } from 'vitest';
import { parseMarkdownHeadings } from '../adapter/build.js';
import { referenceTerms } from '../adapter/reference-terms.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const journalName = '2026-10-30-reference-index-metadata-additions.json';

describe('reference index follows the chapter that owns each explanation', () => {
  it('preserves all preceding content with a reversible six-file metadata journal', () => {
    const journal = JSON.parse(read('docs/audit/' + journalName));
    expect(journal.files).toHaveLength(6);
    for (const row of journal.files) {
      const current = beforeTopicConsolidation(read(row.source), row.source, root, journalName);
      expect(hash(current)).toBe(row.after_sha256);
      const prior = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-30-bleu-reconciliation-additions.json');
      expect(hash(prior)).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(current + '\nunknown edit\n', row.source, root,
        '2026-10-30-bleu-reconciliation-additions.json')).toThrow();
    }
  });

  it('keeps all six formerly manual terms in chapter metadata with unique anchors', () => {
    const expected: Array<[string, string[]]> = [
      ['00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md',
        ['dropconnect', 'stochastic-depth']],
      ['00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md',
        ['gradient-norm-clipping', 'agc-unitwise', 'autoclip']],
      ['00 Учебник/05 Attention и Transformer/03 Полный Transformer.md',
        ['transformer-dropout']],
    ];
    for (const [source, anchors] of expected) {
      const page = matter(read(source));
      const terms = referenceTerms(source, page.data.reference_terms,
        parseMarkdownHeadings(page.content, true));
      expect(terms.map(term => term.anchor)).toEqual(anchors);
      for (const [index, english] of [
        ['01 Справочник/_index.md', false], ['en/01 Reference/_index.md', true],
      ] as const) {
        const lines = read(index).split('\n');
        const owner = lines.findIndex(line => line.startsWith('- [[' + source.slice(0, -3) + '|'));
        expect(owner).toBeGreaterThan(-1);
        for (const [offset, term] of terms.entries()) {
          expect(lines[owner + offset + 1]).toBe('  - [[' + source.slice(0, -3) + '#'
            + term.anchor + '|' + (english ? term.label_en : term.label) + ']]');
        }
      }
    }
  });

  it('keeps the reviewed English Transformer body while advancing only its metadata pin', () => {
    const canonicalSource = '00 Учебник/05 Attention и Transformer/03 Полный Transformer.md';
    const englishSource = 'en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md';
    const journal = JSON.parse(read('docs/audit/' + journalName));
    const canonicalRow = journal.files.find((row: { source: string }) => row.source === canonicalSource);
    const englishRow = journal.files.find((row: { source: string }) => row.source === englishSource);
    for (const source of [canonicalSource, englishSource]) {
      const current = beforeTopicConsolidation(read(source), source, root, journalName);
      const previous = beforeTopicConsolidation(read(source), source, root,
        '2026-10-30-bleu-reconciliation-additions.json');
      expect(matter(current).content).toBe(matter(previous).content);
    }
    const english = matter(read(englishSource)).data;
    expect(english.translation_status).toBe('current');
    expect(english.canonical_revision).toBe(canonicalRow.after_sha256);
    expect(english.translation_reviewed).toBe('2026-10-29');
    expect(englishRow.changes.map((change: { old: string }) => change.old.split(':')[0]))
      .toEqual(['canonical_revision', 'last_updated']);
  });

  it('reproduces both saved indices from current chapter metadata without writing', () => {
    const output = execFileSync(resolve(root, 'publishing/node_modules/.bin/tsx'),
      [resolve(root, 'publishing/tools/sync-topic-index.ts'), '--check'],
      { cwd: root, encoding: 'utf8', timeout: 20000 });
    expect(JSON.parse(output)).toEqual({ mode: 'check', topics: 152,
      indexes: ['01 Справочник/_index.md', 'en/01 Reference/_index.md'] });
  }, 25000);
});
