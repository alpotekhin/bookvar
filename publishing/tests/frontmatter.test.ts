import { describe, expect, it } from 'vitest';
import { readPage } from '../adapter/frontmatter.js';

describe('readPage', () => {
  it.each(['en', 'ru', 'mixed'] as const)('reads the explicitly declared source language %s', (language) => {
    const raw = `---\ntitle: Chapter\ntype: textbook-chapter\nstatus: canonical\nsource_language: ${language}\n---\nOriginal body`;
    const page = readPage('chapter.md', raw);
    expect(page.sourceLanguage).toBe(language);
    expect(page.body).toBe('Original body');
  });
  it('rejects unsupported source-language declarations instead of silently assuming English', () => {
    expect(() => readPage('chapter.md', '---\ntitle: Chapter\ntype: textbook-chapter\nstatus: canonical\nsource_language: original\n---\nBody'))
      .toThrow('source_language must be en, ru or mixed');
  });
  it('validates the declared English review status and revision', () => {
    const base = '---\ntitle: EN\ntype: textbook-chapter\nstatus: canonical\n';
    const tail = "\ntranslation_reviewed: '2026-10-29'\n---\nBody";
    expect(readPage('en.md', base + 'translation_status: partial\ncanonical_revision: ' +
      'a'.repeat(64) + tail).translationStatus).toBe('partial');
    expect(() => readPage('en.md', base + 'translation_status: done\ncanonical_revision: wrong' + tail))
      .toThrow('translation_status must be current or partial');
    expect(() => readPage('en.md', base + 'translation_status: current\ncanonical_revision: wrong' + tail))
      .toThrow('canonical_revision must be a SHA-256');
  });
  it('rejects page types outside the AGENTS.md publication contract', () => {
    expect(() => readPage('legacy.md', [
      '---',
      'title: Legacy',
      'type: course-note',
      'status: legacy',
      '---',
      'Body'
    ].join('\n'))).toThrow('unsupported type "course-note"');
  });
  it('reads textbook metadata and preserves the Markdown body', () => {
    const raw = [
      '---',
      'title: Внимание',
      'type: textbook-chapter',
      'status: stable',
      'last_updated: 2026-07-17',
      '---',
      '# Внимание',
      '',
      'Текст главы.'
    ].join('\n');

    expect(readPage('00 Учебник/Внимание.md', raw)).toEqual({
      title: 'Внимание',
      type: 'textbook-chapter',
      status: 'stable',
      lastUpdated: '2026-07-17',
      sourceFragments: [],
      body: '# Внимание\n\nТекст главы.'
    });
    expect(raw).toContain('last_updated: 2026-07-17');
  });

  it.each([
    ['2026-10-26', '2026-10-23'],
    ['2026-10-23', '2026-10-26']
  ])('uses explicit last_updated=%s independently of last_verified=%s', (updated, verified) => {
    const raw = `---\ntitle: Page\ntype: textbook-chapter\nstatus: canonical\nlast_updated: ${updated}\nlast_verified: ${verified}\n---\nBody`;
    expect(readPage('chapter.md', raw).lastUpdated).toBe(updated);
    expect(raw).toContain(`last_verified: ${verified}`);
  });

  it('falls back to last_verified only when last_updated is absent', () => {
    const raw = '---\ntitle: Page\ntype: source-note\nstatus: active\nlast_verified: 2026-10-23\n---\nBody';
    expect(readPage('source.md', raw).lastUpdated).toBe('2026-10-23');
  });

  it.each(['model-family', 'model-release', 'research-line'])(
    'requires last_verified for %s pages',
    (type) => {
      const raw = `---\ntitle: Page\ntype: ${type}\nstatus: current\n---\nBody`;

      expect(() => readPage('02 Атлас моделей/Page.md', raw)).toThrowError(
        /Publication metadata errors:[\s\S]*02 Атлас моделей\/Page\.md[\s\S]*last_verified/
      );
    }
  );

  it('reports all missing required fields with the source path', () => {
    expect(() => readPage('broken/Page.md', '---\n---\nBody')).toThrowError(
      /Publication metadata errors:[\s\S]*broken\/Page\.md[\s\S]*title[\s\S]*type[\s\S]*status/
    );
  });
});
