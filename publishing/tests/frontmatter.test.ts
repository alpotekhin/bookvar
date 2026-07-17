import { describe, expect, it } from 'vitest';
import { readPage } from '../adapter/frontmatter.js';

describe('readPage', () => {
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
      body: '# Внимание\n\nТекст главы.'
    });
    expect(raw).toContain('last_updated: 2026-07-17');
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
