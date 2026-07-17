import { describe, expect, it } from 'vitest';
import type { RouteRegistry } from '../adapter/types.js';
import { convertWikiSyntax } from '../adapter/links.js';

const registry: RouteRegistry = {
  routeForWikiTarget(target) {
    return target === 'Path/Page' ? '/resolved-route/' : undefined;
  }
};

describe('convertWikiSyntax', () => {
  it('converts links, aliases, headings, and supported image embeds', () => {
    const source = [
      '[[Path/Page]]',
      '[[Path/Page|читаемый текст]]',
      '[[Path/Page#Раздел]]',
      '![[Assets/Figures/attention.svg]]'
    ].join('\n');

    expect(convertWikiSyntax(source, registry)).toEqual({
      markdown: [
        '[Page](/resolved-route/)',
        '[читаемый текст](/resolved-route/)',
        '[Page](/resolved-route/#раздел)',
        '![attention](../../assets/Figures/attention.svg)'
      ].join('\n'),
      unresolved: []
    });
    expect(source).toContain('[[Path/Page]]');
  });

  it('keeps an unresolved embed visible as plain text and reports its target', () => {
    expect(convertWikiSyntax('![[Missing Page]]', registry)).toEqual({
      markdown: 'Missing Page',
      unresolved: ['Missing Page']
    });
  });
});
