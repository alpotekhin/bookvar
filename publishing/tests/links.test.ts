import { describe, expect, it, vi } from 'vitest';
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
    const originalSource = source;

    expect(convertWikiSyntax(source, registry)).toEqual({
      markdown: [
        '[Page](/resolved-route/)',
        '[читаемый текст](/resolved-route/)',
        '[Page](/resolved-route/#раздел)',
        '![attention](../../assets/Figures/attention.svg)'
      ].join('\n'),
      unresolved: [],
      allowlisted: []
    });
    expect(source).toBe(originalSource);
  });

  it('lowercases non-ASCII headings without locale-sensitive conversion', () => {
    const localeLowercase = vi
      .spyOn(String.prototype, 'toLocaleLowerCase')
      .mockImplementation(() => {
        throw new Error('locale-sensitive conversion used');
      });

    try {
      expect(convertWikiSyntax('[[Path/Page#ЁЖ]]', registry).markdown)
        .toBe('[Page](/resolved-route/#ёж)');
    } finally {
      localeLowercase.mockRestore();
    }
  });

  it('keeps an unresolved embed visible as plain text and reports its target', () => {
    expect(convertWikiSyntax('![[Missing Page]]', registry)).toEqual({
      markdown: 'Missing Page',
      unresolved: ['Missing Page'],
      allowlisted: []
    });
  });

  it('does not treat a non-allowlisted extension as an image', () => {
    expect(convertWikiSyntax('![[Assets/Figures/attention.svgx]]', registry)).toEqual({
      markdown: 'attention.svgx',
      unresolved: ['Assets/Figures/attention.svgx'],
      allowlisted: []
    });
  });

  it('keeps explicitly allowlisted targets visible and records their reason', () => {
    const allowlist = new Map([['Missing Page', 'not available for publication']]);
    expect(convertWikiSyntax('[[Missing Page|visible label]]', registry, allowlist)).toEqual({
      markdown: 'visible label',
      unresolved: [],
      allowlisted: [{ target: 'Missing Page', reason: 'not available for publication' }]
    });
  });

  it('converts local heading targets without treating them as missing pages', () => {
    expect(convertWikiSyntax('[[#Local Heading|jump]]', registry)).toEqual({
      markdown: '[jump](#local-heading)',
      unresolved: [],
      allowlisted: []
    });
  });
});
