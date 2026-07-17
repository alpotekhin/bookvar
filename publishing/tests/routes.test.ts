import { describe, expect, it } from 'vitest';
import { createRouteRegistry } from '../adapter/routes.js';

describe('route registry', () => {
  it('resolves a full Obsidian path and an alias', () => {
    const registry = createRouteRegistry([{
      sourcePath: '00 Учебник/05 Attention/01 Attention.md',
      route: '/textbook/attention/',
      title: 'Attention'
    }]);
    expect(registry.routeForWikiTarget('00 Учебник/05 Attention/01 Attention')).toBe('/textbook/attention/');
    expect(registry.routeForWikiTarget('01 Attention')).toBe('/textbook/attention/');
    expect(registry.routeForWikiTarget(
      '02 Areas/ML & DL/00 Учебник/05 Attention/01 Attention'
    )).toBe('/textbook/attention/');
  });

  it('throws on an ambiguous basename', () => {
    expect(() => createRouteRegistry([
      { sourcePath: 'a/Index.md', route: '/a/', title: 'A' },
      { sourcePath: 'b/Index.md', route: '/b/', title: 'B' }
    ])).toThrow(/ambiguous/i);
  });

  it('can preserve full-path routes while omitting ambiguous basename aliases', () => {
    const registry = createRouteRegistry([
      { sourcePath: 'a/Index.md', route: '/a/', title: 'A' },
      { sourcePath: 'b/Index.md', route: '/b/', title: 'B' }
    ], { allowAmbiguousBasenames: true });

    expect(registry.routeForWikiTarget('a/Index')).toBe('/a/');
    expect(registry.routeForWikiTarget('b/Index')).toBe('/b/');
    expect(registry.routeForWikiTarget('Index')).toBeUndefined();
  });
});
