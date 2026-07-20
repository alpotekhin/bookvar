import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { findBrokenBuiltLinks } from '../adapter/check-built-links.js';

function write(path: string, contents: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, contents);
}

function fixture(html: string): string {
  const dist = mkdtempSync(join(tmpdir(), 'built-links-'));
  write(join(dist, 'index.html'), html);
  write(join(dist, 'target', 'index.html'), '<h2 id="existing">Target</h2>');
  write(join(dist, 'assets', 'present.svg'), '<svg/>');
  return dist;
}

describe('findBrokenBuiltLinks', () => {
  it('detects a missing route', async () => {
    await expect(findBrokenBuiltLinks(fixture('<a href="/missing/">missing</a>')))
      .resolves.toContain('/ -> missing route /missing/');
  });

  it('detects a missing fragment', async () => {
    await expect(findBrokenBuiltLinks(fixture('<a href="/target/#absent">missing</a>')))
      .resolves.toContain('/ -> missing fragment /target/#absent');
  });

  it('detects missing assets and other extension-looking local files', async () => {
    const errors = await findBrokenBuiltLinks(fixture([
      '<img src="/assets/present.svg">',
      '<img src="/assets/missing.svg">',
      '<a href="/missing.pdf">document</a>'
    ].join('\n')));
    expect(errors).toContain('/ -> missing file /assets/missing.svg');
    expect(errors).toContain('/ -> missing file /missing.pdf');
    expect(errors).not.toContain('/ -> missing file /assets/present.svg');
  });

  it('checks links below the configured GitHub Pages base path', async () => {
    await expect(findBrokenBuiltLinks(fixture([
      '<a href="/bookvar/target/#existing">target</a>',
      '<img src="/bookvar/assets/present.svg">'
    ].join('\n')), '/bookvar')).resolves.toEqual([]);
  });
});
