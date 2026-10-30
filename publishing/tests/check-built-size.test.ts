import { mkdtempSync, mkdirSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import { assertPublicationSize, measurePublicationSize, PAGES_MAX_BYTES } from '../adapter/check-built-size.js';

describe('Pages artifact size gate', () => {
  it('counts file bytes recursively, not compressed size or filesystem allocation', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'bookvar-size-'));
    mkdirSync(join(dir, 'nested'));
    writeFileSync(join(dir, 'index.html'), 'abc');
    writeFileSync(join(dir, 'nested', 'file.pdf'), Buffer.alloc(1024));
    expect(await measurePublicationSize(dir)).toEqual({ bytes: 1027, files: 2 });
  });
  it('rejects oversized and invalid artifacts at the exact boundary', () => {
    for (const bytes of [0, PAGES_MAX_BYTES]) expect(() => assertPublicationSize(bytes)).not.toThrow();
    for (const bytes of [PAGES_MAX_BYTES + 1, -1, NaN, 1.5]) expect(() => assertPublicationSize(bytes)).toThrow();
  });
  it('does not let symlinks evade the measured upload contents', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'bookvar-size-link-'));
    writeFileSync(join(dir, 'target'), 'abc');
    symlinkSync('target', join(dir, 'link'));
    await expect(measurePublicationSize(dir)).rejects.toThrow('symlink');
  });
});
