import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadLinkAllowlist } from '../adapter/link-policy.js';

function policy(value: unknown): string {
  const path = join(mkdtempSync(join(tmpdir(), 'link-policy-')), 'policy.json');
  writeFileSync(path, JSON.stringify(value));
  return path;
}

describe('loadLinkAllowlist', () => {
  it('loads exact targets and concise reasons', () => {
    expect(loadLinkAllowlist(policy({
      entries: [{ target: 'Missing/Page', reason: 'not authored yet' }]
    }))).toEqual(new Map([['Missing/Page', 'not authored yet']]));
  });

  it('rejects duplicate targets and empty reasons', () => {
    expect(() => loadLinkAllowlist(policy({ entries: [
      { target: 'Same', reason: 'first' },
      { target: 'Same', reason: 'second' }
    ] }))).toThrow('Duplicate link allowlist target: Same');
    expect(() => loadLinkAllowlist(policy({
      entries: [{ target: 'Missing/Page', reason: '' }]
    }))).toThrow('has an empty reason');
  });
});
