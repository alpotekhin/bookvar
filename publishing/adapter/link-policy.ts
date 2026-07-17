import { readFileSync } from 'node:fs';

export interface LinkPolicyEntry {
  target: string;
  reason: string;
}

export function loadLinkAllowlist(path: string): Map<string, string> {
  const raw: unknown = JSON.parse(readFileSync(path, 'utf8'));
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error('Link allowlist must be an object');
  }
  const keys = Object.keys(raw);
  if (keys.length !== 1 || keys[0] !== 'entries') {
    throw new Error('Link allowlist schema requires only "entries"');
  }
  const entries = (raw as { entries?: unknown }).entries;
  if (!Array.isArray(entries)) throw new Error('Link allowlist entries must be an array');

  const result = new Map<string, string>();
  entries.forEach((value, index) => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new Error(`Link allowlist entry ${index} must be an object`);
    }
    const entry = value as Record<string, unknown>;
    if (Object.keys(entry).sort().join(',') !== 'reason,target') {
      throw new Error(`Link allowlist entry ${index} requires only target and reason`);
    }
    if (typeof entry.target !== 'string' || entry.target.trim() === '') {
      throw new Error(`Link allowlist entry ${index} has an empty target`);
    }
    if (typeof entry.reason !== 'string' || entry.reason.trim() === '') {
      throw new Error(`Link allowlist entry ${index} has an empty reason`);
    }
    if (result.has(entry.target)) throw new Error(`Duplicate link allowlist target: ${entry.target}`);
    result.set(entry.target, entry.reason);
  });
  return result;
}
