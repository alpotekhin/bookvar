import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');

describe('learner edits versus immutable course archives', () => {
  for (const assignment of ['a1', 'a2']) {
    it(`${assignment} permits implementing adapters without weakening tests`, () => {
      const contract = parse(readFileSync(
        resolve(root, `06 Практика/Contracts/stanford-cs336-${assignment}.yml`), 'utf8'
      ));
      expect(contract.learner_changes.adapter_bodies).toBe('required');
      expect(contract.learner_changes.adapter_signatures_and_semantics).toBe('preserved');
      expect(contract.learner_changes.tests_and_fixtures).toBe('immutable');
      expect(contract.learner_changes.source_hash_scope).toBe('unmodified archival snapshot, not learner adapter bodies');
      expect(contract.fail_if.join('\n')).not.toMatch(/tests(?:,)?(?: or)? adapters(?: or fixtures)? are modified/);
    });
  }
});
