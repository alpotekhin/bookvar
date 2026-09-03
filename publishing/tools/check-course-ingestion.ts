import { resolve } from 'node:path';
import { validateKnownCourseLedgers } from './course-ledger.ts';

const repositoryRoot = resolve(import.meta.dirname, '../..');
const validated = validateKnownCourseLedgers(repositoryRoot);
console.log(
  validated.length === 0
    ? 'No Stanford or Berkeley course ledger exists yet; nothing to validate.'
    : `Validated course ledgers: ${validated.join(', ')}`
);
