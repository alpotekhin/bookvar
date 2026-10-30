import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
export function beforeConfidentLearningAddition(text, source, root = defaultRoot) {
  const record = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-confident-learning-addition.json'), 'utf8'));
  if (record.source !== source) return text;
  const pattern = /<!-- confident-learning-addition: start -->[\s\S]*?<!-- confident-learning-addition: end -->\n\n/g;
  const blocks = [...text.matchAll(pattern)];
  if (blocks.length === 0) {
    if (createHash('sha256').update(text).digest('hex') === record.before_sha256) return text;
    throw new Error('Missing reviewed Confident Learning addition');
  }
  if (blocks.length !== 1 || createHash('sha256').update(blocks[0][0]).digest('hex') !== record.block_sha256) {
    throw new Error('Unreviewed change inside Confident Learning addition');
  }
  const dates = 'last_updated: 2026-10-29\nlast_verified: 2026-10-29';
  if (!text.includes(dates)) throw new Error('Unrecognized Confident Learning dates');
  return text.replace(blocks[0][0], '').replace(dates,
    'last_updated: 2026-10-28\nlast_verified: 2026-10-28');
}
