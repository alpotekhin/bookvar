import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {beforeServingDecisions} from './serving-decisions-baseline.mjs';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

// Restore only these independently reviewed additions for historical audits.
export function beforeServingOperations(text, source, root = defaultRoot) {
  text = beforeServingDecisions(text, source, root);
  const rows = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-serving-operations-additions.json'), 'utf8')).chapters;
  const row = rows.find(item => item.source === source);
  if (!row || hash(text) === row.before_sha256) return text;
  const pattern = new RegExp(`<!-- serving-operations-completion: ${row.label} start -->[\\s\\S]*?<!-- serving-operations-completion: ${row.label} end -->\\n\\n`, 'g');
  const blocks = [...text.matchAll(pattern)];
  if (blocks.length !== 1 || hash(blocks[0][0]) !== row.block_sha256)
    throw new Error(`Unreviewed serving operations addition: ${source}`);
  return text.replace(blocks[0][0], '');
}
