import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

// Restore only the two separately reviewed additions. Do not normalize prose.
export function beforeSelectionGapsAddition(text, source, root = defaultRoot) {
  const records = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-selection-gaps-additions.json'), 'utf8')).files;
  const record = records.find(row => row.source === source);
  if (!record) return text;
  const pattern = new RegExp(`<!-- ${record.marker}-addition: start -->[\\s\\S]*?<!-- ${record.marker}-addition: end -->\\n\\n`, 'g');
  const blocks = [...text.matchAll(pattern)];
  if (blocks.length === 0 && hash(text) === record.before_sha256) return text;
  if (blocks.length !== 1 || hash(blocks[0][0]) !== record.block_sha256)
    throw new Error(`Unreviewed selection-gap addition: ${source}`);
  text = text.replace(blocks[0][0], '');
  for (const change of record.frontmatter_changes) {
    if (!text.includes(change.new)) throw new Error(`Unrecognized addition date: ${source}`);
    text = text.replace(change.new, change.old);
  }
  return text;
}
