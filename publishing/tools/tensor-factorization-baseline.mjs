import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

// Preserve historical review hashes; restore only this separately reviewed block.
export function beforeTensorFactorization(text, source, root = defaultRoot) {
  const row = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-tensor-factorization-addition.json'), 'utf8'));
  if (source !== row.source) return text;
  if (hash(text) === row.before_sha256) return text;
  const blocks = [...text.matchAll(/<!-- tensor-factorization-addition: start -->[\s\S]*?<!-- tensor-factorization-addition: end -->\n\n/g)];
  if (blocks.length !== 1 || hash(blocks[0][0]) !== row.block_sha256)
    throw new Error(`Unreviewed tensor-factorization addition: ${source}`);
  text = text.replace(blocks[0][0], '');
  for (const change of row.frontmatter_changes) {
    if (!text.includes(change.new)) throw new Error(`Unrecognized tensor date: ${source}`);
    text = text.replace(change.new, change.old);
  }
  return text;
}
