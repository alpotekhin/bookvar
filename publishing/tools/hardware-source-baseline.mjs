import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const sha = text => createHash('sha256').update(text).digest('hex');

/** Remove only the reviewed hardware addition before older baseline helpers. */
export function beforeHardwareSourceCompletion(text, source, root = defaultRoot) {
  const journal = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-hardware-source-comparison.json'), 'utf8'));
  const row = journal.addition;
  if (source !== row.source || sha(text) === row.before_sha256) return text;
  const found = [...text.matchAll(/<!-- hardware-source-completion: mapping:start -->[\s\S]*?<!-- hardware-source-completion: mapping:end -->\n\n/g)];
  if (found.length !== 1 || sha(found[0][0]) !== row.block_sha256)
    throw Error(`Unreviewed hardware source completion: ${source}`);
  return text.replace(found[0][0], '');
}
