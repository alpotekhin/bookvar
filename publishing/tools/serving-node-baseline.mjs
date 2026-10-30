import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

export function beforeServingNode(text, source, root = defaultRoot) {
  const row = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-serving-node-additions.json'), 'utf8'));
  if (source !== row.source || hash(text) === row.before_sha256) return text;
  const pattern = /<!-- serving-node-completion: host-loading start -->[\s\S]*?<!-- serving-node-completion: host-loading end -->\n\n/g;
  const blocks = [...text.matchAll(pattern)];
  if (blocks.length !== 1 || hash(blocks[0][0]) !== row.block_sha256)
    throw new Error(`Unreviewed serving-node addition: ${source}`);
  return text.replace(blocks[0][0], '');
}
