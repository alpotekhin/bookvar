import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

export function beforeServingDecisions(text, source, root = defaultRoot) {
  const row = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-serving-decisions-additions.json'), 'utf8'));
  if (source !== row.source || hash(text) === row.before_sha256) return text;
  for (const block of row.blocks) {
    const pattern = new RegExp(`<!-- serving-decisions-completion: ${block.label} start -->[\\s\\S]*?<!-- serving-decisions-completion: ${block.label} end -->\\n\\n`, 'g');
    const matches = [...text.matchAll(pattern)];
    if (matches.length !== 1 || hash(matches[0][0]) !== block.sha256)
      throw new Error(`Unreviewed serving decision addition: ${source}/${block.label}`);
    text = text.replace(matches[0][0], '');
  }
  return text;
}
