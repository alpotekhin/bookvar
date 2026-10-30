import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');

// Restore only the exact display-math replacements recorded in the layout
// audit. No general whitespace normalization: an unrecorded edit must fail.
export function beforeMathReflow(text, source, root = defaultRoot) {
  const report = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-math-reflow.json'), 'utf8'));
  const record = report.files.find(file => file.source === source);
  if (!record) return text;
  for (const change of [...record.changes].reverse()) {
    const after = `$$\n${change.new}\n$$`;
    const before = `$$\n${change.old}\n$$`;
    if (text.includes(after)) text = text.replace(after, () => before);
    else if (!text.includes(before)) {
      throw new Error(`Unrecognized math revision: ${source}: ${change.old}`);
    }
  }
  return text;
}
