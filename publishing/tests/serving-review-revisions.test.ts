import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');

describe('serving review revision boundaries', () => {
  it('restores exact historical hashes once while rejecting unrecorded current edits', () => {
    const script = `
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'publishing/tools')
from topic_consolidation_baseline import before_topic_consolidation
from serving_decisions_baseline import before_serving_decisions
from serving_node_baseline import before_serving_node
from serving_operations_baseline import before_serving_operations
sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
checked = 0
for name, restore in [('decisions', before_serving_decisions), ('node', before_serving_node)]:
    row = json.loads(Path(f'docs/audit/2026-10-29-serving-{name}-additions.json').read_text())
    raw = Path(row['source']).read_text()
    reviewed = before_topic_consolidation(raw, row['source'])
    assert sha(reviewed) == row['after_sha256']
    assert sha(restore(reviewed, row['source'])) == row['before_sha256']
    checked += 1
for row in json.loads(Path('docs/audit/2026-10-29-serving-operations-additions.json').read_text())['chapters']:
    raw = Path(row['source']).read_text()
    reviewed = before_serving_decisions(before_topic_consolidation(raw, row['source']), row['source'])
    assert sha(reviewed) == row['after_sha256']
    assert sha(before_serving_operations(raw, row['source'])) == row['before_sha256']
    checked += 1
source = '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
try:
    before_topic_consolidation(Path(source).read_text() + '\\nUNREVIEWED\\n', source)
except AssertionError:
    pass
else:
    raise AssertionError('Unrecorded current edit was accepted')
print(json.dumps({'historical_contracts': checked, 'unknown_edit_rejected': True}))
`;
    const result = JSON.parse(execFileSync('python3', ['-c', script], { cwd: root, encoding: 'utf8' }));
    expect(result).toEqual({ historical_contracts: 5, unknown_edit_rejected: true });
  });
});
