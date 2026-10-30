"""Execute the three actual CL reader blocks, then check independent invariants.

This checks arithmetic/bookkeeping and synthetic CPU fits, not Cleanlab package
execution, a real label audit, a theorem proof or paper experiment reproduction.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import sklearn

root = Path(__file__).resolve().parents[2]
journal = json.loads((root / 'docs/audit/2026-10-29-confident-learning-addition.json').read_text())
text = (root / journal['source']).read_text()
assert hashlib.sha256(text.encode()).hexdigest() == journal['after_sha256']
addition = re.search(r'<!-- confident-learning-addition: start -->[\s\S]*?<!-- confident-learning-addition: end -->\n\n', text)[0]
blocks = re.findall(r'^```python\n(.*?)^```\s*$', addition, re.M | re.S)
assert len(blocks) == 3
scopes = []
for i, block in enumerate(blocks):
    scope = {}
    exec(compile(block, f'reader-cl-block-{i+1}', 'exec'), scope)
    scopes.append(scope)

s = scopes[1]
p, y, t = s['p'], s['labels'], s['t']
manual_bins = [[1, 2], [3], [], [7], [5, 6], [8], [], [11], [9, 10]]
oracle = np.array([len(ids) for ids in manual_bins]).reshape(3, 3)
np.testing.assert_array_equal(s['C'], oracle)
np.testing.assert_array_equal(s['estimated_counts'],
    [[8/3, 4/3, 0], [1, 2, 1], [0, 4/3, 8/3]])
np.testing.assert_allclose(s['Q'].sum(axis=0), [11/36, 7/18, 11/36])
# The argmax baseline counts the unassigned rows and wrongly adds ID 12 to review.
argmax_disagreements = (np.flatnonzero(p.argmax(axis=1) != y)+1).tolist()
assert argmax_disagreements == [3, 7, 8, 11, 12]
assert int(s['C'].sum()) == 10 and len(y) == 12
assert np.isclose((s['estimated_counts'].sum()-s['estimated_counts'].trace()), 14/3)

# Permuting records changes neither the thresholds nor the raw count table.
rng = np.random.default_rng(21)
for _ in range(30):
    permutation = rng.permutation(len(y))
    pp, yy = p[permutation], y[permutation]
    tt = np.array([pp[yy == j, j].mean() for j in range(3)])
    np.testing.assert_allclose(tt, t)
    counted = np.zeros((3, 3), dtype=int)
    for label, row in zip(yy, pp):
        passing = np.flatnonzero(row >= tt)
        if len(passing): counted[label, passing[0]] += 1
    np.testing.assert_array_equal(counted, oracle)

# Candidate IDs cannot distinguish a model error from a true annotation error.
# Declaring ID 11 a valid C record leaves the same probabilities/candidate list.
reviewed_true = y.copy()
reviewed_true[[2, 6, 7]] = [1, 0, 2]
assert reviewed_true[10] == y[10] == 2
flagged = s['candidates']
confirmed = reviewed_true != y
assert int((flagged & confirmed).sum()) == 3
assert int(flagged.sum()) == 4

print(json.dumps({
    'chapter_sha256': journal['after_sha256'], 'actual_new_programs': len(blocks),
    'synthetic_oof_fits': 3, 'probability_rows': 12,
    'permutation_checks': 30, 'raw_counted_rows': 10,
    'review_ids': s['ids'][s['order']].tolist(),
    'toy_review_precision': .75, 'toy_review_not_a_measurement': True,
    'numpy': np.__version__, 'sklearn': sklearn.__version__,
    'cleanlab_package_executed': False, 'paper_benchmarks_reproduced': False,
}, indent=2))
