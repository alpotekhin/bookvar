"""Execute sequential-recommendation examples and check data/masking contracts."""
from pathlib import Path
import ast
import hashlib
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 11
scope = {}
for number, code in enumerate(blocks, 1):
    print(f"Execute actual sequential block {number}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{number}", "exec"), scope)

source = ROOT / "05 Источники/Originals/recommender-sequential-2026-10-25"
for name, expected in {
    "model.py": "1e407540b01350a7218ae82be096b3adb5e018dbe3f718a9e2a24b1e43ba97fd",
    "sampler.py": "22721d1b4e7b15402b36fa2db1368eb8efd7ad1c6742dc1adb6608b12201fa24",
    "ssept.py": "b55a8e8e29b2761976e2c0ac1656a36080f6e5137d0895110ed400ed3350f393",
}.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected

# Only the fully read pure pad helper and nested sampler body are executed.
# No upstream module import, training, process creation, queue or network call.
tree = ast.parse((source/"model.py").read_text())
pad = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pad_sequences")
upstream = {"np": np}
exec(compile(ast.Module(body=[pad], type_ignores=[]), "pinned-pad", "exec"), upstream)
tree = ast.parse((source/"sampler.py").read_text())
random_neq = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "random_neq")
wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_function")
nested = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
sampler_scope = {"np": np, "user_train": {1: [1, 2, 3, 4]},
                 "usernum": 1, "itemnum": 6, "maxlen": 5}
exec(compile(ast.Module(body=[random_neq, nested], type_ignores=[]), "pinned-sample", "exec"), sampler_scope)
np.random.seed(25)
for length in range(1, 9):
    sampler_scope["maxlen"] = length
    user, seq, pos, neg = sampler_scope["sample"]()
    own_seq, own_pos = scope["shifted_train"]([1, 2, 3, 4], length)
    np.testing.assert_array_equal(seq, own_seq)
    np.testing.assert_array_equal(pos, own_pos)
    assert user == 1
    assert set(neg[pos != 0]).issubset({5, 6})
    assert np.all(neg[pos == 0] == 0)
for n in range(10):
    for length in range(1, 10):
        values = list(range(1, n+1))
        np.testing.assert_array_equal(scope["pad_left"](values, length),
            upstream["pad_sequences"]([values], length)[0])
        if n >= 3:
            train, valid, test = scope["last_two_split"](values)
            pairs = scope["prefix_examples"](train, length)
            assert all(valid not in context and test not in context for context, _ in pairs)
            for index, (context, target) in enumerate(pairs, start=1):
                assert target == values[index]
                assert context == values[max(0, index-length):index]
print("PASS pinned pad/nested sampler, 90 pad cases and split-before-window contracts")

rng = np.random.default_rng(20261025)
attention = scope["causal_attention"]
for _ in range(100):
    L, d = int(rng.integers(2, 9)), int(rng.integers(1, 6))
    pad_count = int(rng.integers(0, L))
    ids = np.r_[np.zeros(pad_count, dtype=int), np.arange(1, L-pad_count+1)]
    vectors = rng.normal(size=(L, d))
    output, weights = attention(vectors, ids)
    for t in range(L):
        if ids[t] == 0:
            np.testing.assert_array_equal(output[t], np.zeros(d))
            continue
        allowed = [j for j in range(t+1) if ids[j] != 0]
        logits = np.array([vectors[t]@vectors[j]/np.sqrt(d) for j in allowed])
        prob = np.exp(logits-np.max(logits))
        prob /= prob.sum()
        direct = sum(w*vectors[j] for w, j in zip(prob, allowed))
        np.testing.assert_allclose(output[t], direct, atol=2e-14)
        np.testing.assert_allclose(weights[t].sum(), 1.)
        future = vectors.copy()
        future[t+1:] = rng.normal(size=future[t+1:].shape)*50
        np.testing.assert_allclose(attention(future, ids)[0][t], output[t], atol=2e-14)
    changed = vectors.copy()
    changed[:pad_count] = 10000
    np.testing.assert_allclose(attention(changed, ids)[0], output, atol=2e-14)
zero_output, zero_weights = attention(np.ones((4, 2)), np.zeros(4, dtype=int))
assert np.isfinite(zero_output).all() and np.isfinite(zero_weights).all()
np.testing.assert_array_equal(zero_output, np.zeros((4, 2)))
assert attention(np.array([[1., -1.]]), np.array([1]))[1][0, 0] == 1
print("PASS 100 independent attention systems, future/PAD invariance and fully masked rows")

# Independent scalar BCE and finite differences on all non-PAD logits.
positive, negative, mask = (scope[k] for k in ["pos_logits", "neg_logits", "target_mask"])
params = np.stack([positive, negative])
def objective(z):
    return sum(float(np.logaddexp(0., -z[0, r, c])+np.logaddexp(0., z[1, r, c]))
               for r, c in np.argwhere(mask)) / mask.sum()
expected = np.zeros_like(params)
expected[0][mask] = -np.exp(-np.logaddexp(0., positive[mask]))/mask.sum()
expected[1][mask] = np.exp(-np.logaddexp(0., -negative[mask]))/mask.sum()
for index in np.ndindex(params.shape):
    above, below = params.copy(), params.copy()
    above[index] += 1e-6
    below[index] -= 1e-6
    np.testing.assert_allclose((objective(above)-objective(below))/2e-6,
                               expected[index], atol=2e-9)
np.testing.assert_allclose(objective(params), scope["masked_loss"])
try:
    scope["masked_sequence_bce"](positive, negative, np.zeros_like(mask))
except ValueError:
    pass
else:
    raise AssertionError("All-PAD loss must not silently return NaN")
print("PASS masked BCE independent sum and all twelve positive/negative derivatives")

replace_ids, choose = scope["replace_ids"], scope["choose_training_window"]
for _ in range(100):
    result = replace_ids(np.arange(11), .5, np.arange(1, 11), rng)
    assert result[0] == 0 and set(result[1:]).issubset(set(range(1, 11)))
    history = list(range(20))
    result = choose(history, 5, 1., rng)
    start = result[0]
    assert 0 <= start < 15 and result == history[start:start+5]
    assert choose(history, 5, 0., rng) == list(range(15, 20))
assert choose([1, 2], 5, 1., rng) == [1, 2]
assert scope["negative_pool_at"]({"A": 1}, 1, [], "A") == set()
assert scope["fit_categories"](["z", "a", "a"], 2) == {"a": 1}
assert scope["actual_history"] == ["A", "C"]
print("PASS replacement/window sampling, OOV, availability and delayed-event controls")
print("Chapter SHA-256:", hashlib.sha256(content).hexdigest())
