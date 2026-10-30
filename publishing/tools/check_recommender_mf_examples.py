"""Execute the MF chapter and verify gradients, ALS, SAR and retrieval algebra."""
from pathlib import Path
import ast
import hashlib
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 11
scope = {}
for index, code in enumerate(blocks, 1):
    print(f"Execute actual chapter block {index}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

# Independent ridge solution for all user/item biases in a single system.
target, mask, mu = (scope[k] for k in ["target", "mask", "mu"])
pairs = np.argwhere(mask)
design = np.zeros((len(pairs), sum(target.shape)))
for row, (u, i) in enumerate(pairs):
    design[row, u] = 1
    design[row, target.shape[0] + i] = 1
direct = np.linalg.solve(design.T @ design + np.eye(design.shape[1]),
                         design.T @ (target[mask] - mu))
np.testing.assert_allclose(np.r_[scope["bu_bias"], scope["bi_bias"]], direct,
                           atol=1e-12)
assert abs(np.dot([.8, .2], [5, 1]) / sum([.8, .2]) - 4.2) < 1e-12
assert abs(3.5 + np.dot([.8, .2], np.array([5, 1]) - [4, 2]) - 4.1) < 1e-12
print("PASS joint ridge solution, first coordinate sweep and neighbor arithmetic")

# Execute only the fully read pure helper definitions from the pinned source.
source = ROOT / "05 Источники/Originals/recommender-mf-2026-10-25/python_utils.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "fbe5a3559912a632d6b74678fbc1b7a3463f6fd4bf8b535872af664e4d0e5873")
names = {"exponential_decay", "_get_row_and_column_matrix", "jaccard", "lift"}
tree = ast.parse(source.read_text())
defs = [node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in defs} == names
helpers = {"np": np}
exec(compile(ast.Module(body=defs, type_ignores=[]), str(source), "exec"), helpers)
rng = np.random.default_rng(20261025)
for _ in range(40):
    H = np.vstack([rng.integers(0, 2, size=(7, 5)), np.ones((1, 5))])
    C = H.T @ H
    J = helpers["jaccard"](C)
    L = helpers["lift"](C)
    freq = np.diag(C)
    np.testing.assert_allclose(J, C / (freq[:, None] + freq[None, :] - C))
    np.testing.assert_allclose(L, C / (freq[:, None] * freq[None, :]))
    m = len(H)
    probability_lift = (C / m) / ((freq[:, None] / m) * (freq[None, :] / m))
    np.testing.assert_allclose(probability_lift, m * L)
    for i in range(5):
        for j in range(5):
            left, right = set(np.flatnonzero(H[:, i])), set(np.flatnonzero(H[:, j]))
            np.testing.assert_allclose(J[i, j], len(left & right) / len(left | right))
np.testing.assert_allclose(
    helpers["exponential_decay"](np.array([-120., 0.]), 0., 30.),
    [1/16, 1])
np.testing.assert_allclose(helpers["jaccard"](scope["cooccurrence"]),
                           scope["similarity"])
np.testing.assert_allclose(helpers["lift"](scope["cooccurrence"]),
                           scope["sar_lift"])
print("PASS 40 SAR overlap sets and actual pinned helpers, lift convention and decay")

objective = scope["explicit_objective"]
for _ in range(15):
    P = rng.normal(size=(3, 2))
    Q = rng.normal(size=(4, 2))
    bu, bi = rng.normal(size=3), rng.normal(size=4)
    penalty = .4
    residual = np.where(mask, mu + bu[:, None] + bi[None, :] + P @ Q.T - target, 0.)
    gradients = [residual @ Q + penalty * P,
                 residual.T @ P + penalty * Q,
                 residual.sum(axis=1) + penalty * bu,
                 residual.sum(axis=0) + penalty * bi]
    arrays = [P, Q, bu, bi]
    for array, gradient in zip(arrays, gradients):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + 1e-6
            above = objective(*arrays, target, mask, mu, penalty)
            array[index] = original - 1e-6
            below = objective(*arrays, target, mask, mu, penalty)
            array[index] = original
            np.testing.assert_allclose((above - below) / 2e-6,
                                       gradient[index], rtol=2e-6, atol=2e-7)
    changed = target.copy()
    changed[~mask] = 999
    np.testing.assert_allclose(objective(*arrays, changed, mask, mu, penalty),
                               objective(*arrays, target, mask, mu, penalty))
wrong_q = scope["q_old"] + .1 * scope["error"] * scope["p_new"]
assert not np.allclose(wrong_q, scope["q_new"])
assert scope["train_rmse"] < .06
assert scope["explicit_losses"][-1] < .6
print("PASS 315 explicit finite differences, ignored missing targets and stale-parameter control")

dense = scope["implicit_user_update"]
fast = scope["implicit_user_update_sparse"]
for _ in range(100):
    d = int(rng.integers(1, 5))
    n = int(rng.integers(5, 12))
    Q = rng.normal(size=(n, d))
    counts = rng.integers(0, 5, size=n).astype(float)
    alpha = float(rng.uniform(0, 5))
    penalty = float(rng.uniform(.1, 3))
    c = 1 + alpha * counts
    a = counts > 0
    lhs = np.vstack([np.sqrt(c)[:, None] * Q, np.sqrt(penalty) * np.eye(d)])
    rhs = np.r_[np.sqrt(c) * a, np.zeros(d)]
    independently = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    result = dense(Q, counts, alpha, penalty)
    np.testing.assert_allclose(result, independently, atol=1e-11)
    np.testing.assert_allclose(fast(Q, counts, Q.T @ Q, alpha, penalty),
                               result, atol=1e-11)
    gradient = 2 * (Q.T @ (c * (Q @ result - a)) + penalty * result)
    np.testing.assert_allclose(gradient, 0, atol=1e-10)
    positive = np.flatnonzero(counts > 0)
    if len(positive):
        target_index = int(rng.integers(n))
        G = Q.T @ (c[:, None] * Q) + penalty * np.eye(d)
        contributions = [c[j] * Q[target_index] @ np.linalg.solve(G, Q[j])
                         for j in positive]
        np.testing.assert_allclose(sum(contributions), Q[target_index] @ result,
                                   atol=1e-11)
for args in [(np.array([-1., 0., 0.]), 2., 1.),
             (np.zeros(3), -1., 1.), (np.zeros(3), 2., 0.),
             (np.zeros(2), 2., 1.)]:
    try:
        dense(scope["Q_als"], *args)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid ALS input must fail")
assert np.all(np.diff(scope["implicit_losses"]) <= 1e-10)
np.testing.assert_allclose(scope["contributions"], [.5, -.5])
print("PASS 100 independent weighted least-squares systems, sparse equivalence and history explanations")

# A final P block is optimal for its fixed Q; perturbing it cannot lower the loss.
counts = scope["implicit_counts"]
Q = scope["Q_imp"]
P_opt = np.array([dense(Q, row) for row in counts])
base_loss = scope["implicit_objective"](P_opt, Q, counts)
for _ in range(30):
    perturbed = P_opt + rng.normal(scale=.05, size=P_opt.shape)
    assert scope["implicit_objective"](perturbed, Q, counts) >= base_loss - 1e-12
for _ in range(50):
    P = rng.normal(size=(7, 3))
    Q = rng.normal(size=(9, 3))
    bias = rng.normal(size=9)
    explicit = P @ Q.T + bias
    augmented = np.column_stack([P, np.ones(len(P))]) @ np.column_stack([Q, bias]).T
    np.testing.assert_allclose(explicit, augmented, atol=1e-12)
print("PASS block optimum and 50 item-bias retrieval augmentations")
print("Chapter SHA-256:", hashlib.sha256(content).hexdigest())
