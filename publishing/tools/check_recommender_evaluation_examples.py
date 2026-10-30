"""Execute actual chapter fences and compare selected pinned Recommenders functions."""
from pathlib import Path
from math import isclose, log2, sqrt
from itertools import permutations, combinations
import ast
import hashlib
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 7
scope = {}
for number, code in enumerate(blocks, 1):
    print(f"Execute actual evaluation block {number}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{number}", "exec"), scope)

metrics = scope["ranking_metrics"]
# Exhaustive independent rank-position calculation for nonempty relevance sets.
cases = 0
for n in range(1, 5):
    universe = list(range(n))
    for count in range(1, n + 1):
        for relevant in combinations(universe, count):
            for length in range(n + 1):
                for items in permutations(universe, length):
                    for k in range(1, 6):
                        result = metrics(items, relevant, k)
                        ranks = [i + 1 for i, item in enumerate(items[:k]) if item in relevant]
                        hits = len(ranks)
                        ap_sum = sum(j / rank for j, rank in enumerate(ranks, 1))
                        ideal = sum(1 / log2(i + 1) for i in range(1, min(count, k) + 1))
                        expected = {
                            "precision": hits / k, "recall": hits / count,
                            "hit": float(hits > 0), "ap": ap_sum / min(count, k),
                            "ap_all_relevant": ap_sum / count,
                            "ndcg": sum(1 / log2(i + 1) for i in ranks) / ideal,
                        }
                        for name, value in expected.items():
                            assert isclose(result[name], value, abs_tol=1e-14), (name, items, relevant, k)
                            assert 0 <= result[name] <= 1 + 1e-14
                        cases += 1
for k in (0, -1, True, 1.5):
    try:
        metrics([1], {1}, k)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid k accepted: {k}")
try:
    metrics([1, 1], {1}, 2)
except ValueError:
    pass
else:
    raise AssertionError("Duplicates must be rejected")
assert metrics([], set(), 1) is None
print(f"PASS {cases} independent rank-position cases and boundary policies")

source = ROOT / "05 Источники/Originals/recommender-evaluation-2026-10-25"
for name, expected in {
    "python_evaluation.py": "48b9680b189d72d7962aefa758aee509e0de9a8a2042c3b11c8221ef7e97c8cd",
    "spark_evaluation.py": "3fbdfcb6be2bd41279af11c043f72c2fdde3a0f1820153e23dab261c83aac686",
}.items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected

names = {"merge_rating_true_pred", "merge_ranking_true_pred", "precision_at_k",
         "recall_at_k", "ndcg_at_k", "_get_reciprocal_rank", "map", "map_at_k", "get_top_k_items"}
tree = ast.parse((source / "python_evaluation.py").read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in functions} == names
# Omit only cache/column-validation wrappers; evaluate the actual read function bodies.
for node in functions:
    node.decorator_list = []
upstream = {"np": np, "pd": pd, "DEFAULT_USER_COL": "userID", "DEFAULT_ITEM_COL": "itemID",
            "DEFAULT_RATING_COL": "rating", "DEFAULT_PREDICTION_COL": "prediction",
            "DEFAULT_K": 10, "DEFAULT_THRESHOLD": 0}
exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned-ranking-functions", "exec"), upstream)
truth = pd.DataFrame({"userID": [1, 2, 3], "itemID": ["B"] * 3, "rating": [1.] * 3})
pred = pd.DataFrame({"userID": [1, 2], "itemID": ["B", "C"], "prediction": [1., 1.]})
assert upstream["precision_at_k"](truth, pred, k=2) == .25
assert upstream["recall_at_k"](truth, pred, k=2) == .5
assert upstream["map_at_k"](truth, pred, k=2) == .5
assert isclose(upstream["ndcg_at_k"](truth, pred, k=2), 1 / 3)
truth10 = pd.DataFrame({"userID": [1] * 10, "itemID": list(range(10)), "rating": [1.] * 10})
pred2 = pd.DataFrame({"userID": [1, 1], "itemID": [0, 1], "prediction": [2., 1.]})
assert upstream["map_at_k"](truth10, pred2, k=2) == 1.
assert upstream["map"](truth10, pred2, k=2) == .2
true_rating = pd.DataFrame({"userID": [1, 1, 1], "itemID": [0, 1, 2], "rating": [5, 3, 1]})
partial = pd.DataFrame({"userID": [1, 1], "itemID": [0, 1], "prediction": [4, 3]})
actual, predicted = upstream["merge_rating_true_pred"](true_rating, partial)
assert len(actual) == 2 and isclose(sqrt(np.mean((actual - predicted) ** 2)), sqrt(.5))
offset_errors = np.array([5., 3., 1.]) - np.array([6., 4., 2.])
assert isclose(1 - sum(offset_errors ** 2) / sum((np.array([5., 3., 1.]) - 3.) ** 2), .625)
assert 1 - np.var(offset_errors) / np.var([5., 3., 1.]) == 1
print("PASS pinned MAP denominator variants, missing-user joins, NDCG distinction and rating join")

assert scope["no_eligible_target"] == ["q4"]
assert scope["replay_rows"][2][1] == []
assert isclose(np.mean([r[3] for r in scope["replay_rows"]]), 1 / 3)
assert scope["diverse_top_k"]({}, {}, 2, .3) == []
assert scope["diverse_top_k"]({"A": 1}, {"A": (1., 0.)}, 3, .3) == ["A"]
assert isclose(scope["exposure_entropy"]([4, 4]), 1.)
assert isclose(scope["exposure_entropy"]([40, 40]), 1.)
assert isclose(scope["standard_error"], sqrt(.1956 / 10000))
assert np.allclose(scope["confidence_interval"], [.011331568999, .028668431001])
assert isclose(scope["planned_mde"], .011879393923934)
print("PASS complete replay, reranking, entropy and A/B calculations")
print("CHAPTER SHA256", hashlib.sha256(content).hexdigest())
