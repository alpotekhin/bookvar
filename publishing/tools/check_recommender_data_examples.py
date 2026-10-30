"""Run the actual recommendation-data chapter and independent boundary checks."""
from pathlib import Path
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import combinations
from collections import Counter
import hashlib
import random
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 7
scope = {}
for index, code in enumerate(blocks, 1):
    print(f"Execute actual chapter block {index}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

assert abs(scope["mse"](scope["truth"], scope["prediction_a"]) - 1.01 / 3) < 1e-12
assert abs(scope["mse"](scope["truth"], scope["prediction_b"]) - 2 / 3) < 1e-12
print("PASS rating-error / top-one counterexample")

clock = scope["clock"]
labels = scope["click_labels"]
impressions, clicks = scope["impressions"], scope["clicks"]
assert [row[2] for row in labels(impressions, clicks, clock("09:59"),
                                timedelta(minutes=10))] == [None] * 4
assert [row[2] for row in labels(impressions, clicks, clock("10:26"),
                                timedelta(minutes=15))] == [1, 0, 1, None]
at_end = [("r1", "B", clock("10:10"))]
assert labels(impressions, at_end, clock("10:10"),
              timedelta(minutes=10))[1][2] == 1
for wrong_click in [("wrong", "B", clock("10:01")),
                    ("r1", "B", clock("10:00")),
                    ("r1", "B", clock("10:11"))]:
    assert labels(impressions, [wrong_click], clock("10:10"),
                  timedelta(minutes=10))[1][2] == 0
for window in [timedelta(0), timedelta(minutes=-1)]:
    try:
        labels(impressions, clicks, clock("10:26"), window)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonpositive window must fail")
print("PASS mature/censored labels, exact boundaries and request attribution")

parts = [scope["train"], scope["validation"], scope["test"]]
ids = [{e["id"] for e in part} for part in parts]
assert ids[0] | ids[1] | ids[2] == {e["id"] for e in scope["events"]}
assert not any(ids[a] & ids[b] for a, b in [(0, 1), (0, 2), (1, 2)])
assert max(e["at"] for e in parts[0]) < min(e["at"] for e in parts[1])
assert max(e["at"] for e in parts[1]) < min(e["at"] for e in parts[2])
train_users = {e["user"] for e in parts[0]}
assert {e["user"] for e in parts[2]} - train_users == {"Vera"}
assert Counter(e["item"] for e in parts[0]) == {"A": 1, "B": 2}
assert max(e["at"] for e in scope["user_train"]) > min(
    e["at"] for e in scope["user_test"])
assert "Vera" not in {e["user"] for e in scope["user_train"]}
print("PASS global intervals, cold slice and per-user chronology counterexample")

join = scope["point_in_time"]
rows = scope["feature_rows"]
base = scope["base"]
assert join(rows, 999, base, timedelta(hours=2)) is None
assert join(rows, 101, base, timedelta(0))["trips"] == 0
assert join(rows, 101, base + timedelta(hours=7),
            timedelta(hours=2))["trips"] == 8
assert join(rows, 101, base + timedelta(hours=7, seconds=1),
            timedelta(hours=2)) is None
try:
    join(rows, 101, base, timedelta(seconds=-1))
except ValueError:
    pass
else:
    raise AssertionError("Negative TTL must fail")

rng = random.Random(20261025)
for _ in range(200):
    random_rows = [
        dict(driver=rng.choice([101, 102]),
             event_time=base + timedelta(minutes=rng.randrange(-120, 600)),
             available_at=base + timedelta(minutes=rng.randrange(-120, 720)),
             marker=i)
        for i in range(25)
    ]
    at = base + timedelta(minutes=rng.randrange(-150, 750))
    ttl = timedelta(minutes=rng.randrange(0, 300))
    driver = rng.choice([101, 102, 999])
    for require_available in [True, False]:
        best = None
        for row in random_rows:
            if row["driver"] != driver or row["event_time"] > at:
                continue
            if at - row["event_time"] > ttl:
                continue
            if require_available and row["available_at"] > at:
                continue
            if best is None or (row["event_time"], row["available_at"]) > (
                    best["event_time"], best["available_at"]):
                best = row
        assert join(random_rows, driver, at, ttl, require_available) == best
        assert join(list(reversed(random_rows)), driver, at, ttl,
                    require_available) == best
assert scope["safe"]["earnings"] == 50 and scope["leaky"]["earnings"] == 999
print("PASS five original Feast rows, late correction and 400 independent join cases")

sample = scope["sample_unobserved"]
date = scope["date"]
catalog = scope["catalog"]
for user in ["Anya", "Boris", "Vera"]:
    for day in range(1, 8):
        at = date(day)
        available = {i for i, pub in catalog.items() if pub <= at}
        seen = {e["item"] for e in scope["events"]
                if e["user"] == user and e["at"] < at}
        for positive in sorted(available):
            allowed = available - seen - {positive}
            for k in range(len(allowed) + 1):
                result = sample(user, positive, at, k, seed=day)
                assert len(set(result)) == len(result) == k
                assert set(result) <= allowed
                assert result == sample(user, positive, at, k, seed=day)
            try:
                sample(user, positive, at, len(allowed) + 1)
            except ValueError:
                pass
            else:
                raise AssertionError("Exhausted pool must fail")
for positive, at, k in [("F", date(2), 1), ("missing", date(2), 0),
                         ("B", date(2), -1)]:
    try:
        sample("Anya", positive, at, k)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid negative-sampling input must fail")
print("PASS availability, past positives, deterministic sampling and exhausted pools")

# Figure rec-ranking.svg: comparisons for u1, columns i and rows j.
observed = {1, 2}  # zero-based i2 and i3
pair_matrix = []
for j in range(4):
    pair_matrix.append([
        None if i == j else (1 if i in observed and j not in observed
                              else -1 if j in observed and i not in observed
                              else 0)
        for i in range(4)
    ])
assert pair_matrix == [[None, 1, 1, 0], [-1, None, 0, -1],
                       [-1, 0, None, -1], [0, 1, 1, None]]
assert sum(value == 1 for row in pair_matrix for value in row) == 4
print("PASS original D2L u1 pairwise-matrix signs and four comparisons")

universe = range(5)
sets = [set(c) for n in range(6) for c in combinations(universe, n)]
checked = 0
for G in sets:
    if not G:
        continue
    for C in sets:
        if not G & C:
            continue
        for F in sets:
            if not F <= C:
                continue
            candidate = Fraction(len(G & C), len(G))
            conditional = Fraction(len(G & F), len(G & C))
            final = Fraction(len(G & F), len(G))
            assert final == candidate * conditional
            assert final <= candidate
            checked += 1
macro_final = (Fraction(0, 1) + Fraction(1, 4)) / 2
wrong_product = ((1 + Fraction(1, 4)) / 2) * (Fraction(1, 2))
assert macro_final == Fraction(1, 8)
assert wrong_product == Fraction(5, 16)
assert macro_final != wrong_product
assert scope["candidate_sources"]["B"] == ["history", "popular"]
print(f"PASS {checked} exact recall factorizations and macro-average counterexample")

popular = scope["recommend_popular"]
assert scope["popularity"] == Counter({"A": 1, "B": 2})
assert popular("Vera", date(3), 0) == []
for user in ["Anya", "Boris", "Vera"]:
    for day in range(3, 8):
        at = date(day)
        # Frozen train counts; request-time history is a separate input.
        available = {i for i, pub in catalog.items() if pub <= at}
        seen = {e["item"] for e in scope["events"]
                if e["user"] == user and e["at"] < at}
        expected = sorted(available - seen,
                          key=lambda i: (-{"A": 1, "B": 2}.get(i, 0), i))
        assert popular(user, at, 100) == expected
for at in [date(1), date(2)]:
    try:
        popular("Vera", at, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("The model cannot serve a request before its cutoff")
print("PASS frozen train-only popularity, cold user, tie-breaking and short slate")
print("Chapter SHA-256:", hashlib.sha256(content).hexdigest())
