#!/usr/bin/env python3
"""Execute the chapter's examples and independently check its numerical claims."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from math import exp, inf, isclose, log, prod
from pathlib import Path
from random import Random
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель.md"
text = CHAPTER.read_text()
assert chr(0xA4) not in text
blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
assert len(blocks) == 6, len(blocks)
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    if number == 5:
        metrics = namespace["corpus_metrics"]
        finite_loss, overflow_ppl = metrics([[1e-320]])
        assert finite_loss < inf and overflow_ppl == inf
        for invalid in [[[]], [[-0.1]], [[1.1]], [[float("nan")]]]:
            try:
                metrics(invalid)
                raise AssertionError("Invalid input was accepted")
            except ValueError:
                pass
    print(f"PASS published standalone block {number}")

corpus = ["кот кот спит", "пёс спит", "кот спит бежит"]
V = ("кот", "спит", "пёс", "бежит", "</s>", "<unk>")
BOS, EOS = "<s>", "</s>"

def fit(order, documents=corpus):
    out = defaultdict(Counter)
    for document in documents:
        sequence = [BOS] * (order - 1) + document.split() + [EOS]
        for t in range(order - 1, len(sequence)):
            h = tuple(sequence[t - order + 1:t]) if order > 1 else ()
            out[h][sequence[t]] += 1
    return out

unigrams, bigrams, trigrams = fit(1), fit(2), fit(3)
assert [unigrams[()][w] for w in V] == [3, 3, 1, 1, 3, 0]
assert sum(unigrams[()].values()) == 11
assert sum(sum(row.values()) for row in bigrams.values()) == 11
assert sum(sum(row.values()) for row in trigrams.values()) == 11
assert bigrams[(BOS,)] == {"кот": 2, "пёс": 1}
assert (EOS,) not in bigrams
assert (BOS, BOS) in trigrams
assert (EOS, BOS) not in trigrams
assert fit(2, [""])[(BOS,)][EOS] == 1

def mle(row):
    n = sum(row.values())
    return {w: F(row[w], n) for w in V} if n else None

p1 = mle(unigrams[()])
p2 = mle(bigrams[("спит",)])
p3 = mle(trigrams[("кот", "спит")])
assert (p1[EOS], p2[EOS], p3[EOS]) == (F(3, 11), F(2, 3), F(1, 2))
assert mle(trigrams[("пёс", "спит")])[EOS] == 1
assert mle(Counter()) is None
for row in bigrams.values():
    assert sum(mle(row).values()) == 1

def probability(words, order=2):
    rows = fit(order)
    sequence = [BOS] * (order - 1) + list(words) + [EOS]
    factors = []
    for t in range(order - 1, len(sequence)):
        row = rows.get(tuple(sequence[t - order + 1:t]), Counter())
        p = mle(row)
        if p is None:
            return None
        factors.append(p[sequence[t]])
    return prod(factors)

assert probability(["кот", "спит"]) == F(8, 27)
assert probability(["кот", "спит", "бежит"]) == F(4, 27)
assert probability(["пёс", "спит", "бежит"]) == F(1, 9)
assert probability(["пёс", "спит"]) == F(2, 9)
assert probability(["пёс", "спит"], order=3) == F(1, 3)
assert probability(["пёс", "спит", "бежит"], order=3) == 0
assert probability(["кот", "бежит"]) == 0
assert F(8, 27) + F(4, 27) == F(4, 9)
# Sum every supported finite form, with the sole loop accounted for exactly.
cat_mass = (F(8, 27) + F(4, 27)) / (1 - F(1, 3))
dog_mass = F(2, 9) + F(1, 9)
assert cat_mass + dog_mass == 1
assert (F(1, 3) ** 25) < F(1, 10**11)

# A conditional MLE row is the likelihood maximum, not an arbitrary normalization.
optimum = log(1/3) + 2 * log(2/3)
for a, b in [(1/2, 1/2), (.2, .8), (.4, .5), (.01, .99)]:
    assert log(a) + 2 * log(b) < optimum
assert sum(mle(Counter({"кот": 1, "спит": 2})).values()) == 1

def add_k(row, k, vocabulary=V):
    n = sum(row.values())
    return {w: (F(row[w]) + k) / (n + k * len(vocabulary))
            for w in vocabulary}

row = bigrams[("кот",)]
for k in [F(1), F(1, 10), F(1, 1000), F(1000)]:
    p = add_k(row, k)
    assert sum(p.values()) == 1 and min(p.values()) > 0
    mixture = F(3) / (3 + 6 * k)
    assert all(p[w] == mixture * mle(row)[w] + (1 - mixture) / 6 for w in V)
assert add_k(row, F(1))["спит"] == F(1, 3)
assert add_k(row, F(1, 10))["спит"] == F(7, 12)
assert add_k(row, F(1, 10))["бежит"] == F(1, 36)
assert set(add_k(Counter(), F(1)).values()) == {F(1, 6)}
assert F(3 + 1, 3 + 50000) == F(4, 50003)
assert "лиса" not in add_k(row, F(1))

weights = [F(3, 5), F(3, 10), F(1, 10)]
mix = {w: sum(a * p[w] for a, p in zip(weights, [p3, p2, p1])) for w in V}
assert sum(mix.values()) == 1
assert mix[EOS] == F(29, 55)
assert mix["бежит"] == F(9, 22)
assert mix["<unk>"] == 0
assert weights[1] / sum(weights[1:]) == F(3, 4)
assert weights[2] / sum(weights[1:]) == F(1, 4)
fallback = {w: F(3, 4) * p2[w] + F(1, 4) * p1[w] for w in V}
assert sum(fallback.values()) == 1

seen = {w for w in V if row[w]}
q = p1
direct = {w: max(F(row[w]) - F(1, 2), F(0)) / 3 for w in V}
leftover = 1 - sum(direct.values())
lower_mass = sum(q[w] for w in V if w not in seen)
assert (leftover, lower_mass) == (F(1, 3), F(5, 11))
alpha = leftover / lower_mass
backoff = {w: direct[w] if w in seen else alpha * q[w] for w in V}
assert [backoff[w] for w in V] == [F(1, 6), F(1, 2), F(1, 15), F(1, 15), F(1, 5), 0]
assert sum(backoff.values()) == 1
assert 1 + lower_mass == F(16, 11)
assert 1 + F(2, 5) * lower_mass == F(13, 11)
bad_base = {w: int(w == "кот") for w in V}
assert sum(bad_base[w] for w in V if w not in seen) == 0

left_contexts = {w: {h for h, row in bigrams.items() if row[w] > 0} for w in V}
types = sum(map(len, left_contexts.values()))
assert types == 8
continuation = {w: F(len(left_contexts[w]), types) for w in V}
assert list(continuation.values()) == [F(1, 4), F(1, 4), F(1, 8), F(1, 8), F(1, 4), 0]
assert sum(continuation.values()) == 1

def kn(row, base=continuation, discount=F(1, 2)):
    total = sum(row.values())
    if not total:
        return base
    freed = discount * sum(row[w] > 0 for w in V) / total
    return {w: max(F(row[w]) - discount, F(0)) / total + freed * base[w] for w in V}

actual = kn(bigrams[("кот",)])
assert list(actual.values()) == [F(1, 4), F(7, 12), F(1, 24), F(1, 24), F(1, 12), 0]
for row in [*bigrams.values(), Counter()]:
    assert sum(kn(row).values()) == 1
positive_base = {w: F(9, 10) * continuation[w] + F(1, 60) for w in V}
assert kn(bigrams[("кот",)], positive_base)["<unk>"] == F(1, 180)
assert min(positive_base.values()) > 0 and sum(positive_base.values()) == 1
# Repeated raw trigrams are not repeated distinct left contexts.
three = Counter({("a", "b", "c"): 100, ("d", "b", "c"): 1})
assert sum(three.values()) == 101
assert len({h for h, b, c in three if (b, c) == ("b", "c")}) == 2

def nll(probabilities):
    return -sum(log(float(p)) for p in probabilities) if all(probabilities) else inf

test_p = [F(1, 3), F(1), F(2, 3)]
loss = nll(test_p) / 3
assert isclose(loss, .5013591322587581)
assert isclose(exp(loss), 1.6509636244473134)
assert isclose(loss / log(2), .7233083338141042)
assert isclose(2 ** (loss / log(2)), exp(loss))
assert isclose(exp(nll(test_p[:2]) / 2), 3**.5)
assert isclose(exp(-1000 * log(.01) / 1000), 100.)
assert .01**1000 == 0.
assert isclose(1000 * log(.01), -4605.170185988091)
assert nll([1., 0.]) == inf
assert exp(nll([1., 0.])) == inf
assert isclose(exp(3 * log(9) / 4), 5.196152422706632)
assert exp(3 * log(9) / 4) != (1 + 9) / 2
# Correct aggregation is weighted in log space; a fake padding event changes N.
assert exp(nll(test_p) / 3) != exp(nll(test_p + [F(1)]) / 4)
for size in [1, 2, 6, 1000]:
    assert isclose(exp(-log(1/size)), size)

# Draw by cumulative mass and compare to the implemented weighted sampler.
def inverse_cdf(row, u):
    total = sum(row.values())
    edge = 0
    for w, count in row.items():
        edge += count / total
        if u < edge:
            return w
    raise AssertionError(u)

assert inverse_cdf(bigrams[(BOS,)], .2) == "кот"
assert inverse_cdf(bigrams[(BOS,)], .8) == "пёс"
assert inverse_cdf(bigrams[("кот",)], .8) == "спит"
h, greedy = (BOS,), []
while True:
    w = max(bigrams[h], key=bigrams[h].get)
    greedy.append(w)
    if w == EOS:
        break
    h = (w,)
assert greedy == ["кот", "спит", EOS]
rng = Random(13)
samples = Counter(rng.choices(["кот", "спит"], weights=[1, 2], k=50000))
assert abs(samples["кот"] / 50000 - 1/3) < .01

print("PASS boundaries/counts/MLE; Markov order; sequence/prefix/EOS; add-k/OOV; interpolation/fallback; normalized/stupid backoff; continuation/KN/full support; PPL/log bases/aggregation/underflow; sampling/greedy/termination.")
