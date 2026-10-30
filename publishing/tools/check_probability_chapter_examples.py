"""Execute the published block and check the chapter's finite examples."""
from fractions import Fraction as Q
from itertools import product
from math import comb, isclose, log
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 1
    scope = {"__name__": "__chapter_example__"}
    exec(compile(blocks[0], str(CHAPTER), "exec"), scope)

    masses = [Q(55, 100), Q(25, 100), Q(15, 100), Q(5, 100)]
    assert sum(masses) == 1 and sum(masses[:2]) == Q(4, 5)
    density = 1 / (Q(16, 10)-Q(9, 10))
    assert density == Q(10, 7) > 1
    assert density*Q(7, 10) == 1
    assert density*(Q(12, 10)-1) == Q(2, 7)

    # Uniformly choose a row of the explicitly specified 100-record corpus.
    joint = Q(24, 100)
    pa, pb = Q(40, 100), Q(30, 100)
    pba, pab = Q(24, 40), Q(24, 30)
    assert pba == Q(3, 5) and pab == Q(4, 5)
    assert joint == pa*pba == pb*pab
    assert pba*pa/pb == pab
    assert pba*pa + Q(6, 60)*(1-pa) == pb
    assert joint != pa*pb == Q(12, 100)

    theta = Q(9, 13)
    likelihood = lambda t: t**9*(1-t)**4
    assert likelihood(theta) > likelihood(Q(1, 2)) > likelihood(Q(9, 10))
    assert 9/theta-4/(1-theta) == 0
    assert -9/theta**2-4/(1-theta)**2 < 0
    assert likelihood(0) == likelihood(1) == 0
    assert comb(13, 9) == 715
    assert sum(comb(13, k)*theta**k*(1-theta)**(13-k)
               for k in range(14)) == 1
    assert isclose(float(likelihood(theta)), 0.00032746053744694525)
    # Exact integrals of polynomials on [0,1], coefficients from constant up.
    integrate_unit = lambda coefficients: sum(Q(c, k+1) for k, c in enumerate(coefficients))
    normalization = integrate_unit([0, 1])
    assert normalization == Q(1, 2)
    assert integrate_unit([0, 1/normalization]) == 1

    sequence = Q(1, 2)*Q(3, 5)*Q(1, 4)
    assert sequence == Q(3, 40)
    nll = -log(float(sequence))
    assert isclose(nll, 2.5902671654458267)
    assert isclose(nll/3, 0.8634223884819422)
    assert isclose(-2000*log(2), -1386.2943611198905)
    assert Q(1, 2)**2000 > 0 and float(Q(1, 2)**2000) == 0

    # Repeated identical condition, different observed targets.
    conditional = lambda p: p*p*(1-p)
    best_p = Q(2, 3)
    assert conditional(best_p) == Q(4, 27)
    assert conditional(Q(1, 2)) == Q(1, 8)
    assert conditional(Q(9, 10)) == Q(81, 1000)
    assert 2/best_p-1/(1-best_p) == 0
    assert -2/best_p**2-1/(1-best_p)**2 < 0
    # One-hots averaged over the three labels equal their empirical soft target.
    one_hot_nll = -(2*log(float(best_p))+log(float(1-best_p)))/3
    soft_target_nll = -(float(best_p)*log(float(best_p))
                        +float(1-best_p)*log(float(1-best_p)))
    assert isclose(one_hot_nll, soft_target_nll)
    assert isclose(one_hot_nll, scope["conditional_nll"])
    assert isclose(scope["conditional_likelihood"], float(conditional(best_p)))
    for candidate in (Q(1, 10), Q(1, 2), Q(9, 10)):
        assert conditional(candidate) < conditional(best_p)

    values, weights = [Q(0), Q(1), Q(3)], [Q(1, 2), Q(1, 4), Q(1, 4)]
    mean = sum(x*p for x, p in zip(values, weights))
    variance = sum((x-mean)**2*p for x, p in zip(values, weights))
    assert mean == 1 and variance == Q(3, 2)
    assert sum(x*x*p for x, p in zip(values, weights))-mean**2 == variance
    assert sum([0, 0, 1, 3])/4 == 1
    assert sum([0, 1, 3, 3])/4 == 1.75
    # Enumerate all 3^4 value tuples with their probabilities; no Monte Carlo.
    mean_of_means = var_of_means = total = Q(0)
    for indices in product(range(3), repeat=4):
        probability = Q(1)
        for i in indices:
            probability *= weights[i]
        batch_mean = sum(values[i] for i in indices)/4
        total += probability
        mean_of_means += probability*batch_mean
        var_of_means += probability*(batch_mean-mean)**2
    assert total == 1 and mean_of_means == mean
    assert var_of_means == variance/4 == Q(3, 8)
    # Perfect duplicates: n variances and n*(n-1)/2 covariance terms.
    for n in (1, 4, 16):
        assert (n*variance+2*comb(n, 2)*variance)/(n*n) == variance
    for n in (4, 16):
        assert Q(1, 4)/n == (Q(1, 2)*(1-Q(1, 2)))/n

    figures = re.findall(r"!\[\[.*?/([^/]+)\]\]", text)
    assert figures == ["mml-discrete-continuous-density.png", "d2l-likelihood-coin.svg",
                       "d2l-coin-frequency-convergence.svg"]
    print("PASS: published code, probabilities, Bayes, density, conditional MLE/NLL, repeated-context floor, exact batch moments")


if __name__ == "__main__":
    main()
