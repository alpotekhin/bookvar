"""Execute the published block and check the chapter's finite examples."""
from fractions import Fraction as Q
from itertools import product
from math import comb, erf, exp, expm1, isclose, log, log1p, pi, sqrt
from hashlib import sha256
from pathlib import Path
import re
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 1
    scope = {"__name__": "__chapter_example__"}
    exec(compile(blocks[0], str(CHAPTER), "exec"), scope)

    masses = [Q(50, 100), Q(30, 100), Q(15, 100), Q(5, 100)]
    assert sum(masses) == 1 and sum(masses[:2]) == Q(4, 5)
    density = 1 / (Q(16, 10)-Q(9, 10))
    assert density == Q(10, 7) > 1
    assert density*Q(7, 10) == 1
    assert density*(Q(12, 10)-1) == Q(2, 7)

    # CDF of the same uniform distribution, including both support boundaries.
    cdf = lambda x: max(Q(0), min(Q(1), (x-Q(9, 10))/Q(7, 10)))
    assert cdf(Q(9, 10)) == 0 and cdf(Q(16, 10)) == 1
    assert cdf(Q(12, 10))-cdf(Q(1)) == Q(2, 7)
    grid = [Q(k, 10) for k in range(-10, 30)]
    assert all(cdf(a) <= cdf(b) for a, b in zip(grid, grid[1:]))
    # Integrate p(x,y)=4xy over y exactly, then normalize its x marginal.
    integrate = lambda coeff, hi=Q(1): sum(Q(c, k+1)*hi**(k+1)
                                           for k, c in enumerate(coeff))
    for x in (Q(0), Q(1, 4), Q(1, 2), Q(1)):
        assert integrate([0, 4*x]) == 2*x
    assert integrate([0, 2]) == 1 and integrate([0, 2], Q(1, 2)) == Q(1, 4)
    # Original D2L shaded-density source: left grid sum versus analytic CDF.
    normal_pdf = lambda x: exp(-x*x/2)/sqrt(2*pi)
    normal_cdf = lambda x: (1+erf(x/sqrt(2)))/2
    mixed_pdf = lambda x: 0.2*normal_pdf(x-3)+0.8*normal_pdf(x+1)
    shaded_grid = sum(0.01*mixed_pdf(-2+i*0.01) for i in range(500))
    shaded_exact = (0.2*(normal_cdf(0)-normal_cdf(-5))
                    +0.8*(normal_cdf(4)-normal_cdf(-1)))
    assert isclose(shaded_grid, 0.773617148399353, abs_tol=5e-7)
    assert abs(shaded_grid-shaded_exact) < 0.002

    # Density MLE versus an exact finite measurement event, D2L Exercise 1.
    density_alpha = 1/3
    density_likelihood = lambda a: a*exp(-3*a)
    assert isclose(3-1/density_alpha, 0, abs_tol=1e-14)
    event_likelihood = lambda a, width: exp(-3*a)*(-expm1(-a*width))
    event_alpha = log(4/3)
    assert isclose(event_alpha, 0.28768207245178085)
    assert isclose(-3+1/expm1(event_alpha), 0, abs_tol=1e-13)
    assert event_likelihood(event_alpha, 1) > event_likelihood(density_alpha, 1)
    assert density_likelihood(density_alpha) > density_likelihood(event_alpha)
    for width in (1, 0.1, 0.01, 0.0001):
        alpha = log1p(width/3)/width
        assert isclose(-3+width/expm1(alpha*width), 0, abs_tol=1e-12)
        assert 0 < alpha < density_alpha
    assert abs(log1p(1e-6/3)/1e-6-density_alpha) < 1e-7

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

    sequence = masses[0]*Q(3, 5)*Q(1, 4)
    assert sequence == Q(3, 40)
    nll = -log(float(sequence))
    assert isclose(nll, 2.5902671654458267)
    assert isclose(nll/3, 0.8634223884819422)
    assert masses[0]*Q(3, 5)*Q(1, 2) == Q(3, 20)
    answer_given_cat = Q(3, 5)*Q(1, 4)
    assert answer_given_cat == sequence/masses[0] == Q(3, 20)
    assert isclose(-log(float(answer_given_cat))/2, scope["answer_nll"])
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

    figures = re.findall(r"!\[\[[^\]|]+/([^/\]|]+)(?:\|[^\]]+)?\]\]", text)
    assert figures == ["d2l-likelihood-coin.svg", "d2l-density-interval.svg", "d2l-marginal.svg",
                       "d2l-coin-frequency-convergence.svg"]
    prior = before_topic_consolidation(text, CHAPTER,
        stop_before_journal="2026-10-30-derivative-native-additions.json")
    native = before_topic_consolidation(text, CHAPTER,
        stop_before_journal="2026-10-30-probability-native-additions.json")
    assert re.findall(r"^```python\n(.*?)^```\s*$", prior, re.M | re.S) == re.findall(
        r"^```python\n(.*?)^```\s*$", native, re.M | re.S)
    assert "mml-discrete-continuous-density.png" in prior
    print("PASS: published code, Bayes, density/CDF, continuous marginal, finite-window MLE, NLL and exact batch moments")
    print("chapter_sha256=" + sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
