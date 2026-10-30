"""Execute the chapter examples and check the loss derivations numerically."""
from fractions import Fraction as Q
from itertools import combinations_with_replacement
from hashlib import sha256
from math import exp, isclose, log, log1p, pi
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/04 Функции потерь.md"


def central_gradient(f, z, h=1e-5):
    result = np.zeros_like(z, dtype=float)
    for k in range(z.size):
        delta = np.zeros_like(z, dtype=float)
        delta.flat[k] = h
        result.flat[k] = (f(z+delta)-f(z-delta))/(2*h)
    return result


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 5
    scope = {"__name__": "__chapter_example__"}
    for i, block in enumerate(blocks):
        exec(compile(block, str(CHAPTER)+":block-"+str(i+1), "exec"), scope)
    log_softmax = scope["log_softmax"]
    z = np.array([3.2, 5.1, -1.7])
    p = np.exp(log_softmax(z))
    assert np.allclose(np.exp(z), [24.532530197, 164.0219073, .182683524])
    assert isclose(np.exp(z).sum(), 188.737121020972)
    assert np.allclose(p, [.12998254, .86904954, .00096793])
    assert p.argmax() == z.argmax() == 1
    assert isclose(p[1]/p[0], exp(1.9))
    q_hard = np.array([1., 0., 0.])
    q_soft = np.array([.9, .05, .05])
    for q in (q_hard, q_soft, 2*q_soft):
        loss = lambda scores: -np.dot(q, log_softmax(scores))
        assert np.allclose(central_gradient(loss, z), q.sum()*p-q, atol=1e-8)
    g = p-q_hard
    updated = z-.1*g
    assert updated.argmax() == 1
    assert np.allclose(updated, [3.287001746, 5.013095046, -1.700096793])
    assert isclose(-log_softmax(updated)[0], 1.8909237270947268)
    assert -log_softmax(updated)[0] < -log_softmax(z)[0]
    assert np.allclose(np.outer(g, [2., -1.])[0], [-1.7400349256, .8700174628])

    # Full parameter step; differences do not reuse the analytic outer product.
    W = np.array([[1., -1.2], [2., -1.1], [-.5, .7]])
    x, b = np.array([2., -1.]), np.zeros(3)
    scalar_loss = lambda weights, biases, inputs: -log_softmax(weights@inputs+biases)[0]
    assert np.allclose(central_gradient(lambda weights: scalar_loss(weights, b, x), W),
                       scope["dW"], rtol=1e-7, atol=1e-8)
    assert np.allclose(central_gradient(lambda biases: scalar_loss(W, biases, x), b),
                       scope["db"], rtol=1e-7, atol=1e-8)
    assert np.allclose(central_gradient(lambda inputs: scalar_loss(W, b, inputs), x),
                       scope["dx"], rtol=1e-7, atol=1e-8)
    assert isclose(scope["loss_new"], 1.21178042875392)
    assert np.allclose(scope["z_new"], z-.6*g)
    assert scope["loss_new"] < -log_softmax(updated)[0]
    other_x = np.array([-1., .5])
    assert not np.allclose(scope["W_new"]@other_x+scope["b_new"], W@other_x+b)
    # The empirical soft target has the same objective as three one-hot labels.
    predicted = np.array([.7, .3])
    empirical_q = np.array([2/3, 1/3])
    assert isclose(-(2*log(predicted[0])+log(predicted[1]))/3,
                   -np.dot(empirical_q, np.log(predicted)))
    assert isclose(-log_softmax(p)[0], 1.3795068861455881)
    assert np.allclose(.85*q_hard+.15/3, q_soft)
    for invalid in ([np.nan, 0.], [np.inf, 0.]):
        try:
            log_softmax(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite logits accepted")
    with np.errstate(over="ignore", divide="ignore"):
        assert np.isinf(np.exp(1000.))
        assert np.exp(log_softmax([1000., 0.]))[1] == 0
    assert isclose(-log(.8), .2231435513142097)
    assert isclose(-log(.1), 2.3025850929940455)

    q, prediction = np.array([.6, .4]), np.array([.8, .2])
    entropy = -np.dot(q, np.log(q))
    kl = np.dot(q, np.log(q/prediction))
    ce = -np.dot(q, np.log(prediction))
    assert isclose(entropy, .6730116670092565)
    assert isclose(kl, .10464962875290948)
    assert isclose(ce, .777661295762166)
    assert isclose(ce, entropy+kl)
    assert not isclose(kl, np.dot(prediction, np.log(prediction/q)))

    mask = scope["mask"]
    per_token = scope["per_token"]
    masked_mean = lambda values: values.reshape(2, 3)[mask].mean()
    expected = mask.astype(float)/3
    assert np.allclose(central_gradient(masked_mean, per_token), expected)
    assert isclose(scope["token_mean"], .6)
    assert isclose(scope["per_example"].mean(), .75)
    assert not isclose(per_token.mean(), .6)

    errors = np.array([1., -1., 3.])
    assert isclose(errors.mean(), 1)
    assert isclose(np.mean(errors**2), 11/3)
    assert isclose(np.mean(np.abs(errors)), 5/3)
    assert np.allclose(central_gradient(lambda x: np.mean(x*x), errors), 2*errors/3)
    assert isclose(.5*log(2*pi)+9/2, 5.418938533204672)
    for sigma2 in (.5, 1., 3.):
        nll = .5*log(2*pi*sigma2)+errors**2/(2*sigma2)
        density = np.exp(-errors**2/(2*sigma2))/np.sqrt(2*pi*sigma2)
        assert np.allclose(nll, -np.log(density))

    # Independent rational checks: all multisets of length 1..6 on five values.
    # Include ties and even-sample intervals, not just the printed three-point case.
    mean_median_cases = 0
    for n in range(1, 7):
        for sample in combinations_with_replacement([-3, -1, 0, 2, 7], n):
            observations = tuple(map(Q, sample))
            mean = sum(observations) / n
            lo, hi = observations[(n-1)//2], observations[n//2]
            variance = sum((y-mean)**2 for y in observations) / n
            min_mae = sum(abs(y-lo) for y in observations) / n
            candidates = {*observations, mean, lo, hi, (lo+hi)/2, lo-1, hi+1}
            for a in candidates:
                mse = sum((y-a)**2 for y in observations) / n
                mae = sum(abs(y-a) for y in observations) / n
                assert mse == variance + (a-mean)**2
                assert mae >= min_mae
                assert (mae == min_mae) == (lo <= a <= hi)
            mean_median_cases += 1
    assert mean_median_cases == 461
    assert np.allclose(scope["mse_values"], [65/3, 50/3, 38/3, 113/3])
    assert np.allclose(scope["mae_values"], [3, 8/3, 10/3, 5])

    r = Q(3, 5)
    for prob in (Q(0), Q(1, 10), r, Q(4, 5), Q(1)):
        risk = r*(1-prob)**2+(1-r)*prob**2
        assert risk == (prob-r)**2+r*(1-r)
        assert risk >= Q(24, 100)
    assert r*(1-Q(4, 5))**2+(1-r)*Q(4, 5)**2 == Q(28, 100)
    prob = Q(1, 100)
    assert (prob-1)**2 == Q(9801, 10000)
    assert 2*(prob-1)*prob*(1-prob) == Q(-19602, 1000000)
    assert isclose(-log(float(prob)), 4.605170185988091)
    sigmoid = lambda score: 1/(1+exp(-score))
    bce = lambda score, y: max(score, 0)-y*score+log1p(exp(-abs(score)))
    for y in (0, 1):
        for score in (-10., 0., 10.):
            h = 1e-5
            assert isclose((bce(score+h, y)-bce(score-h, y))/(2*h),
                           sigmoid(score)-y, abs_tol=1e-8)
    assert bce(1000., 0) == bce(-1000., 1) == 1000
    assert isclose(.01*(2**2+(-1)**2), .05)
    assert np.allclose(.02*np.array([2., -1.]), [.04, -.02])
    assert len(re.findall(r"!\[\[.*?\]\]", text)) == 9
    assert "neural-language-model-2026-10-21/d2l-softmaxreg.svg" in text
    hinge = scope["multiclass_hinge"]
    assert np.allclose(central_gradient(lambda scores: hinge(scores, 0), z),
                       [-1., 1., 0.], atol=1e-8)
    assert np.isclose(hinge([3., 0., 0.], 0), 0.)
    assert np.isclose(hinge([0., 2., 3.], 0), 7.)  # sum, not max
    assert np.isclose(hinge([13., -7., 11.], 0, delta=10.), 8.)
    assert np.isclose(hinge([13., -7., 11.], 0, delta=1.), 0.)
    assert hinge([3., 0., 0.], 0) != 1.  # true class must be omitted
    assert np.allclose(scope["diagram_scores"], [-2.85, .86, .28])
    diagram_exp = np.array([exp(-2.85), exp(.86), exp(.28)])
    assert np.allclose(np.exp(log_softmax(scope["diagram_scores"])),
                       diagram_exp / diagram_exp.sum())
    assert np.allclose(scope["comparison_ce"],
                       [.0949229564, .2395447662, .5514447139])
    print("PASS: 5 published blocks; softmax/CE, full W/b/x gradients and parameter step, "
          "hinge margins/gradient, full source diagram, smoothing, masks, Gaussian MSE, "
          f"Brier, BCE; {mean_median_cases} exact mean/median cases")
    print("chapter_sha256:", sha256(CHAPTER.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
