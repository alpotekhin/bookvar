"""Verify the published derivative examples without claiming network training."""
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/02 Производная и градиент.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 3, len(blocks)
    scope = {"__name__": "__chapter_example__"}
    for number, code in enumerate(blocks, 1):
        exec(compile(code, f"{CHAPTER}:block-{number}", "exec"), scope)
    numerical_grad = scope["numerical_grad"]
    numerical_gradient = scope["numerical_gradient"]

    # Exact rational checks, independent of finite-difference approximations.
    loss = lambda w: (w-3)**2
    assert loss(1) == 4
    assert loss(Q(101, 100)) == Q(39601, 10000)
    x, h = Q(3), Q(1, 100)
    delta = (x+h)**2-x*x
    assert delta == Q(601, 10000)
    assert delta-2*x*h == h*h
    w = Q(1)
    trace = []
    for _ in range(3):
        grad = 2*(w-3)
        new = w-Q(1, 4)*grad
        trace.append((w, loss(w), grad, new))
        w = new
    assert trace == [(1, 4, -4, 2), (2, 1, -2, Q(5, 2)),
                     (Q(5, 2), Q(1, 4), -1, Q(11, 4))]

    f = lambda x, y: x*x+3*x*y+y*y
    h = Q(1, 100)
    assert f(1+h, 2)-f(1, 2) == 8*h+h*h == Q(801, 10000)
    assert f(1, 2+h)-f(1, 2) == 7*h+h*h == Q(701, 10000)
    new_x, new_y = 1-h*8, 2-h*7
    assert (new_x, new_y) == (Q(92, 100), Q(193, 100))
    assert f(new_x, new_y) == Q(98981, 10000)
    assert f(new_x, new_y)-11 == Q(-11019, 10000)
    assert f(new_x, new_y)-11-(-h*113) == Q(281, 10000)
    assert f(1-8, 2-7) == f(-7, -5) == 179
    assert np.allclose(numerical_gradient(lambda v:f(*v), [1., 2.]), [8, 7])
    gradient = np.array([8., 7.])
    aligned = gradient/np.linalg.norm(gradient)
    assert np.isclose(gradient@aligned, np.sqrt(113))
    direction = np.array([.6, .8])
    assert np.isclose(np.linalg.norm(direction), 1)
    directional = numerical_grad(lambda t:f(*(np.array([1.,2.])+t*direction)), 0.)
    assert np.isclose(directional, 10.4)

    assert np.isclose(numerical_grad(lambda x:3*x*x+1, 2.), 12)
    assert np.isclose(numerical_grad(lambda x:3*x*x+x, 2.), 13)
    for v in [-h, h]:
        assert v*v > 0 and -v*v < 0
    assert (-h)**3 < 0 < h**3

    forward = lambda v:np.array([v[0]+v[1], v[0]*v[1]])
    point = np.array([1., 2.])
    jacobian = np.stack([numerical_gradient(lambda v:forward(v)[i], point)
                         for i in range(2)])
    assert np.allclose(jacobian, [[1, 1], [2, 1]])
    dx = np.array([.01, -.02])
    assert np.allclose(jacobian@dx, [-.01, 0])
    assert np.allclose(forward(point+dx), [2.99, 1.9998])
    assert np.allclose(jacobian.T@np.array([1.,2.]), [5,3])
    assert np.allclose(numerical_gradient(lambda v:forward(v)@np.array([1.,2.]), point),
                       [5,3])

    # Rectangular Jacobian from the preceding chapter's shared linear map.
    linear_w = np.array([[1., 0.], [2., -1.], [0., 2.]])
    input_point = np.array([1., 2., 0.])
    numeric_j = np.stack([
        numerical_gradient(lambda v: (linear_w.T @ v)[i], input_point)
        for i in range(2)])
    assert numeric_j.shape == (2, 3)
    assert np.allclose(numeric_j, linear_w.T, rtol=1e-8, atol=1e-8)
    for upstream, expected in (([1, -2], [1, 4, -4]),
                               ([1, 1], [1, 1, 2])):
        gradient = linear_w @ upstream
        assert np.array_equal(gradient, expected)
        # Expand the scalar independently rather than reuse W @ upstream.
        a, b = upstream
        scalar = lambda v: a*(v[0]+2*v[1])+b*(-v[1]+2*v[2])
        assert np.allclose(numerical_gradient(scalar, input_point), expected,
                           rtol=1e-8, atol=1e-8)
    assert np.array_equal(scope["backward"], [1, 4, -4])

    # One specified output entry, with unrelated entries set to zero for this check.
    def matrix_output(v):
        a = np.zeros((4, 2))
        a[2, 1] = v[1]+v[2]
        return a
    point3 = np.array([1., 2., 3.])
    tensor = np.stack([np.stack([
        numerical_gradient(lambda v:matrix_output(v)[i,j], point3)
        for j in range(2)]) for i in range(4)])
    flat_jacobian = np.stack([
        numerical_gradient(lambda v:matrix_output(v).ravel(order="F")[k], point3)
        for k in range(8)])
    assert tensor.shape == (4, 2, 3)
    assert np.allclose(tensor[2, 1], [0, 1, 1])
    assert np.allclose(flat_jacobian[6], tensor[2, 1])
    assert np.allclose(flat_jacobian[6], [0, 1, 1])

    assert Q(1, 2)**10 == Q(1, 1024)
    assert 2**10 == 1024
    assert Q(1, 2)*Q(1, 2)*8 == 2
    j1, j2 = np.diag([100., .01]), np.diag([.01, 100.])
    assert np.linalg.norm(j1, ord=2) == np.linalg.norm(j2, ord=2) == 100
    assert np.array_equal(j2@j1, np.eye(2))

    x, h = Q(2), Q(1, 10)
    one_sided = ((x+h)**3-x**3)/h
    central = ((x+h)**3-(x-h)**3)/(2*h)
    assert one_sided == Q(1261, 100)
    assert central == Q(1201, 100)
    assert central-3*x*x == h*h
    assert numerical_grad(abs, 0.) == 0
    assert (abs(h)-abs(0))/h == 1
    assert (abs(0)-abs(-h))/h == -1

    calls = []
    def counted(v):
        calls.append(tuple(v))
        return f(*v)
    numerical_gradient(counted, [1., 2.])
    assert len(calls) == 4
    before = (ROOT / "docs/audit/snapshots/2026-10-21-derivatives-gradients.md").read_text()
    embeds = lambda s:Counter(re.findall(r"!\[\[([^\]]+)\]\]", s))
    assert embeds(text) == embeds(before)
    for index, line in enumerate(text.splitlines()):
        if re.match(r"^1\.\s", line):
            assert not text.splitlines()[index-1].strip(), (index+1, line)
    print("PASS: 3 sequential published snippets; rational derivative/step checks; "
          "branch accumulation; rectangular linear-map VJP and matrix-output axes; vanishing/amplification "
          "counterexamples; finite differences and nondifferentiable point; 5 figure embeds. "
          "No neural-network training or upstream autograd suite executed.")


if __name__ == "__main__":
    main()
