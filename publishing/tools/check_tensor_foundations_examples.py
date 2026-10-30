"""Execute the published NumPy example and check the chapter's worked cases."""
from fractions import Fraction
import hashlib
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md"


def main():
    text = CHAPTER.read_text()
    # A wrapped numeric sentence must not become a new Markdown list.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^1\.\s", line):
            assert index == 0 or not lines[index-1].strip(), (index+1, line)
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 2, len(blocks)
    ns = {"__name__": "__chapter_example__"}
    for number, block in enumerate(blocks, 1):
        exec(compile(block, f"{CHAPTER}:fence-{number}", "exec"), ns)
    x, w = ns["X"], ns["W"]

    # Independent row-by-column reconstruction, not an expected-value test only.
    out = [[sum(Fraction(str(a))*Fraction(str(b))
                for a, b in zip(row, column))
            for column in zip(*w.tolist())] for row in x.tolist()]
    assert out == [[5, -2], [2, 5]]
    assert sum(sum(row) for row in x.tolist()) == 7
    assert Fraction(7, x.size) == Fraction(7, 6)
    assert [1+3, 2-1] == [4, 1]  # coefficients of (1+2t)+(3-t)
    for t_value in (-3, 0, 2):
        assert (1+2*t_value)+(3-t_value) == 4+t_value
    assert np.allclose(x.mean(axis=0), [.5, 1.5, 1.5])
    assert np.allclose(x.mean(axis=1), [1, 4/3])
    assert np.array_equal(x*x, [[1, 4, 0], [0, 1, 9]])
    assert np.array_equal(x+x, 2*x)

    for operation in (lambda: x+[10, 20], lambda: x*x.T,
                      lambda: x.reshape(2, 4), lambda: x@ns["K"]):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("The chapter's incompatible shapes were accepted")

    assert np.array_equal(np.array([[0], [1], [2]]) + [[0, 1]],
                          [[0, 1], [1, 2], [2, 3]])
    assert np.array_equal(ns["E"][ns["token_ids"]],
                          [[[1, 0, 0], [0, 1, 0]],
                           [[1, 1, -1], [0, 1, 0]]])
    for dtype, expected in ((np.float32, 96), (np.float64, 192)):
        assert np.zeros((4, 2, 3), dtype=dtype).nbytes == expected
    dependent_rows = np.array([[1, 2], [2, 4]])
    assert dependent_rows.ndim == 2
    assert np.linalg.matrix_rank(dependent_rows) == 1
    for y, dot, cosine in (([2, 0], 2, 1), ([20, 0], 20, 1), ([0, 1], 0, 0)):
        a = np.array([1., 0.])
        b = np.array(y)
        assert a@b == dot
        assert (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)) == cosine
    assert np.linalg.norm([3, 4], ord=1) == 7
    assert np.linalg.norm([0, 0]) == 0
    assert np.isclose(np.linalg.norm(np.array([3., 4.])/5), 1)

    # Verify the head split/permutation independently of equal-size axes.
    b, t, h, dh = 2, 3, 2, 2
    original = np.arange(b*t*h*dh).reshape(b, t, h*dh)
    heads = original.reshape(b, t, h, dh).transpose(0, 2, 1, 3)
    assert heads.shape == (b, h, t, dh)
    for bi in range(b):
        for hi in range(h):
            for ti in range(t):
                assert np.array_equal(heads[bi, hi, ti],
                                      original[bi, ti, hi*dh:(hi+1)*dh])
    assert np.array_equal(heads.transpose(0, 2, 1, 3).reshape(b, t, h*dh), original)
    assert not np.array_equal(heads, original.reshape(b, h, t, dh))

    # Independent contraction: no other batch/head may contribute to a score.
    for batch, n_heads, queries, keys, width in (
        (2, 3, 1, 4, 5), (3, 2, 4, 1, 3), (2, 2, 3, 4, 2),
    ):
        q = np.arange(batch*n_heads*queries*width).reshape(
            batch, n_heads, queries, width)
        k = (1+np.arange(batch*n_heads*keys*width)).reshape(
            batch, n_heads, keys, width)
        got = q @ k.swapaxes(-1, -2)
        reference = np.empty((batch, n_heads, queries, keys), dtype=np.int64)
        for bi in range(batch):
            for hi in range(n_heads):
                for qi in range(queries):
                    for ki in range(keys):
                        reference[bi, hi, qi, ki] = sum(
                            int(q[bi, hi, qi, j])*int(k[bi, hi, ki, j])
                            for j in range(width))
        assert np.array_equal(got, reference)

    # B == H conceals a semantic broadcasting error: [B,n,p] binds to H.
    matrices = np.arange(2*2*3*4).reshape(2, 2, 3, 4)
    per_example = (1+np.arange(2*4*3)).reshape(2, 4, 3)
    correct = matrices @ per_example[:, None]
    misleading = matrices @ per_example
    assert correct.shape == misleading.shape == (2, 2, 3, 3)
    assert not np.array_equal(correct, misleading)
    for bi in range(2):
        for hi in range(2):
            assert np.array_equal(correct[bi, hi],
                                  matrices[bi, hi] @ per_example[bi])
            assert np.array_equal(misleading[bi, hi],
                                  matrices[bi, hi] @ per_example[hi])

    # Round-trip heads: verify every element, including unequal axis lengths.
    assert np.array_equal(ns["restored"], ns["projected"])
    assert np.array_equal(ns["heads"][0, 0], [[0, 1], [4, 5], [8, 9]])
    assert np.array_equal(ns["heads"][0, 1], [[2, 3], [6, 7], [10, 11]])
    assert np.array_equal(ns["scores"][:, 0, 0], [[1, 4], [20, 80]])
    assert np.array_equal(ns["X_batch"] @ ns["W_batch"],
                          [[[5, -2], [2, 5]], [[100, -40], [40, 100]]])

    # The public source's blanket "reshape is O(1)" claim is not adopted.
    array = np.arange(12).reshape(3, 4)
    assert np.shares_memory(array, array.reshape(2, 6))
    assert not np.shares_memory(array, array.T.reshape(2, 6))
    # Each dtype uses a different byte stride but the same element coordinates.
    # Check all addresses against actual buffer bytes, independent of ravel.
    address_cases = 0
    for dtype in (np.int8, np.int16, np.float32, np.float64):
        for rows in range(1, 6):
            for columns in range(1, 7):
                values = np.arange(1, rows*columns+1).reshape(rows, columns)
                for order in ("C", "F"):
                    arr = np.array(values, dtype=dtype, order=order)
                    physical = np.frombuffer(arr.tobytes(order="A"), dtype=dtype)
                    for i in range(rows):
                        for j in range(columns):
                            offset = i*arr.strides[0] + j*arr.strides[1]
                            assert physical[offset//arr.itemsize] == values[i, j]
                            address_cases += 1
                    assert np.array_equal(arr.T, values.T)
                    assert np.shares_memory(arr, arr.T)
                    sliced = arr[:, ::2]
                    assert np.array_equal(sliced, values[:, ::2])
                    assert np.shares_memory(arr, sliced)
    assert ns["layout_c"][0, 0] == 1
    assert ns["layout_slice"].nbytes == 16 and ns["span"] == 24
    assert ns["layout_copy"][0, 0] == 1
    print(f"SHA-256: {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"PASS: {len(blocks)} actual NumPy fences, {address_cases} independent "
          "byte-address cases; explicit matmul, lookup, broadcasting, "
          "reductions, incompatible-shape errors, norms, ndarray/matrix ranks, "
          "head permutation, rectangular batched scores, semantic "
          "broadcasting counterexample and reshape copy counterexample. "
          "The five upstream NumPy figure cases also execute in the chapter fence. "
          "No Transformer training or upstream TinyTorch suite executed.")


if __name__ == "__main__":
    main()
