#!/usr/bin/env python3
"""Verify the pinned 4×4 witness without executing the source notebook."""

import ast
import hashlib
import json
from itertools import product
from pathlib import Path
import sys


SHA256 = "2cce2543e48c89aa3e91614272a698a0147dd2548ea11cf92f1292b7435d38ff"
N = 4
RANK = 48


def doubled(value):
    real, imag = value.real * 2, value.imag * 2
    assert real == int(real) and imag == int(imag)
    return int(real), int(imag)


def multiply(a, b):
    x, y = a
    u, v = b
    return x * u - y * v, x * v + y * u


def load_factors(path):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SHA256, "unexpected notebook"
    cells = json.loads(payload)["cells"]
    cell = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).startswith("#@title Data\ndecomposition_444 =")
    )
    assignment, = ast.parse(cell).body
    assert isinstance(assignment, ast.Assign)
    target, = assignment.targets
    assert isinstance(target, ast.Name) and target.id == "decomposition_444"
    assert isinstance(assignment.value, ast.Tuple)
    arrays = []
    for call in assignment.value.elts:
        assert isinstance(call, ast.Call) and len(call.args) == 1
        # Read the literal only. Do not evaluate the call or execute any cell.
        rows = ast.literal_eval(call.args[0])
        assert len(rows) == N * N and all(len(row) == RANK for row in rows)
        arrays.append([[doubled(value) for value in row] for row in rows])
    assert len(arrays) == 3
    return arrays


def first_mismatch(factors):
    left, right, output_transposed = factors
    for i, j, k in product(range(N * N), repeat=3):
        a, b = divmod(i, N)
        c, d = divmod(j, N)
        col, row = divmod(k, N)
        expected = (8 if b == c and (a, d) == (row, col) else 0, 0)
        terms = (
            multiply(multiply(u, v), w)
            for u, v, w in zip(left[i], right[j], output_transposed[k], strict=True)
        )
        real, imag = map(sum, zip(*terms))
        if (real, imag) != expected:
            return i, j, k
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} mathematical_results.ipynb")
    factors = load_factors(Path(sys.argv[1]))
    mismatch = first_mismatch(factors)
    assert mismatch is None, f"tensor coefficient mismatch at {mismatch}"
    u, v = factors[0][0][0]
    factors[0][0][0] = u + 1, v
    assert first_mismatch(factors) is not None, "mutation went undetected"
    print("48 terms; 4096 tensor coefficients verified exactly; mutation rejected")
