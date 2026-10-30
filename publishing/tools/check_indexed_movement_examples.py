"""Execute S8 indexed-movement fences and independent CPU checks.

Run: uv run --offline --no-project --with torch==2.14.1 --with numpy python
publishing/tools/check_indexed_movement_examples.py
No GPU, distributed process group, ONNX Runtime or benchmark is exercised.
"""
import ast
import hashlib
import itertools
from pathlib import Path
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md'
originals = ROOT / '05 Источники/Originals/indexed-movement-2026-10-27'
text = chapter.read_text()
section = text.split('<!-- BEGIN REVIEW: indexed movement 2026-10-27 -->')[1].split(
    '<!-- END REVIEW: indexed movement 2026-10-27 -->')[0]
blocks = re.findall(r'```python\n(.*?)\n```', section, re.S)
assert len(blocks) == 4
scopes = []
for number, code in enumerate(blocks, 1):
    scope = {}
    exec(compile(code, f'{chapter}:indexed-fence-{number}', 'exec'), scope)
    scopes.append(scope)
assert scopes[1]['out'].tolist() == [8, 7, 5, 4, 0]
assert scopes[2]['y'].tolist() == [30., 10., 30., 40.]
assert scopes[2]['ds'].tolist() == [3., 1., 3., 4.]


def reference_helper(filename, name):
    # Only these complete, reviewed functions are needed from ONNX source.
    tree = ast.parse((originals / filename).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == name]
    assert len(selected) == 1
    module = ast.Module(body=selected, type_ignores=[])
    scope = {'np': np}
    exec(compile(module, str(originals / filename), 'exec'), scope)
    return scope[name]


onnx_gather = reference_helper('onnx-gather-elements.py', 'gather_numpy')
onnx_scatter = reference_helper('onnx-scatter-elements.py', 'scatter_elements')
generator = torch.Generator().manual_seed(27)
cases = 0
for rank in (1, 2, 3):
    shape = tuple(range(3, 3 + rank))
    for axis in range(rank):
        for extent in (1, 2, 5):
            out_shape = tuple(extent if d == axis else max(1, size - 1)
                              for d, size in enumerate(shape))
            index = torch.randint(shape[axis], out_shape, generator=generator)
            data = torch.randn(shape, generator=generator, dtype=torch.float64,
                               requires_grad=True)
            gathered = data.gather(axis, index)
            manual = torch.empty(out_shape, dtype=data.dtype)
            weights = torch.randn(out_shape, generator=generator, dtype=data.dtype)
            expected_grad = torch.zeros_like(data)
            for coordinate in itertools.product(*(range(size) for size in out_shape)):
                source = list(coordinate)
                source[axis] = int(index[coordinate])
                manual[coordinate] = data.detach()[tuple(source)]
                expected_grad[tuple(source)] += weights[coordinate]
            torch.testing.assert_close(gathered, manual)
            dx, = torch.autograd.grad((gathered * weights).sum(), (data,))
            torch.testing.assert_close(dx, expected_grad)
            np.testing.assert_allclose(onnx_gather(data.detach().numpy(), axis, index.numpy()),
                                       manual.numpy())
            np.testing.assert_allclose(onnx_scatter(np.zeros(shape), index.numpy(),
                                                   weights.numpy(), axis, 'add'),
                                       expected_grad.numpy())
            # Fixed-index gather and zero-initialized scatter-add are adjoints.
            torch.testing.assert_close((manual * weights).sum(),
                                       (data.detach() * expected_grad).sum())
            cases += 1

# ONNX Gather selects slices; GatherElements selects one value per index.
data = np.arange(1, 7).reshape(3, 2)
index = np.array([[0, 1], [1, 2]])
np.testing.assert_array_equal(np.take(data, index, axis=0),
                             [[[1, 2], [3, 4]], [[3, 4], [5, 6]]])
assert np.take(data, index, axis=0).shape == (2, 2, 2)
np.testing.assert_array_equal(onnx_gather(data, 0, index), [[1, 4], [3, 6]])

# The reference ONNX helper accepts negative indices; torch.gather rejects them.
negative = np.array([[-1, 0], [0, -1]])
np.testing.assert_array_equal(onnx_gather(data, 0, negative), [[5, 2], [1, 6]])
try:
    torch.tensor(data).gather(0, torch.tensor(negative))
except RuntimeError:
    pass
else:
    raise AssertionError('negative PyTorch gather indices were accepted')

for invalid in (torch.tensor([0, 1]), torch.zeros((4, 2), dtype=torch.int64)):
    try:
        torch.zeros(3, 2).gather(1, invalid)
    except RuntimeError:
        pass
    else:
        raise AssertionError('invalid rank/non-axis shape was accepted')
assert torch.empty(3, 2).gather(0, torch.empty((0, 2), dtype=torch.int64)).shape == (0, 2)
empty = torch.zeros(4).scatter_add_(0, torch.empty(0, dtype=torch.int64), torch.empty(0))
assert empty.tolist() == [0., 0., 0., 0.]

# Independent numerical gradient, omitted coordinate, multiplicity and inverse.
index = torch.tensor([2, 0, 2, 3])
x = torch.tensor([10., 20., 30., 40.], dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(lambda value: value.gather(0, index), (x,))
updates = torch.randn(4, dtype=x.dtype, requires_grad=True)
base = torch.randn(4, dtype=x.dtype, requires_grad=True)
assert torch.autograd.gradcheck(lambda b, s: b.scatter_add(0, index, s), (base, updates))
permutation = torch.tensor([2, 0, 3, 1])
torch.testing.assert_close(torch.zeros_like(x).scatter_add_(0, permutation,
    x.gather(0, permutation)), x)
changed = x.detach().clone()
changed[1] = -999
torch.testing.assert_close(changed.gather(0, index), x.gather(0, index))
counts = torch.zeros(4, dtype=x.dtype).scatter_add_(0, index, torch.ones(4, dtype=x.dtype))
torch.testing.assert_close(counts * x, torch.zeros_like(x).scatter_add_(0, index,
    x.gather(0, index)))

# CPU interleaving counterexample, not a simulation of GPU atomic hardware.
read_a = read_b = 0
write_a, write_b = read_a + 5, read_b + 1
assert write_a != 6 and write_b != 6
sum_a = (np.float32(1e20) + np.float32(-1e20)) + np.float32(1.)
sum_b = np.float32(1e20) + (np.float32(-1e20) + np.float32(1.))
assert sum_a == 1 and sum_b == 0
units = scopes[3]['fp32_access_units']
assert len(units(range(32), base_bytes=4)) == 5
assert len(units([32*k for k in range(32)]))*32 == 1024

# Original TeX inputs reproduce the complete native diagram.
tex = (originals / 'scatter-add.tex').read_text()
arrays = [list(map(int, re.search(r'\\def\\' + name + r'\{\{([^}]+)\}\}', tex)
                        .group(1).split(','))) for name in ('indices', 'inputs', 'outputs')]
reproduced = torch.zeros(4, dtype=torch.int64).scatter_add_(0,
    torch.tensor(arrays[0]), torch.tensor(arrays[1])).tolist()
assert reproduced == arrays[2] == [8, 7, 5, 4]
print('PASS four actual printed CPU fences; torch=' + torch.__version__)
print(f'PASS {cases} rank/axis/shape cases: scalar-loop gather/gradient, adjoint identity, ONNX reference helpers')
print('PASS Gather vs GatherElements shapes, negative-index distinction, invalid shapes, empty input, both gradchecks')
print('PASS missing/repeated/permuted coordinates, lost-update and FP32-order counterexamples, access-unit counts')
print('PASS original SVG numerical data from complete TeX source; no GPU/distributed/ONNX Runtime execution or timing')
print('chapter_sha256=' + hashlib.sha256(chapter.read_bytes()).hexdigest())
