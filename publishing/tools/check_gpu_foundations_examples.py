"""CPU reference checks for the actual GPU-foundations chapter; no GPU benchmark."""
from collections import defaultdict
from pathlib import Path
import hashlib
import re
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md'
body = CHAPTER.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', body, re.S)
assert len(blocks) == 2
namespace = {}
exec(compile(blocks[0], str(CHAPTER), 'exec'), namespace)
tiled = namespace['tiled_matmul']
device_namespace = {}
exec(compile(blocks[1], str(CHAPTER) + ':device-placement', 'exec'), device_namespace)
assert device_namespace['source_cpu'].device.type == 'cpu'
torch.testing.assert_close(device_namespace['source_cpu'],
                           torch.arange(6., dtype=torch.float32).reshape(2, 3))
assert device_namespace['same_cpu'] is device_namespace['source_cpu']
assert device_namespace['independent_cpu'][0, 0].item() == -1.
assert device_namespace['source_cpu'][0, 0].item() == 0.
assert device_namespace['created'].device == device_namespace['target_device']
assert device_namespace['explicit_cpu'].device.type == 'cpu'
print('PASS actual device-placement block; CUDA executed=' +
      str(device_namespace['target_device'].type == 'cuda'))

# Independent scalar product, including incomplete tiles on all three axes.
def reference(A, B):
    return np.array([[sum(A[i, k] * B[k, j] for k in range(A.shape[1]))
                      for j in range(B.shape[1])] for i in range(A.shape[0])])

checked = 0
rng = np.random.default_rng(23)
for m, k, n in [(1, 1, 1), (2, 3, 2), (3, 5, 7), (7, 6, 5), (8, 8, 8)]:
    A, B = rng.normal(size=(m, k)), rng.normal(size=(k, n))
    for size in [1, 2, 3, 4, 8]:
        np.testing.assert_allclose(tiled(A, B, size), reference(A, B), rtol=1e-13, atol=1e-13)
        checked += 1
A = np.array([[1., 2., 3.], [4., 5., 6.]])
B = np.array([[7., 8.], [9., 10.], [11., 12.]])
np.testing.assert_array_equal(A[:, :2] @ B[:2], [[25, 28], [73, 82]])
np.testing.assert_array_equal(A[:, 2:] @ B[2:], [[33, 36], [66, 72]])
assert not np.array_equal(A[:, :2] @ B[:2], reference(A, B))

# Count sectors by covered byte, not by lane order or claimed HBM traffic.
def sectors(offset, stride):
    return {byte // 32 for lane in range(32)
            for byte in range(offset + 4 * stride * lane, offset + 4 * stride * lane + 4)}
assert [len(sectors(0, s)) for s in [1, 2, 4, 8, 32]] == [4, 8, 16, 32, 32]
assert len(sectors(4, 1)) == 5
assert [len(sectors(4 * offset, 1)) for offset in range(8)] == [4] + [5] * 7

# Same-word reads broadcast. Different words in one bank serialize.
def bank_degree(words, banks=32):
    requests = defaultdict(set)
    for word in words:
        requests[word % banks].add(word)
    return max(map(len, requests.values()))
assert bank_degree(range(32)) == 1
assert bank_degree([0] * 32) == 1
assert bank_degree([32 * r for r in range(32)]) == 32
assert bank_degree([33 * r for r in range(32)]) == 1
for side in [4, 32]:
    for row in range(side):
        assert len({col ^ row for col in range(side)}) == side
    for col in range(side):
        assert len({col ^ row for row in range(side)}) == side
    assert bank_degree([side * r + (r ^ r) for r in range(side)], side) == side
    data = np.arange(side * side).reshape(side, side)
    stored = np.empty_like(data)
    for r in range(side):
        for c in range(side):
            stored[r, c ^ r] = data[r, c]
    assert all(stored[r, c ^ r] == data[r, c] for r in range(side) for c in range(side))
    assert not np.array_equal(stored, data)  # Reading without the map is wrong.

assert 4 * 256 - 1000 == 24 and 2 * 256 + 7 == 519
assert (65536 // (128 * 160)) * 4 / 64 == .1875
assert (65536 // (128 * 64)) * 4 / 64 == .5
assert 32 * 4 == 128 and 128 / 4096 == .03125
assert (2 * 16**3, 2 * 16**2 * 4) == (8192, 2048)
assert 128 * 32 * 4 == 16 * 1024
assert 24 * 2**20 == 3 * 8 * 2**20

# Event dependencies: one copy engine, one compute engine, two input buffers.
copy_end = compute_end = 0
free = [0, 0]
schedule = []
for batch in range(3):
    slot = batch % 2
    copy_start = max(copy_end, free[slot])
    copy_end = copy_start + 3
    compute_start = max(copy_end, compute_end)
    compute_end = compute_start + 7
    free[slot] = compute_end
    schedule.append((copy_start, copy_end, compute_start, compute_end))
assert schedule == [(0, 3, 3, 10), (3, 6, 10, 17), (10, 13, 17, 24)]
assert 6 < schedule[0][3]  # Copying batch 3 at t=6 would overwrite a live input.
print(f'PASS actual NumPy block, {checked} tiled products, sector/bank/XOR checks, '
      'occupancy, traffic and two-buffer dependencies; CPU only')
print('chapter_sha256=' + hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
