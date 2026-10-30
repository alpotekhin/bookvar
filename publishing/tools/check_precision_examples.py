"""CPU verification of the chapter's examples; does not emulate a GPU GEMM."""
import ast
import contextlib
from fractions import Fraction as Q
from hashlib import sha256
import io
from math import isclose
from pathlib import Path
import re
import struct
import numpy as np

root = Path(__file__).resolve().parents[2]
path = root / '00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md'
body = path.read_text()
blocks = re.findall(r'(?:```|~~~)python\n(.*?)\n(?:```|~~~)', body, re.S)
assert len(blocks) == 2
for block in blocks:
    ast.parse(block)
cpu = next(block for block in blocks if 'def repeated_sum' in block)
ns = {}
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(compile(cpu, path.name, 'exec'), ns)
results = [float(line.split()[-1]) for line in output.getvalue().splitlines()]
assert results == [10.000133514404297, 9.953125, 10.00213623046875]
assert float(np.float16(0.01)) * 1000 == results[2]
assert np.float16(6.25).view(np.uint16) == int('0100011001000000', 2)
assert float(np.float16(1 + 2**-10)) == 1 + 2**-10

def bf16(x):
    bits = struct.unpack('>I', struct.pack('>f', float(x)))[0]
    assert bits & 0x7f800000 != 0x7f800000, 'finite inputs only'
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('>f', struct.pack('>I', rounded))[0]

assert bf16(1 + 2**-10) == 1
assert bf16(100000) == 99840
for dtype, expected in [(bf16, 256), (np.float16, 2048), (np.float32, 4096)]:
    s = dtype(0)
    for _ in range(4096):
        s = dtype(float(s) + 1)
    assert s == expected
assert np.float32(bf16(1 + 2**-10)) - np.float32(bf16(1)) == 0
assert np.float16(2**-26) == 0
assert np.float16(2**-25) == 0  # midpoint rounds to the even zero code
assert np.float16(3*2**-25) == 2**-23
g, scale = 2**-26, 2**16
assert np.float16(g*scale) == 2**-10
assert np.float32(np.float16(g*scale))/np.float32(scale) == g

low, master = np.float16(1), np.float32(1)
for _ in range(10):
    low = np.float16(low - np.float16(1e-4))
    master = np.float32(master - np.float32(1e-4))
assert low == 1
assert float(master) == 0.998999834060669
assert float(np.float16(master)) == 0.9990234375
assert Q(2*2+6*10, 8) == 8
assert Q(2+10, 2) == 6
# The same weighting follows from an independent derivative of mean squared loss.
y = np.array([-1, -3, -5, -7, -9, -11, -13, -15], dtype=np.float64)
full_grad = -y.mean()
parts = [-y[:2].mean(), -y[2:].mean()]
assert full_grad == 8 == Q(2,8)*parts[0] + Q(6,8)*parts[1]
assert np.mean(parts) == 6
loss = lambda theta: np.mean((theta-y)**2)/2
assert isclose((loss(1e-4)-loss(-1e-4))/2e-4, full_grad, rel_tol=1e-10)

# Enumerate every nonnegative finite E4M3FN bit pattern, including subnormals.
codebook = []
for code in range(127):
    e, m = code >> 3, code & 7
    value = Q(m, 512) if e == 0 else (1+Q(m,8))*Q(2)**(e-7)
    codebook.append((value, code))
assert codebook[1][0] == Q(2)**-9 and codebook[-1][0] == 448

def e4m3(x):
    x = Q(x)
    value, _ = min(codebook, key=lambda pair: (abs(pair[0]-abs(x)), pair[1]&1))
    return -value if x < 0 else value

for value, code in codebook:
    assert e4m3(value) == value and e4m3(-value) == -value
for (a, ac), (b, bc) in zip(codebook, codebook[1:]):
    assert e4m3((a+b)/2) == (a if ac % 2 == 0 else b)
assert e4m3(Q(6,5)) == Q(5,4)
assert e4m3(896)*1 == 448
assert e4m3(Q(896,2))*2 == 896

def block_scale(values):
    amax = max(map(abs, values))
    s = Q(1)
    while s*448 < amax:
        s *= 2
    while s*224 >= amax:
        s /= 2
    return s

groups = [[Q(2)**20] + [Q(1)]*31, [Q(1)]*32]
all_values = groups[0]+groups[1]
global_scale = block_scale(all_values)
assert global_scale == 4096
global_result = [global_scale*e4m3(x/global_scale) for x in all_values]
assert global_result == [Q(2)**20]+[Q(0)]*63
scales = [block_scale(group) for group in groups]
assert scales == [4096, Q(2)**-8]
local_result = [[s*e4m3(x/s) for x in group] for s,group in zip(scales,groups)]
assert local_result[0] == [Q(2)**20]+[Q(0)]*31
assert local_result[1] == [Q(1)]*32
assert Q(1,32)*100 == Q(25,8)
assert Q(3*10**8*4, 15*10**11) == Q(8,10000)
assert Q(3*10**8*2, 15*10**11) == Q(4,10000)
assert Q(8,6) == Q(4,3)
assert len(re.findall(r'!\[\[02 Areas/ML & DL/', body)) == 6
assert body.count('id="cs336-systems-precision"') == 1
print('PASS actual NumPy block, BF16/FP16 rounding, sums, master weights, loss scaling, '
      'weighted gradients, all 127 positive E4M3FN codes and 126 midpoints, FP8/MX blocks; CPU only')
print('CUDA AMP block parsed, not executed')
print('chapter_sha256=' + sha256(path.read_bytes()).hexdigest())
