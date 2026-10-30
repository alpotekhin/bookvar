"""Execute the CNN chapter and verify its arithmetic and preserved D2L code."""
from pathlib import Path
from itertools import product
from types import SimpleNamespace
import ast
import hashlib
import importlib.util
import re
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 21
scope = {}
for number, code in enumerate(blocks, 1):
    print(f"Execute actual CNN block {number}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{number}", "exec"), scope)

# The accepted explanation is retained byte-for-byte outside this addition.
without_winograd = re.sub(
    r'<!-- BEGIN REVIEW: Winograd 2026-10-27 -->\n.*?<!-- END REVIEW: Winograd 2026-10-27 -->\n\n',
    '', content.decode(), flags=re.S)
assert hashlib.sha256(without_winograd.encode()).hexdigest() == \
    'faf4455eb24ef822f831521f1549b44f9f5fc5770a74c69f9b3f61f068bd66da'
without_native = re.sub(
    r'<!-- BEGIN REVIEW: Harvard im2row 2026-10-27 -->\n.*?<!-- END REVIEW: Harvard im2row 2026-10-27 -->\n\n',
    '', without_winograd, flags=re.S).replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26')
assert hashlib.sha256(without_native.encode()).hexdigest() == \
    'd2fe8eaa5ceda18400dbbfae92d1fdcb13a61c07a35964adae0ef76cda65ee6e'
previous = re.sub(r'<!-- group-equivariance 2026-10-26 -->\n.*?<!-- /group-equivariance -->\n\n',
                  '', without_native, flags=re.S)
assert hashlib.sha256(previous.encode()).hexdigest() == \
    'c6cceb20910882ae1408dc7013c286ce77c3e00b0ab32070505e7df6b6bf5355'
source_readme = (ROOT / '05 Источники/Originals/group-equivariance-2026-10-26/escnn-README.md').read_text()
assert hashlib.sha256(source_readme.encode()).hexdigest() == \
    '946ac8fb2cd6d40e734f20cb27f7d9706626403c6a35f79f5e3503b6ed58e6cb'
for name in ['escnn-feature-fields', 'escnn-demo']:
    quote = re.search(r'<!-- source-excerpt: '+name+r' -->\n(.*?)\n<!-- /source-excerpt -->',
                      content.decode(), re.S)[1]
    plain = '\n'.join(re.sub(r'^> ?', '', line) for line in quote.splitlines())
    assert ' '.join(plain.split()) in ' '.join(source_readme.split())
print('PASS preserved accepted CNN revision and two verbatim English escnn excerpts')

# Compare every printed Winograd matrix against the original symbolic generator.
win_sources = ROOT / '05 Источники/Originals/winograd-2026-10-27'
original_module = win_sources / 'wincnn.py'
assert hashlib.sha256(original_module.read_bytes()).hexdigest() == \
    '144e2c2d0ab7b81241e4f267fa67ab55f141b0cf76bb7d697f4087c3e262c317'
spec = importlib.util.spec_from_file_location('pinned_wincnn', original_module)
wincnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wincnn)
original_at, original_g, original_bt, _ = wincnn.cookToomFilter((0, 1, -1), 2, 3)
for printed, symbolic in [('win_AT', original_at), ('win_G', original_g), ('win_BT', original_bt)]:
    torch.testing.assert_close(scope[printed], torch.tensor(symbolic.tolist(), dtype=torch.float64))
original_readme = (win_sources / 'wincnn-README.md').read_text()
quote = re.search(r'<!-- source-excerpt: wincnn-f23-verification -->\n(.*?)\n<!-- /source-excerpt -->',
                  content.decode(), re.S)[1]
plain = '\n'.join(re.sub(r'^> ?', '', line) for line in quote.splitlines())
assert ' '.join(plain.split()) in ' '.join(original_readme.split())
win_generator = torch.Generator().manual_seed(20261027)
win_cases = 0
for h, w, n, c, k in product([3, 4, 7, 8], [3, 4, 7, 8], [1, 2], [1, 3], [1, 4]):
    image = torch.randn(n, c, h, w, generator=win_generator, dtype=torch.float64)
    weight = torch.randn(k, c, 3, 3, generator=win_generator, dtype=torch.float64)
    result = scope['winograd_valid_3x3'](image, weight)
    torch.testing.assert_close(result, F.conv2d(image, weight), atol=2e-12, rtol=2e-12)
    win_cases += 1
win_image = torch.randn(1, 2, 5, 6, generator=win_generator, dtype=torch.float64, requires_grad=True)
win_weight = torch.randn(2, 2, 3, 3, generator=win_generator, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(scope['winograd_valid_3x3'], (win_image, win_weight))
direct_1d = torch.tensor([1., 2048., 1., 1.], dtype=torch.float64)
torch.testing.assert_close(F.conv1d(direct_1d[None, None],
    torch.tensor([[[0., 0., 1.]]], dtype=torch.float64)).flatten(), torch.ones(2, dtype=torch.float64))
print(f'PASS original wincnn matrices/excerpt, {win_cases} complete batched/channel/boundary cases, Winograd gradcheck and FP16 counterexample')

# Recover the displayed four rows from independently indexed input windows.
diagram_input = scope['harvard_input']
independent_rows = torch.stack([
    diagram_input[0, :, i:i+2, j:j+2].reshape(-1)
    for i in range(2) for j in range(2)])
torch.testing.assert_close(independent_rows, scope['harvard_rows'])
channel_sums = torch.stack([
    scope['corr2d'](diagram_input[0, c], scope['harvard_filter'][0, c])
    for c in range(2)])
torch.testing.assert_close(channel_sums.flatten(1),
    torch.tensor([[37., 47., 67., 77.], [319., 345., 397., 423.]], dtype=torch.float64))
torch.testing.assert_close(channel_sums.sum(0).flatten(), scope['harvard_values'])
counts = F.fold(torch.ones_like(scope['harvard_columns']), (3, 3), kernel_size=2)
torch.testing.assert_close(counts[0, 0],
    torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=torch.float64))
assert counts.sum().item() == 32 and diagram_input.numel() == 18
print('PASS original Harvard window values, both channel contributions, transposed unfold and overlap multiplicities')

rotation_generator = torch.Generator().manual_seed(20261026)
rotation_cases = 0
for _ in range(20):
    for side, in_channels, out_channels in product([5, 7, 9], [1, 2], [1, 3]):
        image = torch.randn(2, in_channels, side, side, generator=rotation_generator, dtype=torch.float64)
        kernel = torch.randn(out_channels, in_channels, 3, 3, generator=rotation_generator, dtype=torch.float64)
        maps = scope['lift_c4'](image, kernel)
        for turn in [1, 2, 3]:
            turned = scope['lift_c4'](torch.rot90(image, turn, (-2, -1)), kernel)
            expected = torch.rot90(maps, turn, (-2, -1)).roll(turn, dims=2)
            torch.testing.assert_close(turned, expected, atol=2e-12, rtol=2e-12)
            torch.testing.assert_close(turned.amax(dim=(2, 3, 4)), maps.amax(dim=(2, 3, 4)))
            rotation_cases += 1
grad_image = torch.randn(1, 1, 5, 5, generator=rotation_generator, dtype=torch.float64, requires_grad=True)
grad_kernel = torch.randn(1, 1, 3, 3, generator=rotation_generator, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(scope['lift_c4'], (grad_image, grad_kernel))
print(f'PASS {rotation_cases} quarter-turn equivariance/pooling cases and a lifting gradcheck')

sources = ROOT / "05 Источники/Originals/cnn-2026-10-26"
def extract_node(path, expected_sha, pattern, kind, name):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha
    code = re.search(pattern, data.decode(), re.S).group(1)
    tree = ast.parse(code)
    return next(node for node in tree.body if isinstance(node, kind) and node.name == name)

source_vgg = extract_node(sources / "d2l-vgg.md",
    "c27ef2b630bab632e4090da38f0377f26e5642ab686f4d3ba4b79a4b934d0bfa",
    r"%%tab pytorch\n(def vgg_block\(.*?)(?=\n```)", ast.FunctionDef, "vgg_block")
source_residual = extract_node(ROOT / "05 Источники/Originals/initialization-normalization-2026-10-21/d2l-resnet.md",
    "64325be1880bac5ed388512412877b9fae653e4c967789c93319a4a91a316ed5",
    r"%%tab pytorch\n(class Residual\(.*?)(?=\n```)", ast.ClassDef, "Residual")
chapter_tree = ast.parse("\n\n".join(blocks))
for original, name in [(source_vgg, "vgg_block"), (source_residual, "Residual")]:
    actual = next(node for node in chapter_tree.body if isinstance(node, type(original)) and node.name == name)
    assert ast.dump(actual) == ast.dump(original), f"{name} differs from pinned D2L"
print("PASS exact AST fidelity of original D2L vgg_block and Residual")

source_corr = extract_node(sources / "d2l-conv-layer.md",
    "5f41b0ebf6844a96372e103e3dd0efe0341ea959532896a340be13d0c8c23275",
    r"%%tab pytorch\n(def corr2d\(.*?)(?=\n```)", ast.FunctionDef, "corr2d")
source_scope = {"torch": torch, "d2l": SimpleNamespace(zeros=torch.zeros, reduce_sum=torch.sum)}
exec(compile(ast.Module(body=[source_corr], type_ignores=[]), "pinned-d2l-corr2d", "exec"), source_scope)
generator = torch.Generator().manual_seed(26)
corr_comparisons = 0
for seed in range(20):
    for height, width, kh, kw in [(3, 4, 2, 2), (5, 3, 3, 1), (7, 7, 3, 3), (3, 3, 1, 1)]:
        x = torch.randn(height, width, generator=generator)
        k = torch.randn(kh, kw, generator=generator)
        original = source_scope["corr2d"](x, k)
        actual = scope["corr2d"](x, k)
        torch.testing.assert_close(actual, original)
        torch.testing.assert_close(actual, F.conv2d(x[None, None], k[None, None])[0, 0], atol=2e-6, rtol=2e-5)
        corr_comparisons += 1
for seed in range(4):
    x = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(scope["corr2d"], (x, k))
print(f"PASS {corr_comparisons} pinned D2L/library correlation comparisons and four finite-difference gradchecks")

geometry_cases = 0
for height, width, kh, kw, stride, padding, dilation in product(
        [3, 5, 10], [4, 7], [1, 2, 3], [1, 3], [1, 2, 3], [0, 1], [1, 2]):
    oh = scope["out_size"](height, kh, stride, padding, dilation)
    ow = scope["out_size"](width, kw, stride, padding, dilation)
    if min(oh, ow) <= 0:
        continue
    x = torch.arange(height * width, dtype=torch.float64).reshape(1, 1, height, width) / 10
    k = torch.arange(kh * kw, dtype=torch.float64).reshape(1, 1, kh, kw) + 1
    output = F.conv2d(x, k, stride=stride, padding=padding, dilation=dilation)
    padded = F.pad(x[0, 0], (padding, padding, padding, padding))
    expected = torch.empty(oh, ow, dtype=torch.float64)
    for i in range(oh):
        for j in range(ow):
            expected[i, j] = sum(
                padded[i * stride + u * dilation, j * stride + v * dilation] * k[0, 0, u, v]
                for u in range(kh) for v in range(kw))
    assert output.shape == (1, 1, oh, ow)
    torch.testing.assert_close(output[0, 0], expected)
    geometry_cases += 1
print(f"PASS {geometry_cases} independently indexed rectangular stride/padding/dilation cases")

# Reversing a specified kernel changes the result; relabelling its parameters can compensate.
x, k = scope["X"], scope["K"]
assert not torch.equal(F.conv2d(x[None, None], k[None, None]),
                       F.conv2d(x[None, None], k.flip((0, 1))[None, None]))
# 'same' cannot silently accept a strided configuration in this library.
try:
    nn.Conv2d(1, 1, 3, stride=2, padding="same")
except ValueError:
    pass
else:
    raise AssertionError("Conv2d same/stride contract changed")
even_input = torch.ones(1, 1, 4, 5, dtype=torch.float64)
assert F.conv2d(F.pad(even_input, (0, 1, 0, 1)), torch.ones(1, 1, 2, 2, dtype=torch.float64)).shape == even_input.shape
negative = -torch.ones(1, 1, 2, 2)
assert torch.equal(F.max_pool2d(negative, 3, stride=1, padding=1), negative)
ones = torch.ones(1, 1, 2, 2)
assert F.avg_pool2d(ones, 3, stride=1, padding=1, count_include_pad=True)[0, 0, 0, 0] == 4/9
assert F.avg_pool2d(ones, 3, stride=1, padding=1, count_include_pad=False)[0, 0, 0, 0] == 1
left = torch.tensor([[[0., 1., 0., 0.]]])
right = torch.tensor([[[0., 0., 1., 0.]]])
assert F.max_pool1d(left, 2).tolist() == [[[1., 0.]]]
assert F.max_pool1d(right, 2).tolist() == [[[0., 1.]]]

# Depthwise + pointwise is a constrained spatial kernel, not an arbitrary dense one.
depth = torch.randn(3, 1, 3, 3, generator=generator, dtype=torch.float64)
point = torch.randn(16, 3, 1, 1, generator=generator, dtype=torch.float64)
image = torch.randn(2, 3, 8, 9, generator=generator, dtype=torch.float64)
combined = point[:, :, 0, 0, None, None] * depth[:, 0][None]
torch.testing.assert_close(F.conv2d(F.conv2d(image, depth, groups=3), point), F.conv2d(image, combined))
assert depth.numel() + point.numel() == 75 and combined.numel() == 432
assert sum(p.numel() for p in nn.Conv2d(3, 16, 3).parameters()) == 448

# A shortcut does not rule out cancellation of the input derivative.
scalar = torch.tensor([1.], requires_grad=True)
cancelled = scalar + (-scalar)
assert torch.autograd.grad(cancelled.sum(), scalar)[0].item() == 0
assert scope["edge_losses"][-1] < scope["edge_losses"][0]
assert len(scope["lenet_shapes"]) == 12
assert sum(isinstance(m, scope["Residual"]) for m in scope["resnet18"].modules()) == 8
assert sum(isinstance(m, nn.Conv2d) for m in scope["resnet18"].modules()) == 20  # 17 main-path + 3 projections.
assert scope["resnet_shapes"][-1] == (1, 1000)
assert 9216 * 4096 * 4 / 1024**2 == 144
print("PASS grouped/depthwise algebra, pooling conventions, padding boundaries, parameter counts and shortcut cancellation")
print("CHAPTER SHA256", hashlib.sha256(content).hexdigest())
