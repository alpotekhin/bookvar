"""Execute the actual chapter blocks and compare with independent operators/source code."""
from pathlib import Path
import ast
import hashlib
import itertools
import math
import re
from types import SimpleNamespace
from typing import Optional
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/07 Анатомия современной LLM/04 RoPE.md'
body = CHAPTER.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', body, re.S)
assert len(blocks) == 4
ns = {}
for i, block in enumerate(blocks, 1):
    exec(compile(block, f'{CHAPTER.name}:block-{i}', 'exec'), ns)
    if i == 1:
        assert abs((ns['qr']*ns['kr']).sum().item()-2) < 1e-13
    elif i == 2:
        scores = [math.cos(j-4)/math.sqrt(2) for j in range(5)]
        exps = [math.exp(z) for z in scores]
        weights = [e/sum(exps) for e in exps]
        torch.testing.assert_close(ns['weights'][0,0,-1], torch.tensor(weights,dtype=torch.float64))
        assert abs(ns['full'][0,-1,0,0].item()-sum((j+1)*w for j,w in enumerate(weights))) < 1e-14
    elif i == 4:
        assert abs((ns['fresh']-ns['stale']).abs().max().item()-.0023262835464963327) < 1e-14
    print('PASS published block', i)
rope, frequencies, attention = [ns[k] for k in ('rope', 'frequencies', 'attention')]
torch.manual_seed(22)

def matrix_rope(x, positions, omega):
    """Independent real 2x2 multiplication, including unrotated tail."""
    result = x.clone()
    if positions.ndim == 1:
        positions = positions[None, :].expand(x.shape[0], -1)
    for b, t, h in itertools.product(range(x.shape[0]), range(x.shape[1]), range(x.shape[2])):
        for j, w in enumerate(omega):
            angle = positions[b, t] * w
            c, s = angle.cos(), angle.sin()
            matrix = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
            result[b, t, h, 2*j:2*j+2] = matrix @ x[b, t, h, 2*j:2*j+2]
    return result

x = torch.randn(2, 5, 3, 8, dtype=torch.float64, requires_grad=True)
pos = torch.tensor([[0, 1, 5, 9, 10], [3, 4, 6, 8, 12]])
omega = frequencies(6)
actual = rope(x, pos, omega)
expected = matrix_rope(x, pos, omega)
torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
assert torch.equal(actual[..., 6:], x[..., 6:])
torch.testing.assert_close(rope(x, torch.zeros_like(pos), omega), x)
torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                           torch.autograd.grad(expected.sum(), x)[0])
assert torch.autograd.gradcheck(lambda z: rope(z, pos, omega), (x,), fast_mode=True)
q, k = [torch.randn(1, 1, 2, 8, dtype=torch.float64) for _ in range(2)]
for delta in (0, 1, 100):
    score = (rope(q, [2+delta], frequencies(8))*rope(k, [5+delta], frequencies(8))).sum(-1)
    reference = (rope(q, [2], frequencies(8))*rope(k, [5], frequencies(8))).sum(-1)
    torch.testing.assert_close(score, reference, atol=2e-13, rtol=2e-13)
print('PASS independent matrices, per-batch positions, partial tail, norms, gradients, shifts')

def extract_functions(path, names, environment):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), environment)
    return environment

meta_path = ROOT / '05 Источники/Originals/llama-2026-10-22/llama3-model.py'
assert hashlib.sha256(meta_path.read_bytes()).hexdigest() == '20d93de2fdb2c98fff085a1f1026d20cc4172e40809a0891e384d5c10bbb6965'
meta = extract_functions(meta_path, ['precompute_freqs_cis', 'reshape_for_broadcast', 'apply_rotary_emb'], {'torch': torch})
q = torch.randn(2, 5, 4, 8)
k = torch.randn(2, 5, 2, 8)
freqs = meta['precompute_freqs_cis'](8, 20, 10000.)[3:8]
mq, mk = meta['apply_rotary_emb'](q, k, freqs)
for actual, ref in [(rope(q, torch.arange(3, 8), frequencies(8)), mq),
                    (rope(k, torch.arange(3, 8), frequencies(8)), mk)]:
    torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-6)
for dtype in (torch.float16, torch.bfloat16):
    a, b = meta['apply_rotary_emb'](q.to(dtype), k.to(dtype), freqs)
    assert torch.equal(rope(q.to(dtype), torch.arange(3, 8), frequencies(8)), a)
    assert torch.equal(rope(k.to(dtype), torch.arange(3, 8), frequencies(8)), b)
torch.testing.assert_close(rope(k, torch.arange(5), frequencies(8)).repeat_interleave(2, dim=2),
                           rope(k.repeat_interleave(2, dim=2), torch.arange(5), frequencies(8)))
print('PASS pinned Meta source functions: FP32/FP16/BF16 and GQA repeat')

# Independent split-half implementation and a wrong-layout negative control.
p = torch.tensor([0, 2, 1, 3])
q, k = [torch.tensor(v, dtype=torch.float64).reshape(1, 1, 1, 4)
        for v in ([1, 2, 3, 4], [4, 3, 2, 1])]
omega = frequencies(4)
def split_half(x, position):
    c, s = (position*omega).cos(), (position*omega).sin()
    a, b = x[..., :2], x[..., 2:]
    return torch.cat((a*c-b*s, a*s+b*c), -1)
torch.testing.assert_close(split_half(q[..., p], 2), rope(q, [2], omega)[..., p])
correct = (rope(q, [2], omega)*rope(k, [5], omega)).sum()
converted = (split_half(q[..., p], 2)*split_half(k[..., p], 5)).sum()
wrong = (split_half(q, 2)*split_half(k, 5)).sum()
torch.testing.assert_close(correct, converted)
assert abs(correct-wrong) > .1
print('PASS pairing permutation and wrong-layout counterexample', correct.item(), wrong.item())

q, k, v = [torch.randn(1, 5, 2, d, dtype=torch.float64, requires_grad=True) for d in (4, 4, 3)]
pos = torch.arange(5)
omega = frequencies(4)
qr, kr = rope(q, pos, omega), rope(k, pos, omega)
full = attention(qr, kr, v, pos, pos)[0]
full_grad = torch.autograd.grad(full.square().sum(), (q, k, v), retain_graph=True)
for cuts in itertools.product((False, True), repeat=4):
    boundaries = [0]+[i+1 for i, cut in enumerate(cuts) if cut]+[5]
    kc = vc = None
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        new_k, new_v = rope(k[:, start:end], pos[start:end], omega), v[:, start:end]
        kc = new_k if kc is None else torch.cat((kc, new_k), 1)
        vc = new_v if vc is None else torch.cat((vc, new_v), 1)
        outputs.append(attention(rope(q[:, start:end], pos[start:end], omega), kc, vc, pos[start:end], pos[:end])[0])
    cached = torch.cat(outputs, 1)
    torch.testing.assert_close(cached, full)
    for a, b in zip(torch.autograd.grad(cached.square().sum(), (q, k, v), retain_graph=True), full_grad):
        torch.testing.assert_close(a, b)
wrong = attention(qr, rope(kr, pos, omega), v, pos, pos)[0]
assert (wrong-full).abs().max() > .01
print('PASS all 16 cached partitions and q/k/v gradients; double rotation fails')

# The causal mask is not invariant under permuting content while holding the mask fixed.
values = torch.tensor([1., 3., 7.], dtype=torch.float64)
def causal_means(z):
    return z.cumsum(0)/torch.arange(1, 4)
assert not torch.allclose(causal_means(values.flip(0)), causal_means(values).flip(0))

# A concrete two-layer counterexample to merely rephasing old upper-layer keys.
x = torch.tensor([[-1.,.5,.3,-.2],[.2,-.7,.8,.1],[.4,.6,-.9,.3],
                  [-.5,.2,.1,.7],[.9,-.4,.2,-.3]], dtype=torch.float64).reshape(1,5,1,4)
def block(x, omega):
    y = attention(rope(x, pos, omega), rope(x, pos, omega), x, pos, pos)[0]
    return x+y
old, new = frequencies(4, 100.), frequencies(4, 10000.)
h_old, h_new = block(x, old), block(x, new)
fresh = block(h_new, new)[:, -1:]
stale = h_new[:, -1:]+attention(rope(h_new[:, -1:], pos[-1:], new),
    rope(torch.cat((h_old[:, :-1], h_new[:, -1:]), 1), pos, new),
    torch.cat((h_old[:, :-1], h_new[:, -1:]), 1), pos[-1:], pos)[0]
assert (fresh-stale).abs().max() > 1e-6
print('PASS changed-grid upper-layer cache counterexample', (fresh-stale).abs().max().item())

hf_path = ROOT/'05 Источники/Originals/rope-2026-10-22/modeling_rope_utils-v4.57.1.py'
assert hashlib.sha256(hf_path.read_bytes()).hexdigest() == '55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac'
hf = extract_functions(hf_path, ['_compute_default_rope_parameters', '_compute_linear_scaling_rope_parameters',
     '_compute_dynamic_ntk_parameters', '_compute_yarn_parameters'], {'torch':torch, 'math':math})
cfg = SimpleNamespace(rope_theta=10000., head_dim=8, hidden_size=8, num_attention_heads=1,
                      max_position_embeddings=2048, rope_scaling={'factor':4})
linear, _ = hf['_compute_linear_scaling_rope_parameters'](cfg, torch.device('cpu'))
yarn, a = hf['_compute_yarn_parameters'](cfg, torch.device('cpu'))
torch.testing.assert_close(linear.double(), frequencies(8)/4)
torch.testing.assert_close(yarn.double(), ns['yarn_frequencies'](8,10000.,4,2048)[0])
assert abs(a-(1+.1*math.log(4))) < 1e-15
ntk, _ = hf['_compute_dynamic_ntk_parameters'](cfg, torch.device('cpu'), seq_len=3584)
torch.testing.assert_close(ntk.double(), ns['static_ntk_frequencies'](8,10000.,4))
dynamic, _ = hf['_compute_dynamic_ntk_parameters'](cfg, torch.device('cpu'), seq_len=4096)
torch.testing.assert_close(dynamic.double(), ns['static_ntk_frequencies'](8,10000.,5))
for t in (None, 1024, 2048):
    freq, _ = hf['_compute_dynamic_ntk_parameters'](cfg, torch.device('cpu'), seq_len=t)
    torch.testing.assert_close(freq.double(), frequencies(8))
torch.testing.assert_close(rope(x, pos/4, frequencies(4)), rope(x, pos, frequencies(4)/4))
print('PASS pinned HF default/linear/dynamic/YaRN; position/frequency scaling equivalence')
assert torch.tensor([2048,2049]).half().tolist() == [2048.,2048.]
assert torch.tensor([256,257]).bfloat16().tolist() == [256.,256.]
assert torch.tensor([2**24,2**24+1]).float()[0] == torch.tensor([2**24,2**24+1]).float()[1]
print('PASS position precision counterexamples')
print('PASS RoPE chapter CPU operator verification; not full-model quality or GPU kernel validation')
print('chapter SHA256', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
