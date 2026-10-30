"""Execute published MLA examples, independent gradients and pinned DeepSeek code."""
from pathlib import Path
import ast, copy, dataclasses, hashlib, itertools, math, re
from typing import Optional, Literal, Tuple
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md'
body = CHAPTER.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', body, re.S)
assert len(blocks) == 4
torch.set_num_threads(1)
ns = {}
for i, block in enumerate(blocks, 1):
    exec(compile(block, f'{CHAPTER.name}:block{i}', 'exec'), ns)
    print('PASS actual published block', i)
TinyMLA = ns['TinyMLA']
torch.testing.assert_close(ns['full_scores'], torch.tensor([1., 0.], dtype=torch.float64))
assert not torch.allclose(ns['full_scores'].softmax(-1),
                         (ns['c'] @ ns['q_latent'] / math.sqrt(2)).softmax(-1))
torch.testing.assert_close(ns['rope_scores'], torch.tensor([0., 1.], dtype=torch.float64))
assert not torch.allclose(ns['rope_scores'].softmax(-1), ns['lost_position_scores'].softmax(-1))
print('PASS numerical outputs, unchanged temperature and explicit RoPE obstruction')

def independent(layer, x):
    """Per batch, query, head and history position; no absorption or vectorized mask."""
    def norm(y, gain):
        return y / (y.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gain
    cq = norm(x @ layer.wq_a.weight.T, layer.q_gain)
    q = (cq @ layer.wq_b.weight.T).reshape(x.shape[0], x.shape[1], 2, 6)
    raw = x @ layer.wkv_a.weight.T
    c = norm(raw[..., :2], layer.kv_gain)
    up = layer.wkv_b.weight.reshape(2, 6, 2)
    def rotate_pair(pair, pos):
        matrix = pair.new_tensor([[math.cos(pos), -math.sin(pos)],
                                  [math.sin(pos), math.cos(pos)]])
        return matrix @ pair
    batches = []
    for b in range(x.shape[0]):
        rows = []
        for t in range(x.shape[1]):
            heads = []
            for head in range(2):
                query = torch.cat((q[b,t,head,:4], rotate_pair(q[b,t,head,4:], t)))
                keys = torch.stack([torch.cat((up[head,:4] @ c[b,j],
                                               rotate_pair(raw[b,j,2:], j))) for j in range(t+1)])
                values = torch.stack([up[head,4:] @ c[b,j] for j in range(t+1)])
                weights = (keys @ query / math.sqrt(6)).softmax(-1)
                heads.append(weights @ values)
            rows.append(torch.cat(heads) @ layer.wo.weight.T)
        batches.append(torch.stack(rows))
    return torch.stack(batches)

def partitions(n):
    for cuts in itertools.product((False, True), repeat=n-1):
        edges = [0] + [i+1 for i, cut in enumerate(cuts) if cut] + [n]
        yield list(zip(edges, edges[1:]))

torch.manual_seed(19)
base = TinyMLA()
original_x = torch.randn(2, 5, 4, dtype=torch.float64)
ref_layer = copy.deepcopy(base)
ref_x = original_x.clone().requires_grad_()
reference = independent(ref_layer, ref_x)
reference.square().sum().backward()
for mode in ('full', 'latent'):
    for parts in partitions(5):
        layer = copy.deepcopy(base)
        x = original_x.clone().requires_grad_()
        cache, outputs = None, []
        for start, end in parts:
            y, cache = layer(x[:,start:end], cache, mode)
            outputs.append(y)
            shapes = [(2,end,2,6),(2,end,2,2)] if mode=='full' else [(2,end,2),(2,end,2)]
            assert [tuple(c.shape) for c in cache] == shapes
        actual = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-11)
        actual.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-10, atol=1e-11)
        for (name, value), (ref_name, ref_value) in zip(layer.named_parameters(), ref_layer.named_parameters()):
            assert name == ref_name
            torch.testing.assert_close(value.grad, ref_value.grad, rtol=1e-10, atol=1e-11)
print('PASS independent full attention; all 16 partitions x 2 cache layouts, B=2, outputs and all gradients')

def mutated(change):
    altered = dict(ns)
    code = blocks[2].replace(*change)
    assert code != blocks[2]
    exec(compile(code, 'negative-control', 'exec'), altered)
    layer = altered['TinyMLA']()
    layer.load_state_dict(base.state_dict())
    _, cache = layer(original_x[:,:2], mode='latent')
    result, _ = layer(original_x[:,2:], cache, 'latent')
    assert not torch.allclose(result, reference[:,2:])

mutated(('scores = scores / math.sqrt(4 + 2)', 'scores = scores / math.sqrt(2 + 2)'))
mutated(('torch.einsum("bthr,bsr->bths", qr, stored[1])',
         'torch.einsum("bthr,bsr->bths", torch.zeros_like(qr), stored[1])'))
mutated(('key_positions[None, :] <= positions[:, None]',
         'key_positions[None, :] <= torch.arange(tokens, device=x.device)[:, None]'))
mutated(('heads = torch.einsum("bthc,hdc->bthd", mixed, uv)',
         'heads = torch.einsum("bthc,hdc->bthd", mixed.mean(2, keepdim=True).expand_as(mixed), uv)'))
print('PASS negative controls: latent scale, missing positional term, unshifted mask, collapsed heads')

# Execute original class/function bodies, not a rewritten source-code imitation.
source = ROOT / '05 Источники/Originals/mla-2026-10-22/model.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'fdb7995bc234badbf0011b5af8cf391be94343abc51305428cb9a04452116f02'
tree = ast.parse(source.read_text())
names = {'ModelArgs', 'linear', 'Linear', 'ColumnParallelLinear', 'RowParallelLinear',
         'RMSNorm', 'precompute_freqs_cis', 'apply_rotary_emb', 'MLA'}
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
assert {node.name for node in nodes} == names
def no_cuda(*args, **kwargs):
    raise AssertionError('A CUDA/quantization/distributed branch was unexpectedly used')
scope = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclasses.dataclass,
             Optional=Optional, Literal=Literal, Tuple=Tuple, world_size=1, rank=0,
             block_size=128, gemm_impl='bf16', act_quant=no_cuda, weight_dequant=no_cuda,
             fp8_gemm=no_cuda)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), scope)
scope['Linear'].dtype = torch.float32
args = scope['ModelArgs'](max_batch_size=2, max_seq_len=5, original_seq_len=5,
    dim=4, n_heads=2, q_lora_rank=3, kv_lora_rank=2, qk_nope_head_dim=4,
    qk_rope_head_dim=2, v_head_dim=2)
freqs = scope['precompute_freqs_cis'](args)
source_error = 0.
with torch.inference_mode():
    expected = base(original_x, mode='full')[0].float()
    for mode in ('naive', 'absorb'):
        scope['attn_impl'] = mode
        source_layer = scope['MLA'](args)
        mapping = {'q_norm.weight':'q_gain','kv_norm.weight':'kv_gain'}
        state = base.state_dict()
        for name, parameter in source_layer.named_parameters():
            parameter.copy_(state[mapping.get(name, name)].float())
        for parts in partitions(5):
            ys = []
            for start, end in parts:
                mask = torch.zeros(end-start, end)
                mask.masked_fill_(torch.arange(end)[None,:] > torch.arange(start,end)[:,None], -torch.inf)
                y = source_layer(original_x[:,start:end].float(), start, freqs[start:end], mask)
                ys.append(y)
            actual = torch.cat(ys, dim=1)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            source_error = max(source_error, (actual-expected).abs().max().item())
            if mode == 'naive':
                cache = base(original_x, mode='full')[1]
                torch.testing.assert_close(source_layer.k_cache, cache[0].float(), rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(source_layer.v_cache, cache[1].float(), rtol=2e-5, atol=2e-6)
            else:
                cache = base(original_x, mode='latent')[1]
                torch.testing.assert_close(source_layer.kv_cache, cache[0].float(), rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(source_layer.pe_cache, cache[1].float(), rtol=2e-5, atol=2e-6)
print('PASS pinned DeepSeek-V3 original naive/absorb bodies, FP32 CPU, all 16 partitions; max error', source_error)

for elements, expected_bytes in [(32768,16*2**30),(2048,2**30),(256,128*2**20),(576,288*2**20)]:
    assert elements*8192*32*2 == expected_bytes
assert 128*(128+64+128) == 40960
assert 40960//8 == 5120 and 512+64 == 576
assert (2*512+64)/(128+64+128) == 3.4
h,r,d,c,s = 128,1536,128,512,4096
assert h*(r*d+d*c+c*s) == 301989888
assert h*(r*c+c*s) == 369098752
assert h*(r*d*c+r*c+c*s) == 13254000640
assert r*c == 3*(r*d+d*c)
print('PASS cache, TP replication and MAC calculations')
print('No released checkpoint, GPU kernel, distributed execution, timing or quality benchmark run')
print('chapter SHA256', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
