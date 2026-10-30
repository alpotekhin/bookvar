#!/usr/bin/env python3
"""Execute every recurrent-attention chapter block and verify mechanisms and crops."""
from pathlib import Path
import hashlib
import math
import re
import subprocess
import tempfile
import copy
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, re.S)
assert len(blocks) == 3, len(blocks)
namespaces = []
for index, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{index}", "exec"), namespace)
    namespaces.append(namespace)
    print("PASS standalone published block", index)

n = namespaces[0]
torch.testing.assert_close(n["weights"], torch.tensor([
    .1797594729474699, .6734236212380392, .14681690581449072
], dtype=torch.float64))
torch.testing.assert_close(n["context"], torch.tensor([
    .03294256713297919, 1.4936641482905693
], dtype=torch.float64))
assert math.isclose(n["loss"].item(), .12873060372781273)
assert abs(n["delta"].sum().item()) < 1e-14
assert n["delta"][1] < 0 and n["delta"][0] > 0 and n["delta"][2] > 0
torch.testing.assert_close(n["linear1"], n["linear2"])
assert not torch.allclose(n["weights"], n["weights2"])
# Context need not identify the weights.
memory = torch.tensor([[1.,0.],[0.,1.],[.5,.5]], dtype=torch.float64)
torch.testing.assert_close(torch.tensor([.4,.4,.2], dtype=torch.float64) @ memory,
                           torch.tensor([.2,.2,.6], dtype=torch.float64) @ memory)
for row in [[.05,.76,.12,.02,.01,.02,.02],
            [.03,.11,.78,.04,.01,.01,.02],
            [.01,.02,.09,.80,.02,.02,.04],
            [.01,.01,.01,.08,.66,.10,.13],
            [.01,.01,.02,.04,.09,.13,.70]]:
    assert math.isclose(sum(row), 1.)
print("PASS all additive backward paths, query dependence, shift invariance, nonunique weights and example rows")

l = namespaces[1]
assert math.isclose(l["raw"].sum().item(), .42355685549107513)
assert math.isclose(l["context_raw"].item(), 1.2706705664732254)
assert math.isclose(l["context_normalized"].item(), 3.)
assert math.isclose(l["center"].grad.item(), .36089408863096717)
# Check the conditional derivative while holding the chosen window fixed.
def local_context(center):
    prior = torch.exp(-.5 * ((l["positions"]-center)/l["sigma"]).square())
    return (l["base"]*prior) @ l["values"]
eps = 1e-5
finite = (local_context(3+eps)-local_context(3-eps))/(2*eps)
torch.testing.assert_close(finite, l["center"].grad, rtol=1e-8, atol=1e-9)
print("PASS local Gaussian, both normalization conventions and fixed-window derivative")

m = namespaces[2]
model, src, lengths = m["model"], m["src"], m["lengths"]
PAD = m["PAD"]
assert m["after"] < .01 and m["after"] < m["before"]/100
assert m["targets"].ne(PAD).sum().item() == 18
for i, row in enumerate(m["rows"]):
    result, ended, attention = model.greedy(src[i:i+1], lengths[i:i+1])
    assert ended and result == row[::-1]
    assert attention.shape == (len(row)+1, src.size(1))
    assert attention[:, len(row):].abs().sum() == 0
    torch.testing.assert_close(attention.sum(-1), torch.ones(len(row)+1))
result, ended, empty = model.greedy(src[:1], lengths[:1], max_steps=0)
assert result == [] and not ended and empty.shape == (0, 3)
result, ended, _ = model.greedy(src[:1], lengths[:1], max_steps=1)
assert result == [5] and not ended
logits, alignment = model(src, lengths, m["dec_input"])
assert logits.shape == (6,4,6) and alignment.shape == (6,4,3)
manual = -logits.log_softmax(-1).gather(-1, m["targets"][...,None]).squeeze(-1)
torch.testing.assert_close(manual[m["targets"].ne(PAD)].mean(), m["objective"]())
memory, initial, valid = model.encode(src, lengths)
for i, length in enumerate(lengths.tolist()):
    mem, state, _ = model.encode(src[i:i+1, :length], torch.tensor([length]))
    torch.testing.assert_close(mem, memory[i:i+1, :length])
    torch.testing.assert_close(state, initial[i:i+1])
# Check query/memory masks independently of the encoder.
att = m["AdditiveAttention"](3, 4, 5).double()
torch.manual_seed(7)
query = torch.randn(2,3,dtype=torch.float64,requires_grad=True)
H = torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
mask = torch.tensor([[True,True,True],[True,False,True]])
context, weights = att(query, H, mask)
changed = H.detach().clone()
changed[~mask] = 1e5
torch.testing.assert_close(att(query, changed, mask)[0], context)
projected = torch.tanh(att.Wq(query)[:,None,:] + att.Wh(H))
expected_weights = att.v(projected).squeeze(-1).masked_fill(~mask,-torch.inf).softmax(-1)
torch.testing.assert_close(context, torch.bmm(expected_weights[:,None,:], H).squeeze(1))
context.square().sum().backward()
assert H.grad[~mask].abs().sum() == 0
assert query.grad.abs().sum() > 0
for p in att.parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0
for bad in [torch.zeros_like(mask), mask.long(), mask[:,:2]]:
    try:
        att(query, H, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mask accepted")
# Trained exact output need not select the copied source position.
_, _, first_alignment = model.greedy(src[:1], lengths[:1])
assert first_alignment[0].argmax().item() == 1
print("PASS trained reversal/EOS, masking, packed bidirectional states, manual CE and nonliteral alignment")

# One shared memory projection has the same outputs and all gradient paths
# as recomputing it for each query. This does not cache the attention weights.
torch.manual_seed(29)
prototype = m["AdditiveAttention"](8, 8, 8).double()
query_values = torch.randn(4, 1, 8, dtype=torch.float64)
memory_values = torch.randn(1, 3, 8, dtype=torch.float64)

def projection_reuse(cached, detach_keys=False):
    module = copy.deepcopy(prototype)
    queries = query_values.clone().requires_grad_()
    memory = memory_values.clone().requires_grad_()
    calls = []
    hook = module.Wh.register_forward_hook(lambda *args: calls.append(1))
    prepared = module.Wh(memory) if cached else None
    contexts, attention_weights = [], []
    for query in queries:
        keys = prepared if cached else module.Wh(memory)
        if detach_keys:
            keys = keys.detach()
        features = torch.tanh(module.Wq(query)[:, None, :] + keys)
        weights = module.v(features).squeeze(-1).softmax(-1)
        contexts.append(torch.bmm(weights[:, None, :], memory).squeeze(1))
        attention_weights.append(weights)
    output = torch.stack(contexts)
    gradients = torch.autograd.grad(output.square().sum(),
        [queries, memory, *module.parameters()], allow_unused=detach_keys)
    hook.remove()
    return output, torch.stack(attention_weights), gradients, len(calls)

uncached, cached = projection_reuse(False), projection_reuse(True)
torch.testing.assert_close(cached[0], uncached[0])
torch.testing.assert_close(cached[1], uncached[1])
for left, right in zip(cached[2], uncached[2]):
    torch.testing.assert_close(left, right)
assert cached[3] == 1 and uncached[3] == 4
assert not torch.allclose(cached[1][0], cached[1][1])
assert 3 * 8 == 24 and 3 * 8 * 8 == 192 and 4 * 3 * 8 * 8 == 768
detached = projection_reuse(True, detach_keys=True)
torch.testing.assert_close(detached[0], cached[0])
torch.testing.assert_close(detached[1], cached[1])
# Wh is the second named parameter, after Wq; its only path is via keys.
assert list(prototype.named_parameters())[1][0] == "Wh.weight"
assert detached[2][3] is None and cached[2][3].norm() > 0
assert not torch.allclose(detached[2][1], cached[2][1])
print("PASS shared memory projection: 4 vs 1 calls, 24 outputs, 192/768 MACs, identical outputs/weights/gradients")
print("PASS detach negative control: same forward, missing Wh gradient and changed memory gradient")

# Re-render the licensed PDF independently and compare crop pixels.
pdf = ROOT / "05 Источники/Originals/attention-translation-2026-10-21/luong-D15-1166.pdf"
assert hashlib.sha256(pdf.read_bytes()).hexdigest() == "e8aca09523b8807edcafb48b8ee9fa60478e9be3a3682f9ffa61fd00de40e79a"
specs = [
    ("global",3,(1522,240,2640,1208)),
    ("local",4,(315,240,1440,1208)),
    ("input-feeding",5,(405,270,1320,1208)),
]
with tempfile.TemporaryDirectory(prefix="bookvar-att-crops-") as tmp:
    subprocess.run(["pdftoppm","-f","3","-l","5","-r","360","-png",str(pdf),str(Path(tmp)/"page")],check=True)
    for name, page, box in specs:
        full = Image.open(Path(tmp)/f"page-{page:02}.png")
        assert full.size == (2977,4210)
        expected = full.crop(box).convert("RGB")
        actual = Image.open(ROOT/f"00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-{name}.png").convert("RGB")
        assert expected.size == actual.size
        assert ImageChops.difference(expected, actual).getbbox() is None
print("PASS all three PDF crops pixel-exact at recorded page/dpi/coordinates")
print("chapter_sha256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

