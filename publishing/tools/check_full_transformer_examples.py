#!/usr/bin/env python3
"""Execute the published examples and compare layers, gradients and causality."""
from pathlib import Path
import hashlib
import math
import re
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/05 Attention и Transformer/03 Полный Transformer.md"
text = CHAPTER.read_text()
# Preserve the original standalone model checks; the added dropout example is
# executed by check_structured_dropout_examples.py against both live locales.
model_text = re.sub(r"<!-- BEGIN REVIEW: Transformer dropout 2026-10-29 -->.*?<!-- END REVIEW: Transformer dropout 2026-10-29 -->", "", text, flags=re.S)
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", model_text, re.S)
assert len(blocks) == 5
spaces = []
torch.set_num_threads(1)
for i, code in enumerate(blocks, 1):
    ns = {}
    exec(compile(code, f"{CHAPTER}:block{i}", "exec"), ns)
    spaces.append(ns)
    print("PASS standalone published block", i)

dtype = torch.float64
x = torch.tensor([1., 2., 3., 4.], dtype=dtype)
mean, var = x.mean(), x.var(correction=0)
ln = F.layer_norm(x, (4,), eps=1e-12)
torch.testing.assert_close(ln, (x-2.5)/math.sqrt(1.25))
post = F.layer_norm(3*x, (4,), eps=1e-12)
pre = x+2*ln
torch.testing.assert_close(post, ln)
assert not torch.allclose(post, pre)
ff = spaces[0]
torch.testing.assert_close(ff["u"], torch.tensor([[1., -2., -2.], [-1., 4., 2.]]))
changed = ff["x"].detach().clone()
changed[1] += 7
torch.testing.assert_close(((changed@ff["w1"]+ff["b1"]).relu()@ff["w2"])[0], ff["y"][0])
cross = spaces[1]
sdpa = F.scaled_dot_product_attention(
    cross["q"], cross["k"], cross["v"], attn_mask=torch.tensor([[True, True, False]]))
torch.testing.assert_close(sdpa, cross["context"])
head = spaces[2]
assert math.isclose(head["loss"].item(), .34901221676818633)
assert math.isclose(head["smooth_ce"].item(), .5490122167681863)
torch.testing.assert_close(head["smooth_ce"]-head["smooth_kl"],
                           -(head["soft_target"]*head["soft_target"].log()).sum())
print("PASS worked FFN, norm layout, rectangular cross-attention, CE/KL and gradients")

n = spaces[3]
def copy_mha(dst, src):
    with torch.no_grad():
        dst.in_proj_weight.copy_(torch.cat([src.q.weight, src.k.weight, src.v.weight]))
        dst.in_proj_bias.copy_(torch.cat([src.q.bias, src.k.bias, src.v.bias]))
        dst.out_proj.load_state_dict(src.o.state_dict())

def copy_ff(dst, src):
    dst.linear1.load_state_dict(src.ff[0].state_dict())
    dst.linear2.load_state_dict(src.ff[2].state_dict())

def compare_parameter_gradients(ours, native, attention_pairs):
    for source, target in attention_pairs:
        for suffix in ("weight", "bias"):
            combined = torch.cat([ours[f"{source}.{part}.{suffix}"]
                                  for part in ("q", "k", "v")])
            torch.testing.assert_close(combined, native[f"{target}.in_proj_{suffix}"])
            torch.testing.assert_close(ours[f"{source}.o.{suffix}"],
                                       native[f"{target}.out_proj.{suffix}"])
    pairs = [("ff.0", "linear1"), ("ff.2", "linear2"),
             ("n1", "norm1"), ("n2", "norm2")]
    if "n3.weight" in ours:
        pairs.append(("n3", "norm3"))
    for source, target in pairs:
        for suffix in ("weight", "bias"):
            torch.testing.assert_close(ours[f"{source}.{suffix}"],
                                       native[f"{target}.{suffix}"])

torch.manual_seed(19)
enc = n["EncoderBlock"](8, 2, 16, 0).double()
native_enc = nn.TransformerEncoderLayer(
    8, 2, 16, dropout=0, batch_first=True, norm_first=False, dtype=dtype)
copy_mha(native_enc.self_attn, enc.attn)
copy_ff(native_enc, enc)
native_enc.norm1.load_state_dict(enc.n1.state_dict())
native_enc.norm2.load_state_dict(enc.n2.state_dict())
valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
inp = torch.randn(2, 4, 8, dtype=dtype, requires_grad=True)
ref_inp = inp.detach().clone().requires_grad_()
actual = enc(inp, valid[:, None, None, :])
expected = native_enc(ref_inp, src_key_padding_mask=~valid)
torch.testing.assert_close(actual, expected)
g = torch.randn_like(actual)
ag = torch.autograd.grad((actual*g).sum(), (inp, *enc.parameters()))
rg = torch.autograd.grad((expected*g).sum(), (ref_inp, *native_enc.parameters()))
torch.testing.assert_close(ag[0], rg[0])
compare_parameter_gradients(
    dict(zip(dict(enc.named_parameters()), ag[1:])),
    dict(zip(dict(native_enc.named_parameters()), rg[1:])), [("attn", "self_attn")])

dec = n["DecoderBlock"](8, 2, 16, 0).double()
native_dec = nn.TransformerDecoderLayer(
    8, 2, 16, dropout=0, batch_first=True, norm_first=False, dtype=dtype)
copy_mha(native_dec.self_attn, dec.self_attn)
copy_mha(native_dec.multihead_attn, dec.cross_attn)
copy_ff(native_dec, dec)
for i in (1, 2, 3):
    getattr(native_dec, f"norm{i}").load_state_dict(getattr(dec, f"n{i}").state_dict())
tgt_valid = torch.tensor([[True, True, True], [True, True, False]])
causal = torch.ones(3, 3, dtype=torch.bool).tril()
tgt = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
mem = torch.randn(2, 4, 8, dtype=dtype, requires_grad=True)
rt, rm = tgt.detach().clone().requires_grad_(), mem.detach().clone().requires_grad_()
allowed = causal[None, None] & tgt_valid[:, None, None, :]
actual = dec(tgt, mem, allowed, valid[:, None, None, :])
expected = native_dec(rt, rm, tgt_mask=~causal,
                      tgt_key_padding_mask=~tgt_valid, memory_key_padding_mask=~valid)
torch.testing.assert_close(actual, expected)
g = torch.randn_like(actual)
ag = torch.autograd.grad((actual*g).sum(), (tgt, mem, *dec.parameters()))
rg = torch.autograd.grad((expected*g).sum(), (rt, rm, *native_dec.parameters()))
for a, b in zip(ag[:2], rg[:2]):
    torch.testing.assert_close(a, b)
compare_parameter_gradients(
    dict(zip(dict(dec.named_parameters()), ag[2:])),
    dict(zip(dict(native_dec.named_parameters()), rg[2:])),
    [("self_attn", "self_attn"), ("cross_attn", "multihead_attn")])
changed_mem = mem.detach().clone()
changed_mem[~valid] = 10000
torch.testing.assert_close(actual.detach(),
    dec(tgt.detach(), changed_mem, allowed, valid[:, None, None, :]))
assert dec.cross_attn.last_weights.shape == (2, 2, 3, 4)
assert torch.count_nonzero(dec.cross_attn.last_weights.masked_select(
    (~valid[:, None, None, :]).expand_as(dec.cross_attn.last_weights))) == 0
try:
    enc(inp, torch.zeros(2, 1, 1, 4, dtype=torch.bool))
except ValueError:
    pass
else:
    raise AssertionError("all-masked source must be rejected")
print("PASS both post-norm layers against PyTorch: unequal lengths, padding and all input/memory/parameter gradients")

model, src, ids, labels = n["model"], n["src"], n["decoder_ids"], n["labels"]
model.eval()
before = n["logits"].detach()
expected_losses = torch.tensor(
    [9.786661926179915, 4.6317245489217065, 10.881613312063578,
     12.110339920979285, 0.], dtype=dtype)
torch.testing.assert_close(n["token_losses"].detach(), expected_losses)
torch.testing.assert_close(n["target_probabilities"][0, :4], torch.tensor(
    [5.619617149893091e-5, .009737951040379715, 1.8800759085342424e-5,
     5.50232448717143e-6], dtype=dtype))
assert math.isclose(n["token_losses"].mean().item(), 7.482067941628897)
torch.testing.assert_close(n["loss"], n["token_losses"].sum()/4)
probe_logits = before.clone().requires_grad_()
F.cross_entropy(probe_logits.flatten(0, 1), labels.flatten(), ignore_index=0).backward()
assert torch.count_nonzero(probe_logits.grad[0, 4]) == 0
torch.testing.assert_close(probe_logits.grad[0, :4, 0], before.softmax(-1)[0, :4, 0]/4)
assert torch.count_nonzero(probe_logits.grad[0, :4, 0]) == 4
assert n["new_loss"].item() < n["loss"].item()
assert model.emb.weight is model.head.weight
assert model.enc[0].attn.q.weight is not model.enc[1].attn.q.weight
assert model.dec[0].cross_attn.k.weight is not model.dec[1].cross_attn.k.weight
full = model(src, ids)
mutated = ids.clone()
mutated[:, 3] = 8
torch.testing.assert_close(model(src, mutated)[:, :3], full[:, :3])
memory, source_valid = model.encode(src)
for length in range(1, ids.shape[1]+1):
    prefix = model.decode(ids[:, :length], memory, source_valid)
    torch.testing.assert_close(prefix, full[:, :length])
lp = full.log_softmax(-1)
selected = [lp[0, i, int(labels[0, i])] for i in range(labels.shape[1])
            if labels[0, i] != 0]
manual_loss = -torch.stack(selected).mean()
torch.testing.assert_close(manual_loss,
    F.cross_entropy(full.flatten(0, 1), labels.flatten(), ignore_index=0))
model.zero_grad()
manual_loss.backward()
for param in model.parameters():
    assert param.grad is not None and torch.isfinite(param.grad).all()
assert model.enc[0].attn.q.weight.grad.abs().max() > 0

# Untie an otherwise identical model to separate the two gradient paths.
split = n["TinyTranslation"]().double()
split.load_state_dict(model.state_dict())
split.head.weight = nn.Parameter(split.emb.weight.detach().clone())
split.eval()
split_logits = split(src, ids)
torch.testing.assert_close(split_logits, full)
F.cross_entropy(split_logits.flatten(0, 1), labels.flatten(), ignore_index=0).backward()
assert torch.count_nonzero(split.emb.weight.grad[0]) == 0
assert torch.count_nonzero(split.head.weight.grad[0]) > 0
torch.testing.assert_close(model.emb.weight.grad,
                           split.emb.weight.grad + split.head.weight.grad)

# A trained output table alone does not establish a path through the encoder.
detached = n["TinyTranslation"]().double()
detached.load_state_dict(model.state_dict())
detached.eval()
detached_memory, detached_valid = detached.encode(src)
detached_logits = detached.decode(ids, detached_memory.detach(), detached_valid)
torch.testing.assert_close(detached_logits, full)
F.cross_entropy(detached_logits.flatten(0, 1), labels.flatten(), ignore_index=0).backward()
assert all(p.grad is None for p in detached.enc.parameters())
assert detached.head.weight.grad is not None and detached.head.weight.grad.abs().max() > 0
with torch.no_grad():
    pad_row = model.emb.weight[0].clone()
    model.emb.weight[0] += 1000
    changed_memory, _ = model.encode(src)
    model.emb.weight[0].copy_(pad_row)
torch.testing.assert_close(changed_memory[:, :3], memory[:, :3])
print("PASS full translation graph: four targets, tied weights, all prefixes, no future/PAD leakage, SGD")
print("PASS tokenwise CE, valid-target denominator, ignored rows versus PAD class, tied gradient sum and detached-memory negative control")
print("loss", n["loss"].item(), "->", n["new_loss"].item())

m = spaces[4]
norm = m["RMSNorm"](4, eps=1e-12).double()
torch.testing.assert_close(norm(x), x/math.sqrt(7.5))
nx = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
assert torch.autograd.gradcheck(norm, (nx,))
attn = m["RotarySelfAttention"](8, 2).double()
inp = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
actual = attn(inp)
q, k, v = [proj(inp).reshape(2, 3, 2, 4).transpose(1, 2)
           for proj in (attn.q, attn.k, attn.v)]
# Independent dense block-diagonal rotations, applied with row-vector convention.
rots = torch.zeros(3, 4, 4, dtype=dtype)
for t in range(3):
    for pair in range(2):
        theta = t * 10000 ** (-2*pair/4)
        c, s = math.cos(theta), math.sin(theta)
        rots[t, 2*pair:2*pair+2, 2*pair:2*pair+2] = torch.tensor(
            [[c, -s], [s, c]], dtype=dtype)
qr = torch.einsum("bhtd,tfd->bhtf", q, rots)
kr = torch.einsum("bhtd,tfd->bhtf", k, rots)
ref = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)
expected = attn.o(ref.transpose(1, 2).reshape(2, 3, 8))
torch.testing.assert_close(actual, expected)
g = torch.randn_like(actual)
params = (inp, *attn.parameters())
ag = torch.autograd.grad((actual*g).sum(), params, retain_graph=True)
rg = torch.autograd.grad((expected*g).sum(), params)
for a, b in zip(ag, rg):
    torch.testing.assert_close(a, b)
gate = m["SwiGLU"](8, 24).double()
gx = torch.randn(2, 3, 8, dtype=dtype)
u = gx@gate.gate.weight.T
manual = ((u*u.sigmoid())*(gx@gate.up.weight.T))@gate.down.weight.T
torch.testing.assert_close(gate(gx), manual)
lm, ids = m["lm"], m["tokens"][:, :-1]
full = lm(ids)
for length in range(1, ids.shape[1]+1):
    torch.testing.assert_close(lm(ids[:, :length]), full[:, :length])
changed = ids.clone()
changed[:, -1] = 9
torch.testing.assert_close(lm(changed)[:, :-1], full[:, :-1])
for p in lm.parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all()
assert lm.head.weight is not lm.emb.weight
print("PASS modern LM: dense RoPE/SDPA, all attention gradients, RMSNorm gradcheck, SwiGLU and causal prefixes")
print("PASS chapter SHA256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
