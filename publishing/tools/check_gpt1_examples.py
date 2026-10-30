#!/usr/bin/env python3
"""Execute published GPT-1 examples and check source-derived contracts."""
from pathlib import Path
import ast
import hashlib
import math
import re
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md"
body = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", body, re.S)
assert len(blocks) == 2
spaces = []
for i, block in enumerate(blocks, 1):
    ns = {}
    exec(compile(block, f"{CHAPTER}:block{i}", "exec"), ns)
    spaces.append(ns)
    print("PASS standalone chapter block", i)

n = spaces[0]
assert math.isclose(n["nli_loss"].item(), .4076059644, abs_tol=1e-6)
assert math.isclose(n["choice_loss"].item(), .20141327798, abs_tol=1e-6)
assert math.isclose(.4076059644 + .5 * math.log(2), .75417955468, abs_tol=1e-9)
assert math.isclose(.4076059644 + .5 * 8 * math.log(2), 3.18019468664, abs_tol=1e-9)
assert 2*math.log(.8) < math.log(.7) and math.log(.8) > math.log(.7)
# Independently differentiate the published NLI head, including its weights.
hv = n["h"].detach().numpy().astype(np.float64)
wv = n["head"].detach().numpy().astype(np.float64)
z = hv @ wv
p = np.exp(z-z.max()); p /= p.sum()
g = p-np.array([1., 0., 0.])
np.testing.assert_allclose(n["h"].grad.numpy(), g @ wv.T, atol=1e-7)
np.testing.assert_allclose(n["head"].grad.numpy(), np.outer(hv, g), atol=1e-7)
def independent_ce(a, w):
    values = a @ w
    maximum = values.max()
    return maximum+np.log(np.exp(values-maximum).sum())-values[0]
for value, expected, other in [(hv, g @ wv.T, wv), (wv, np.outer(hv, g), hv)]:
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        evaluate = (lambda a: independent_ce(a, other)) if value.ndim == 1 else (lambda w: independent_ce(other, w))
        numeric = (evaluate(plus)-evaluate(minus))/2e-5
        assert math.isclose(numeric, expected[index], abs_tol=3e-10)
print("PASS NLI/choice CE, both NLI gradients and independent finite differences, symmetry and loss normalization")

n = spaces[1]
model, ids, valid = [n[k] for k in ("model", "flat", "mask")]
model.eval()
assert n["pool"].tolist() == [5, 6, 6, 5]
assert n["target_mask"].sum(1).tolist() == [5, 6, 6, 5]
expected_score_gradient = (n["scores"].detach().softmax(-1)-F.one_hot(n["labels"], 2))/2
torch.testing.assert_close(n["scores"].grad, expected_score_gradient)
assert torch.count_nonzero(n["scores"].grad) == 4
expected_token_gradient = .5*n["target_mask"]/(4*n["target_mask"].sum(1, keepdim=True))
torch.testing.assert_close(n["token_loss"].grad, expected_token_gradient)
assert torch.count_nonzero(n["token_loss"].grad[~n["target_mask"]]) == 0
torch.testing.assert_close(n["lm_loss"], n["per_candidate"].mean())
torch.testing.assert_close(n["total_loss"], n["classification_loss"]+.5*n["lm_loss"])
assert torch.count_nonzero(n["token_loss"].grad[torch.tensor([1, 2])]) == 12
print("PASS actual joint-loss gradients: both choice scores, 1/40 versus 1/48 per target, wrong candidates and PAD")
assert model.blocks[0].self_attn.in_proj_weight is not model.blocks[1].self_attn.in_proj_weight
assert all(not b.norm_first for b in model.blocks)
with torch.no_grad():
    hidden = model(ids)
    padded = ids.clone()
    padded[~valid] = 9
    after = model(padded)
    torch.testing.assert_close(hidden[valid], after[valid])
    perturbed = ids.clone()
    perturbed[0, 4] = 8
    changed = model(perturbed)
    torch.testing.assert_close(hidden[0, :4], changed[0, :4])
    assert (hidden[0, 5] - changed[0, 5]).abs().max() > 1e-7
    pooled = hidden[torch.arange(4), n["pool"]]
    assert (pooled - hidden[:, -1]).abs().max() > 1e-3
    # Vectorized B*C computation is the same as separate candidate passes.
    serial = torch.cat([model(row[None]) for row in ids])
    torch.testing.assert_close(hidden, serial, atol=2e-6, rtol=2e-5)
    # The toy uses tied token-only output weights, not the release's combined table.
    torch.testing.assert_close(model.lm_logits(hidden), hidden @ model.word.weight.T)
    original_tokens = model.word.weight[:12].clone()
    assert not torch.equal(original_tokens, n["old_words"])
assert n["choice_head"].weight.grad.norm() > 0
assert model.position.weight.grad.norm() > 0

# Separately prove that each loss reaches the shared backbone.
for objective in ("classification", "lm"):
    model.zero_grad()
    n["choice_head"].zero_grad()
    h = model(ids)
    if objective == "classification":
        scores = n["choice_head"](h[torch.arange(4), n["pool"]]).reshape(2, 2)
        loss = F.cross_entropy(scores, n["labels"])
    else:
        losses = F.cross_entropy(model.lm_logits(h[:, :-1]).reshape(-1, 15),
                                 ids[:, 1:].reshape(-1), reduction="none").reshape(4, -1)
        mask = valid[:, 1:]
        loss = ((losses * mask).sum(1) / mask.sum(1)).mean()
    loss.backward()
    assert model.blocks[0].self_attn.in_proj_weight.grad.norm() > 0
    assert model.word.weight.grad.norm() > 0
print("PASS two actual parameter updates; post-norm/causality/PAD/pooling/candidate isolation/tying; both backbone gradients")

losses = torch.tensor([[2., 100., 100.], [1., 1., 1.]], requires_grad=True)
mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
per_candidate = (losses * mask).sum(1) / mask.sum(1)
torch.testing.assert_close(per_candidate.mean(), torch.tensor(1.5))
torch.testing.assert_close((losses * mask).sum() / mask.sum(), torch.tensor(1.25))
per_candidate.mean().backward()
torch.testing.assert_close(losses.grad, torch.tensor([[.5, 0., 0.], [1/6, 1/6, 1/6]]))
print("PASS per-candidate normalization, all candidates included, PAD has zero loss gradient")

source_dir = ROOT / "05 Источники/Originals/gpt1-2026-10-22"
source = (source_dir / "train.py").read_text()
assert hashlib.sha256((source_dir / "train.py").read_bytes()).hexdigest() == "3559bfbd8b661cb6acad1b12e119f5542705703634b1cfa70badaac371f1ae24"
excerpt = re.search(r'\x60\x60\x60python title="Original implementation — MIT"\n(.*?)\n\x60\x60\x60', body, re.S)[1]
assert excerpt.strip("\n") in source
assert (ROOT / "site/public/licenses/openai-gpt1-MIT.txt").read_bytes() == (source_dir / "LICENSE").read_bytes()
fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == "transform_roc")
ns = dict(np=np, n_ctx=12, max_len=4, n_vocab=12, n_special=3,
          encoder={"_start_": 12, "_delimiter_": 13}, clf_token=14)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "original:transform_roc", "exec"), ns)
x, m = ns["transform_roc"](n["contexts"], [a[0] for a in n["candidates"]], [a[1] for a in n["candidates"]])
np.testing.assert_array_equal(x[..., 0], n["ids"].numpy())
np.testing.assert_array_equal(m.astype(bool), n["valid"].numpy())
np.testing.assert_array_equal(x[..., 1], np.broadcast_to(np.arange(15, 27), (2, 2, 12)))
truncated, tm = ns["transform_roc"]([[0] * 20], [[1] * 20], [[2] * 20])
assert tm.sum(2).tolist() == [[11., 11.]]
assert (truncated[0, :, 10, 0] == 14).all()
print("PASS unchanged MIT excerpt and executed original transform_roc: tokens/positions/PAD/truncation")
note = (ROOT / "Papers/GPT 1.0.md").read_text()
for fact in ["85.8%", "74.7", "59.9", "75.0", "69.1", "93.2", "14.8", "5.6",
             "получает 56.0 против 61.7", "82.3 F1 против 86.0", "не было изобретено"]:
    assert fact in note
assert "few-shot prompting" not in note and "архитектура не масштабируется" not in note
print("PASS corrected legacy note: chronology, SST-2, ablations, reproduction and zero-shot scope")
print("PASS chapter SHA256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
