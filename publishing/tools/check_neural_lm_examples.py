#!/usr/bin/env python3
"""Execute every published block; check the chapter's mechanisms independently."""
from collections import Counter, defaultdict
from math import exp, isclose, log
from pathlib import Path
import re
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction.md"
text = CHAPTER.read_text()
assert chr(0xA4) not in text
blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
assert len(blocks) == 5, len(blocks)
namespaces = []
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    namespaces.append(namespace)
    print(f"PASS published standalone block {number}")

n = namespaces[1]
E, W = n["E"], n["W"]
p = torch.tensor([1/12, 1/12, 1/6, 1/3, 1/12, 1/12, 1/12, 1/12], dtype=torch.float64)
g = p.clone()
g[3] -= 1
torch.testing.assert_close(W.grad[:, 0], g)
torch.testing.assert_close(W.grad[:, 1], torch.zeros(8, dtype=W.dtype))
torch.testing.assert_close(E.grad[torch.arange(8) != 2], torch.zeros(7, 2, dtype=E.dtype))
assert abs(g.sum().item()) < 1e-14
assert isclose(n["before"].item(), log(3), rel_tol=1e-12)
assert isclose((E[2] @ W.T).softmax(-1)[3].item(), .4717002842540874, rel_tol=1e-12)
assert isclose((E[4] @ W.T).softmax(-1)[3].item(), .3865220204863896, rel_tol=1e-12)
assert isclose(E[2, 0].item(), 1 + .3 * 7*log(2)/6, rel_tol=1e-12)
torch.testing.assert_close(E[4], torch.tensor([1., 1.], dtype=E.dtype))
# Finite differences verify the input-vector and output-matrix derivatives together.
h = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
w = torch.zeros(8, 2, dtype=torch.float64)
w[2, 0], w[3, 0] = log(2), log(4)
w.requires_grad_()
assert torch.autograd.gradcheck(lambda a, b: F.cross_entropy((a @ b.T)[None], torch.tensor([3])), (h, w))
# A large step is not covered by the observed small-step decrease.
h0, w0 = h.detach(), w.detach()
dh, dw = torch.autograd.grad(F.cross_entropy((h @ w.T)[None], torch.tensor([3])), (h, w))
large_step_loss = F.cross_entropy(((h0-10*dh) @ (w0-10*dw).T)[None], torch.tensor([3]))
assert torch.isfinite(large_step_loss)
print("PASS whole-vocabulary gradient, shared weights, finite differences")

bad = torch.full((1, 8), -torch.inf, dtype=torch.float64)
assert not torch.isfinite(F.cross_entropy(bad, torch.tensor([3])))
uniform = F.cross_entropy(torch.zeros(1, 8, dtype=torch.float64), torch.tensor([3])).item()
assert isclose(uniform, log(8)) and isclose(exp(uniform), 8)
for correct in [.4, .9]:
    probs = torch.full((8,), (1-correct)/7, dtype=torch.float64)
    probs[3] = correct
    assert probs.argmax() == 3
    assert isclose(F.cross_entropy(probs.log()[None], torch.tensor([3])).item(), -log(correct))
assert isclose(17/11, (4*1+3*3+4*1)/11)
assert not isclose(17/11, (1+3+1)/3)
assert 3*log(.75)+log(.25) > 3*log(.9)+log(.1)
print("PASS CE/PPL, invalid logits, same argmax, token weighting, conditional MLE")

mask = namespaces[3]
assert mask["keep"].tolist() == [[True, True, False]]
assert mask["targets"][mask["keep"]].tolist() == [2, 7]
assert torch.isfinite(mask["loss"])
assert mask["wrong_keep"].sum() == 1
assert isclose(mask["loss"].item(), log(4))
assert isclose(mask["wrong"].item(), log(8))
# Multiplication after CE does not repair NaN from an excluded row.
all_losses = F.cross_entropy(mask["logits"].detach().reshape(-1, 8),
                             mask["targets"].reshape(-1), reduction="none")
assert torch.isnan((all_losses * mask["keep"].reshape(-1)).sum())
print("PASS real EOS retained, excluded NaN removed before CE, bad ID mask detected")

# Derive the optimal conditional loss from independent text counts.
corpus = ["кот кот спит", "пёс спит", "кот спит бежит"]
rows = defaultdict(Counter)
for doc in corpus:
    seq = ["BOS", "BOS"] + doc.split() + ["EOS"]
    for t in range(2, len(seq)):
        rows[tuple(seq[t-2:t])][seq[t]] += 1
assert sum(sum(count.values()) for count in rows.values()) == 11
assert rows[("BOS", "BOS")] == {"кот": 2, "пёс": 1}
assert rows[("BOS", "кот")] == {"кот": 1, "спит": 1}
assert rows[("кот", "спит")] == {"EOS": 1, "бежит": 1}
bound = -sum(count * log(count/sum(row.values())) for row in rows.values()
             for count in row.values()) / 11
assert isclose(bound, .4256482933749291, rel_tol=1e-12)
n = namespaces[4]
model, inputs, keep = n["model"], n["inputs"], n["keep"]
assert sum(p.numel() for p in model.parameters()) == 128
assert model(inputs).shape == (3, 4, 8)
windows = F.pad(inputs, (1, 0), value=6).unfold(1, 2, 1)
assert windows[0].tolist() == [[6, 6], [6, 2], [2, 2], [2, 3]]
assert windows[2].tolist() == [[6, 6], [6, 2], [2, 3], [3, 5]]
assert isclose(n["floor"], bound, rel_tol=1e-12)
assert bound <= n["final"]+1e-8 < bound+.01
with torch.no_grad():
    probs = model(inputs).softmax(-1)
    torch.testing.assert_close(probs[0, 0, [2, 4]], torch.tensor([2/3, 1/3], dtype=torch.float64), atol=.003, rtol=0)
    torch.testing.assert_close(probs[2, 2, [5, 7]], torch.tensor([.5, .5], dtype=torch.float64), atol=.003, rtol=0)
    altered = inputs.clone()
    altered[:, 2:] = torch.tensor([[4, 4], [2, 2], [5, 5]])
    torch.testing.assert_close(model(inputs)[:, :2], model(altered)[:, :2])
    # Causality also holds before training for this window construction.
    fresh = n["WindowLM"]().double()
    torch.testing.assert_close(fresh(inputs)[:, :2], fresh(altered)[:, :2])
n["keep"] = torch.zeros_like(keep)
try:
    n["batch_loss"]()
except ValueError as error:
    assert str(error) == "Нет целевых позиций"
else:
    raise AssertionError("Empty target mask was accepted")
print("PASS counted entropy bound, 128-parameter fit, causal windows, empty-mask guard")
print("PASS all neural LM chapter checks; torch", torch.__version__, "CPU")

