"""Run chapter examples and independently verify dropout and regularization arithmetic."""
from pathlib import Path
from itertools import product
import ast
import hashlib
import math
import re
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 11
scope = {}
for number, code in enumerate(blocks, 1):
    print(f"Execute actual dropout block {number}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{number}", "exec"), scope)

dropout = scope["dropout"]
for p in (-1., 1., 1.1, float("nan"), float("inf")):
    for training in (True, False):
        try:
            dropout(torch.ones(2), p, training)
        except ValueError:
            pass
        else:
            raise AssertionError((p, training))
for p in (0., .1, .5, .999):
    x = torch.randn(2, 3, dtype=torch.float64)
    assert dropout(x, p, False) is x
    assert dropout(x, 0., True) is x
library_one = nn.Dropout(1.)
assert torch.equal(library_one(torch.ones(3)), torch.zeros(3))
library_one.eval()
assert torch.equal(library_one(torch.ones(3)), torch.ones(3))

source_file = ROOT / "05 Источники/Originals/dropout-2026-10-26/d2l-dropout.md"
assert hashlib.sha256(source_file.read_bytes()).hexdigest() == "e5fd93ac9cca9ad6a563f216df28af485ea885b07a4f8e0ce874c5045a619cd3"
source = source_file.read_text()
source_code = re.search(r"%%tab pytorch\n(def dropout_layer\(.*?)(?=\n```)", source, re.S).group(1)
tree = ast.parse(source_code)
assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
source_scope = {"torch": torch}
exec(compile(tree, "pinned-d2l-dropout-layer", "exec"), source_scope)
comparisons = 0
for seed in range(100):
    for shape in [(1,), (3, 4), (2, 3, 4, 4)]:
        x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 1
        for p in (.25, .5, .75):
            torch.manual_seed(seed)
            original = source_scope["dropout_layer"](x, p)
            torch.manual_seed(seed)
            actual = dropout(x, p, True)
            torch.testing.assert_close(actual, original)
            comparisons += 1
print(f"PASS {comparisons} pinned D2L float32 helper comparisons and p-boundary contracts")

generator = torch.Generator().manual_seed(20261026)
cases = 0
for dimensions in range(1, 6):
    masks = torch.tensor(list(product([0., 1.], repeat=dimensions)), dtype=torch.float64)
    for p in (0., .1, .25, .5, .8, .95):
        q = 1 - p
        probabilities = torch.where(masks == 1, torch.full_like(masks, q), torch.full_like(masks, p))
        weights = torch.prod(probabilities, dim=1)
        for _ in range(10):
            h = torch.randn(dimensions, generator=generator, dtype=torch.float64)
            w = torch.randn(dimensions, generator=generator, dtype=torch.float64, requires_grad=True)
            target, bias = .7, -.3
            hidden = masks * h / q
            exact_mean = weights @ hidden
            exact_var = weights @ (hidden - h).square()
            torch.testing.assert_close(weights.sum(), torch.tensor(1., dtype=h.dtype))
            torch.testing.assert_close(exact_mean, h, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(exact_var, (p / q) * h.square(), atol=1e-12, rtol=1e-12)
            enumerated_loss = weights @ (target - hidden @ w - bias).square()
            closed_loss = (target - h @ w - bias).square() + (p / q) * (w * h).square().sum()
            torch.testing.assert_close(enumerated_loss, closed_loss, atol=1e-12, rtol=1e-12)
            gradient, = torch.autograd.grad(enumerated_loss, w)
            expected = 2 * h * (h @ w.detach() + bias - target) + 2 * (p / q) * w.detach() * h.square()
            torch.testing.assert_close(gradient, expected, atol=1e-11, rtol=1e-11)
            cases += 1
print(f"PASS {cases} exact nonuniform-mask means, variances, losses and gradients")

for seed in range(30):
    torch.manual_seed(seed)
    x = (torch.randn(2, 5, dtype=torch.float64) + 2).requires_grad_()
    upstream_gradient = torch.randn_like(x)
    output = dropout(x, .25, True)
    gradient, = torch.autograd.grad((output * upstream_gradient).sum(), x)
    coefficient = torch.where(output != 0, torch.full_like(x, 1 / .75), torch.zeros_like(x))
    torch.testing.assert_close(gradient, upstream_gradient * coefficient)

class WrongFunctional(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=.5)

class ModeAwareFunctional(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=.5, training=self.training)

torch.manual_seed(26)
assert not torch.equal(WrongFunctional().eval()(torch.ones(64)), torch.ones(64))
assert torch.equal(ModeAwareFunctional().eval()(torch.ones(64)), torch.ones(64))
parameter = nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=1.)
optimizer.step()
torch.testing.assert_close(parameter, torch.tensor([1., 2.], dtype=parameter.dtype))
parameter.grad = torch.zeros_like(parameter)
optimizer.step()
torch.testing.assert_close(parameter, torch.tensor([.9, 1.8], dtype=parameter.dtype))

model = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    model.weight.fill_(1.)
shallow = model.state_dict()
snapshot = {key: value.detach().clone() for key, value in shallow.items()}
with torch.no_grad():
    model.weight.fill_(2.)
assert shallow["weight"].item() == 2. and snapshot["weight"].item() == 1.
assert scope["stopped_epoch"] == 5 and scope["best_epoch"] == 3
assert scope["checkpoint_model"].weight.item() == 3
print("PASS 30 actual random-mask backward passes, functional mode, zero/None gradients and checkpoint aliases")
print("CHAPTER SHA256", hashlib.sha256(content).hexdigest())
