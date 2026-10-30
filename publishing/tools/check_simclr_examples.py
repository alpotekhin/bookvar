"""Execute reader programs and compare the actual pinned UvA loss method."""
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений.md"
NOTEBOOK = ROOT / "05 Источники/Originals/simclr-2026-10-28/uva/SimCLR.ipynb"
text = CHAPTER.read_text()
blocks = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
assert len(blocks) == 5
namespace = {}
for code in blocks:
    exec(compile(code, str(CHAPTER), "exec"), namespace)
loss_fn = namespace["nt_xent"]
notebook = json.loads(NOTEBOOK.read_text())
wrapper = "".join(notebook["cells"][8]["source"])
assert blocks[0].startswith(wrapper + "\n")

# Run the actual source method on supplied features, not a historical trainer.
tree = ast.parse("".join(notebook["cells"][17]["source"]))
method = next(node for node in tree.body[0].body
              if isinstance(node, ast.FunctionDef) and node.name == "info_nce_loss")
source_scope = {"torch": torch, "F": F}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             "UvA cell 17 info_nce_loss", "exec"), source_scope)
source_loss = source_scope["info_nce_loss"]

cases = 0
generator = torch.Generator().manual_seed(83)
for n in [1, 2, 3, 5]:
    for dim in [2, 3, 7]:
        for temperature in [.05, .5, 2.]:
            for _ in range(3):
                x = torch.randn(2 * n, dim, generator=generator, dtype=torch.float64)
                a = x.clone().requires_grad_()
                b = x.clone().requires_grad_()
                actual, _, positive = loss_fn(a, temperature)
                rows = []
                for i in range(2 * n):
                    candidates = torch.stack([F.cosine_similarity(b[i], b[k], dim=0)
                                              for k in range(2 * n) if k != i]) / temperature
                    pos = F.cosine_similarity(b[i], b[positive[i]], dim=0) / temperature
                    rows.append(torch.logsumexp(candidates, dim=0) - pos)
                reference = torch.stack(rows).mean()
                torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-11)
                ga, = torch.autograd.grad(actual, a)
                gb, = torch.autograd.grad(reference, b)
                torch.testing.assert_close(ga, gb, rtol=1e-9, atol=1e-10)
                c = x.clone().requires_grad_()
                instance = SimpleNamespace(
                    convnet=lambda values: values,
                    hparams=SimpleNamespace(temperature=temperature),
                    log=lambda *args, **kwargs: None,
                )
                old = source_loss(instance, ([c[:n], c[n:]], None))
                gc, = torch.autograd.grad(old, c)
                torch.testing.assert_close(actual, old, rtol=1e-11, atol=1e-11)
                torch.testing.assert_close(ga, gc, rtol=1e-9, atol=1e-10)
                permutation = torch.randperm(n, generator=generator)
                reordered = torch.cat([x[:n][permutation], x[n:][permutation]])
                permuted, _, _ = loss_fn(reordered, temperature)
                torch.testing.assert_close(actual, permuted, rtol=1e-11, atol=1e-11)
                cases += 1

# Central finite differences, away from zero-vector inputs.
x = torch.tensor([[1., .3], [.2, 1.], [.8, .4], [-.1, .9]], dtype=torch.float64,
                 requires_grad=True)
analytic, = torch.autograd.grad(loss_fn(x, .7)[0], x)
finite_differences = 0
for row in range(4):
    for column in range(2):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[row, column] += 1e-6
        minus[row, column] -= 1e-6
        numerical = (loss_fn(plus, .7)[0] - loss_fn(minus, .7)[0]) / 2e-6
        torch.testing.assert_close(analytic[row, column], numerical, rtol=1e-6, atol=1e-8)
        finite_differences += 1

bad = [
    (torch.ones(3, 2), .5), (torch.ones(1, 2), .5),
    (torch.zeros(4, 2), .5), (torch.ones(4, 2, dtype=torch.int64), .5),
    (torch.full((4, 2), float("nan")), .5), (torch.ones(4, 2), 0.),
    (torch.ones(4, 2), -1.), (torch.ones(4, 2), float("inf")),
    (torch.ones(4, 2), float("nan")), (torch.empty(4, 0), .5),
]
for values, temperature in bad:
    try:
        loss_fn(values, temperature)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input was silently accepted")

assert "trainer.fit(model, train_loader, test_loader)" in "".join(notebook["cells"][35]["source"])
assert "checkpoint_callback.best_model_path" in "".join(notebook["cells"][35]["source"])
assert 8192 ** 2 * 4 == 268435456
print(json.dumps({
    "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
    "programs_executed_in_order": len(blocks),
    "independent_loss_gradient_cases": cases,
    "actual_uva_method_comparisons": cases,
    "pair_preserving_permutations": cases,
    "finite_difference_coordinates": finite_differences,
    "invalid_input_cases": len(bad),
    "source_wrapper": "unchanged class body",
    "source_test_selection": "test loader selects checkpoint; not an untouched test",
    "torch": torch.__version__,
}, indent=2))
