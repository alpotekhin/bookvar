"""Run the chapter examples and compare clipping with the pinned author code."""
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md"
SOURCE = ROOT / "05 Источники/Originals/gradient-clipping-2026-10-29"
text = CHAPTER.read_text()
addition = text.split("<!-- BEGIN REVIEW: adaptive clipping 2026-10-29 -->\n")[1]
addition = addition.split("<!-- END REVIEW: adaptive clipping 2026-10-29 -->")[0]
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\x60\x60\x60", addition, re.S)
assert len(blocks) == 2
scopes = []
for number, code in enumerate(blocks, 1):
    scope = {}
    exec(compile(code, f"{CHAPTER}:adaptive-block-{number}", "exec"), scope)
    scopes.append(scope)
agc, autoclip = scopes[0]["agc_linear_rows"], scopes[1]["autoclip_vector"]

# Execute the actual original norm helpers and AGC clipping body. The SGD
# delegate is intercepted so this test observes the clipped gradient only;
# it does not claim to run the JAX optimizer or accelerator kernels.
original = ast.parse((SOURCE / "nfnets-optim-f5de0ede.py").read_text())
functions = [n for n in original.body if isinstance(n, ast.FunctionDef)
             and n.name in ("compute_norm", "unitwise_norm")]
agc_class = next(n for n in original.body if isinstance(n, ast.ClassDef)
                 and n.name == "SGD_AGC")
update = next(n for n in agc_class.body if isinstance(n, ast.FunctionDef)
              and n.name == "update_param")
class CaptureSGD:
    @staticmethod
    def update_param(self, param, grad, state, opt_params):
        return grad, state

jnp = SimpleNamespace(
    squeeze=np.squeeze, maximum=np.maximum, where=np.where,
    sum=lambda x, axis, keepdims: np.sum(
        x, axis=tuple(axis) if isinstance(axis, list) else axis, keepdims=keepdims))
env = {"jnp": jnp, "SGD": CaptureSGD}
exec(compile(ast.Module(body=functions + [update], type_ignores=[]),
             "pinned-nfnet-agc-body", "exec"), env)
rng = np.random.default_rng(29)
cases = 0
for shape in ((2, 3), (3, 4), (4, 2)):
    for scale in (0., 1e-5, 1., 100.):
        W = torch.tensor(rng.normal(size=shape) * scale, dtype=torch.float64)
        for grad_scale in (0., 1e-7, .5, 1000.):
            G = torch.tensor(rng.normal(size=shape) * grad_scale, dtype=W.dtype)
            for threshold in (.01, .2, 1.):
                ours = agc(W, G, clipping=threshold)
                source, _ = env["update_param"](None, W.numpy().T, G.numpy().T, {},
                    {"clipping": threshold, "eps": 1e-3})
                np.testing.assert_allclose(ours.numpy(), source.T, atol=1e-13, rtol=1e-13)
                bounds = threshold * W.norm(dim=1).clamp_min(1e-3)
                assert torch.all(ours.norm(dim=1) <= bounds + 1e-12)
                cases += 1
conv = rng.normal(size=(5, 3, 2, 2))  # PyTorch OIHW.
iohw = conv.transpose(2, 3, 1, 0)    # Original HWIO.
np.testing.assert_allclose(env["unitwise_norm"](iohw).reshape(-1),
                           np.linalg.norm(conv.reshape(5, -1), axis=1))

# Compare the published percentile example with the author's actual Ignite
# handler, supplying only its event-registration interface (no Ignite install).
class Engine:
    def on(self, event):
        def register(fn):
            self.handler = fn
            return fn
        return register

original = ast.parse((SOURCE / "autoclip.py").read_text())
functions = [n for n in original.body if isinstance(n, ast.FunctionDef)]
env = {"np": np, "torch": torch,
       "BackwardsEvents": SimpleNamespace(BACKWARDS_COMPLETED="backward")}
exec(compile(ast.Module(body=functions, type_ignores=[]),
             "pinned-autoclip-handler", "exec"), env)
engine = Engine()
model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
env["add_autoclip_gradient_handler"](engine, model, 50.)
for raw, expected in zip((10., 2., 4., 100.), (10., 2., 4., 7.)):
    model.weight.grad = torch.tensor([[.6 * raw, .8 * raw]], dtype=torch.float64)
    engine.handler(None)
    assert abs(model.weight.grad.norm().item() - expected) < 1e-6

# A linear-percentile implementation independent of NumPy's percentile API.
def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])

percentile_cases = 0
for p in (0., 1., 10., 50., 90., 100.):
    history, scaled_history = [], []
    for value in (0., 2., 4., 100., .5, 25.):
        g = torch.tensor([value, 0.], dtype=torch.float64)
        out, bound = autoclip(g, history, percentile=p)
        expected = percentile(history, p)
        assert abs(bound - expected) < 1e-12
        assert abs(out.norm().item() - min(value, bound)) < 1e-12
        scaled_out, scaled_bound = autoclip(17. * g, scaled_history, percentile=p)
        torch.testing.assert_close(scaled_out, 17. * out)
        assert abs(scaled_bound - 17. * bound) < 1e-12
        percentile_cases += 1
bad_history = [2.]
try:
    autoclip(torch.tensor([float("inf")]), bad_history)
except ValueError:
    assert bad_history == [2.]
else:
    raise AssertionError("Nonfinite input must be rejected before changing history")

print(json.dumps({
    "standalone_chapter_blocks": 2, "nfnet_source_cases": cases,
    "original_hwio_norm_check": "PASS", "original_autoclip_handler": "PASS",
    "independent_percentile_and_scaling_cases": percentile_cases,
    "nonfinite_history_guard": "PASS", "device": "cpu", "gpu_tested": False,
    "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
}))
