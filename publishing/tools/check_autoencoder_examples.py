"""Execute the AE/VAE chapter and independently check its examples and source fidelity."""
from pathlib import Path
from itertools import product
import ast
import hashlib
import inspect
import json
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 14
scope = {}
for number, code in enumerate(blocks, 1):
    print(f"Execute actual AE/VAE block {number}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{number}", "exec"), scope)

sources = ROOT / "05 Источники/Originals/autoencoder-2026-10-26"
def read_pinned(name, digest):
    data = (sources / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, name
    return data.decode()

nb = json.loads(read_pinned("uva-ae.ipynb",
    "ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be"))
uva_code = "\n\n".join("".join(c["source"]) for c in nb["cells"]
    if c["cell_type"] == "code" and any("class " + name + "(" in "".join(c["source"])
                                       for name in ["Encoder", "Decoder"]))
pytorch_code = read_pinned("pytorch-vae.py",
    "366116e8252ac68ef1f180cce0b3c017f5a957595d863c8d4904210a80cdab6d")
normalize = lambda text: "\n".join(line.rstrip() for line in text.splitlines())
actual_tree = ast.parse(normalize("\n\n".join(blocks)))
for source_code, name, kind in [(uva_code, "Encoder", ast.ClassDef),
        (uva_code, "Decoder", ast.ClassDef), (pytorch_code, "VAE", ast.ClassDef),
        (pytorch_code, "loss_function", ast.FunctionDef)]:
    original = next(n for n in ast.parse(normalize(source_code)).body if isinstance(n, kind) and n.name == name)
    actual = next(n for n in actual_tree.body if isinstance(n, kind) and n.name == name)
    assert ast.dump(actual) == ast.dump(original), name
    # AST checks semantics/docstrings; this also preserves original comments and spelling.
    source_segment = ast.get_source_segment(source_code, original)
    assert normalize(source_segment) in normalize("\n\n".join(blocks)), name
before = ROOT / "docs/audit/snapshots/2026-10-26-autoencoder-before.md"
assert hashlib.sha256(before.read_bytes()).hexdigest() == "091ab377baf722ca7e5a917c77a614224a73290ef77e531aa981c5ca3cbd4900"
old_blocks = re.findall(r"```python\n(.*?)```", before.read_text(), re.S)
assert len(old_blocks) == 2
assert all(any(ast.dump(ast.parse(old)) == ast.dump(ast.parse(new)) for new in blocks) for old in old_blocks)
for source_name, public_name in [
        ("uva-MIT.txt", "uva-c7d9e18f-MIT.txt"),
        ("pytorch-examples-LICENSE.txt", "pytorch-examples-acc295dc-BSD-3-Clause.txt"),
        ("conv-arithmetic-MIT.txt", "conv-arithmetic-af6f818b-MIT.txt")]:
    assert (sources / source_name).read_bytes() == (ROOT / "site/public/licenses" / public_name).read_bytes()
print("PASS source classes/loss modulo trailing whitespace, both preserved old examples, and three complete license copies")

dtype = torch.float64
conjugate_cases = 0
for x, prior_var, noise_var, mq, vq in product(
        [-3., -1., 0., 2.], [.5, 1., 3.], [.25, 1., 2.], [-2., .5, 2.], [.2, 1., 4.]):
    post_var = prior_var * noise_var / (prior_var + noise_var)
    post_mean = prior_var / (prior_var + noise_var) * x
    rec = -.5 * math.log(2 * math.pi * noise_var) - .5 * ((x - mq)**2 + vq) / noise_var
    prior_kl = scope["normal_kl"](mq, vq, 0., prior_var)
    posterior_gap = scope["normal_kl"](mq, vq, post_mean, post_var)
    log_evidence = Normal(torch.tensor(0., dtype=dtype),
        torch.tensor(math.sqrt(prior_var + noise_var), dtype=dtype)).log_prob(torch.tensor(x, dtype=dtype)).item()
    q = Normal(torch.tensor(mq, dtype=dtype), torch.tensor(math.sqrt(vq), dtype=dtype))
    for mean_p, var_p, expected in [(0., prior_var, prior_kl), (post_mean, post_var, posterior_gap)]:
        p = Normal(torch.tensor(mean_p, dtype=dtype), torch.tensor(math.sqrt(var_p), dtype=dtype))
        assert abs(kl_divergence(q, p).item() - expected) < 1e-12
    assert abs(log_evidence - (rec - prior_kl) - posterior_gap) < 1e-12
    conjugate_cases += 1
print(f"PASS {conjugate_cases} conjugate ELBO identities and two library KL comparisons per case")

def fixed_objective(mu, logvar):
    z = mu + (.5 * logvar).exp() * scope["fixed_epsilon"]
    logits = torch.cat((z, -z), dim=-1)
    return F.binary_cross_entropy_with_logits(logits, scope["binary_observation"], reduction="sum") + \
        .5 * (mu.square() + logvar.exp() - 1 - logvar).sum()
mu = scope["fixed_mu"].detach().requires_grad_()
lv = scope["fixed_logvar"].detach().requires_grad_()
assert torch.autograd.gradcheck(fixed_objective, (mu, lv))
analytic = torch.autograd.grad(fixed_objective(mu, lv), (mu, lv))
h = 1e-5
for index in range(2):
    plus = [mu.detach().clone(), lv.detach().clone()]
    minus = [mu.detach().clone(), lv.detach().clone()]
    plus[index] += h
    minus[index] -= h
    numerical = (fixed_objective(*plus) - fixed_objective(*minus)) / (2*h)
    torch.testing.assert_close(numerical, analytic[index].squeeze(), atol=1e-9, rtol=1e-9)
mu = torch.tensor([.5], dtype=dtype, requires_grad=True)
lv = torch.tensor([math.log(4)], dtype=dtype, requires_grad=True)
dist = Normal(mu, (.5*lv).exp())
assert not dist.sample().requires_grad
assert dist.rsample().requires_grad
mu_grad, lv_grad = torch.autograd.grad(dist.rsample().sum(), (mu, lv))
torch.testing.assert_close(mu_grad, torch.ones_like(mu))
assert torch.isfinite(lv_grad).all()
assert "no_grad" in inspect.getsource(Normal.sample)
assert "self.loc + eps * self.scale" in inspect.getsource(Normal.rsample)
print("PASS fixed-noise autograd, independent central differences, sample/rsample execution and installed implementation")

geometry_cases = 0
gen = torch.Generator().manual_seed(26)
for h, w, k, stride, pad, dilation in product([3, 5], [4, 6], [1, 3], [1, 2, 3], [0, 1], [1, 2]):
    oh = (h + 2*pad - dilation*(k-1) - 1)//stride + 1
    ow = (w + 2*pad - dilation*(k-1) - 1)//stride + 1
    if min(oh, ow) <= 0:
        continue
    image = torch.randn(1, 2, h, w, generator=gen, dtype=dtype)
    kernel = torch.randn(3, 2, k, k, generator=gen, dtype=dtype)
    forward = F.conv2d(image, kernel, stride=stride, padding=pad, dilation=dilation)
    dual = torch.randn(forward.shape, generator=gen, dtype=dtype)
    base_h = (oh - 1)*stride - 2*pad + dilation*(k - 1) + 1
    base_w = (ow - 1)*stride - 2*pad + dilation*(k - 1) + 1
    op = (h - base_h, w - base_w)
    assert all(0 <= v < stride for v in op)
    transpose = F.conv_transpose2d(dual, kernel, stride=stride, padding=pad,
                                 dilation=dilation, output_padding=op)
    assert transpose.shape == image.shape
    torch.testing.assert_close((forward*dual).sum(), (image*transpose).sum(), atol=1e-10, rtol=1e-10)
    geometry_cases += 1
print(f"PASS {geometry_cases} rectangular/multichannel transpose shapes and adjoint identities")

# A full UvA encoder/decoder gradient step, independently of the chapter's shape-only pass.
image_encoder, image_decoder = scope["image_encoder"], scope["image_decoder"]
image = torch.rand(2, 3, 32, 32, generator=gen, dtype=dtype)*2 - 1
ae_parameters = list(image_encoder.parameters()) + list(image_decoder.parameters())
weights_before = [p.detach().clone() for p in ae_parameters]
optimizer = torch.optim.SGD(ae_parameters, lr=1e-4)
optimizer.zero_grad()
ae_loss = (image_decoder(image_encoder(image)) - image).square().flatten(1).sum(-1).mean()
ae_loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ae_parameters)
optimizer.step()
assert any(not torch.equal(a, b) for a,b in zip(weights_before, ae_parameters))
assert image_encoder(image).shape == (2, 16)
for candidate, expected in [(torch.tensor([1.,0.]),1.), (torch.tensor([0.,1.]),1.),
                             (torch.tensor([.5,.5]),.5)]:
    targets = torch.tensor([[1.,0.],[0.,1.]])
    assert (targets - candidate).square().sum(-1).mean().item() == expected
assert (torch.tensor([1.,1.])-torch.tensor([0.,1.])).square().sum().item() == 1
assert (torch.tensor([1.,1.])-torch.tensor([1.,1.])).square().sum().item() == 0
mu, lv = scope["original_mu"].detach(), scope["original_logvar"].detach()
p, x = scope["original_reconstruction"].detach(), scope["original_binary_batch"]
manual_per_item = F.binary_cross_entropy(p,x.flatten(1),reduction="none").sum(-1) + \
    .5 * (mu.square() + lv.exp() - 1 - lv).sum(-1)
torch.testing.assert_close(scope["loss_function"](p,x,mu,lv), manual_per_item.sum())
assert not torch.isclose(scope["binary_rec"].mean()/2 + scope["binary_kl"],
                         scope["binary_rec"].mean() + scope["binary_kl"]).all()
assert not torch.equal(scope["stochastic_reconstruction1"], scope["stochastic_reconstruction2"])
# Nonlinear decoding of a mean is not the mean of decodings.
assert (torch.tensor([-1.,1.]).mean().square() != torch.tensor([-1.,1.]).square().mean()).item()
print("PASS full UvA gradient step, denoising/MSE examples, loss reductions, stochastic eval and nonlinear expectation")
torch.testing.assert_close(scope["interpolation_codes"][0], scope["eval_mu"][0])
torch.testing.assert_close(scope["interpolation_codes"][-1], scope["eval_mu"][1])
torch.testing.assert_close(scope["interpolation_codes"][2], scope["eval_mu"].mean(0))
assert scope["interpolation_codes"].shape == (5, 20)
assert scope["interpolation_probabilities"].shape == (5, 784)
assert ((torch.tensor([0., 0.]) + torch.tensor([2., 4.])) / 2).tolist() == [1., 2.]
assert .5**2 + .5**2 == .5
collapsed_mu = torch.zeros(2, 1, dtype=dtype)
collapsed_logvar = torch.zeros_like(collapsed_mu)
collapsed_kl = .5 * (collapsed_mu.square() + collapsed_logvar.exp() - 1 - collapsed_logvar).sum(-1)
collapsed_targets = torch.tensor([[1.,0.],[0.,1.]], dtype=dtype)
collapsed_reconstruction = F.binary_cross_entropy_with_logits(
    torch.zeros_like(collapsed_targets),collapsed_targets,reduction="none").sum(-1)
torch.testing.assert_close(collapsed_kl, torch.zeros(2, dtype=dtype))
torch.testing.assert_close(collapsed_reconstruction, torch.full((2,),2*math.log(2),dtype=dtype))
print("PASS interpolation endpoints/midpoint, prior variance change and the explicit zero-KL collapse case")
print("TORCH", torch.__version__)
print("CHAPTER SHA256", hashlib.sha256(content).hexdigest())
