"""Execute the current MLP chapter and check its forward/training mechanisms."""
from contextlib import redirect_stdout
from io import StringIO
from itertools import permutations
from pathlib import Path
import copy
import hashlib
import json
import math
import re

import numpy as np
import torch
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md"
chapter_text = CHAPTER.read_text()
chapter_text = re.sub(r'<!-- BEGIN REVIEW: Module hooks 2026-10-27 -->\n.*?'
    r'<!-- END REVIEW: Module hooks 2026-10-27 -->\n\n', '', chapter_text, flags=re.S)
blocks = re.findall(r"```python\n(.*?)```", chapter_text, re.S)
assert len(blocks) == 7
original_text = re.sub(r'<!-- BEGIN REVIEW: Module state 2026-10-27 -->\n.*?'
    r'<!-- END REVIEW: Module state 2026-10-27 -->\n\n', '', chapter_text, flags=re.S)
original_text = original_text.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-21', 1)
for reflow in json.loads((ROOT / 'publishing/tools/mlp-review-layout.json').read_text())['reflows']:
    assert original_text.count(reflow['after']) == 1
    original_text = original_text.replace(reflow['after'], reflow['before'], 1)
assert hashlib.sha256(original_text.encode()).hexdigest() == \
    '2fd8e90374a5802bd2c3997d36ee2d945e9aa706dc2e2810c023b39a4a695e05'
frameworks = (ROOT / '05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
source_custom_layer = re.search(r'\{#lst-parameter_registration.*?```\{\.python\}\n(.*?)```',
                               frameworks, re.S).group(1)
assert blocks[2] == source_custom_layer
scope = {}
with redirect_stdout(StringIO()) as captured:
    for i, code in enumerate(blocks, 1):
        exec(compile(code, str(CHAPTER) + f":block{i}", "exec"), scope)
print(captured.getvalue(), end="")

np.testing.assert_array_equal(scope["H_xor"], [[0, 0], [1, 0], [1, 0], [2, 1]])
np.testing.assert_array_equal(scope["o_xor"] > .5, [False, True, True, False])
# Finite strict separation would allow rescaling positive scores to margin >= 1.
# These inequalities are infeasible even allowing the negative class score == 0.
constraints = np.array([[0, -1, -1], [-1, 0, -1], [0, 0, 1], [1, 1, 1.]])
fit = linprog(np.zeros(3), A_ub=constraints, b_ub=[-1, -1, 0, 0],
              bounds=[(None, None)] * 3, method="highs")
assert fit.status == 2
outside = np.maximum(np.array([[2., 2.]]) @ scope["W1_xor"] + scope["b1_xor"], 0)
assert (outside @ scope["W2_xor"]).item() == -2

X = scope["X"]
W1 = scope["W1"]
W2 = scope["W2"]
b1 = torch.tensor([0., 1., -1.], dtype=torch.float64)
Z = X @ W1 + b1
H = Z.relu()
torch.testing.assert_close(Z, torch.tensor([[5., 2., -1.], [0., -2., 4.]], dtype=torch.float64))
torch.testing.assert_close(H, torch.tensor([[5., 2., 0.], [0., 0., 4.]], dtype=torch.float64))
torch.testing.assert_close(scope["logits"], torch.tensor([[1.2, -.8], [-1., 1.]], dtype=torch.float64))
p = torch.softmax(scope["logits"], dim=-1)
q = 1 / (1 + math.exp(2))
np.testing.assert_allclose(p.detach().numpy(), [[1-q, q], [q, 1-q]], atol=1e-15)
np.testing.assert_allclose(scope["before"], np.logaddexp(0, -2), atol=1e-15)
np.testing.assert_allclose(scope["grad"], -2.5*q, atol=1e-15)
np.testing.assert_allclose(scope["model"].fc2.weight[0, 0].item(), .2 + .25*q, atol=1e-15)

# Central difference is taken in W2, where ReLU's kink at Z[1,0]=0 is irrelevant.
def loss_with_weight(value):
    w = W2.clone()
    w[0, 0] = value
    return torch.nn.functional.cross_entropy(H @ w, scope["y"]).item()

epsilon = 1e-6
finite_difference = (loss_with_weight(.2+epsilon)-loss_with_weight(.2-epsilon))/(2*epsilon)
np.testing.assert_allclose(finite_difference, scope["grad"], rtol=1e-8)

# A second softmax changes the loss; rowwise common-logit shifts do not.
raw_logits = scope["logits"].detach()
double_softmax = torch.nn.functional.cross_entropy(raw_logits.softmax(-1), scope["y"])
assert double_softmax.item() > scope["before"] + .2
torch.testing.assert_close(raw_logits.softmax(-1), (raw_logits+1000).softmax(-1))

model = scope["model"]
assert sum(p.numel() for p in model.parameters()) == (2*3+3)+(3*2+2) == 17
pictured = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
assert sum(p.numel() for p in pictured.parameters()) == 41

# Affine collapse for nonzero biases, and a strict intermediate-rank limit.
rng = np.random.default_rng(4)
for batch, din, hidden, dout in [(1, 2, 3, 2), (7, 4, 2, 6), (100, 3, 7, 1)]:
    xx = rng.normal(size=(batch, din))
    a = rng.normal(size=(din, hidden)); b = rng.normal(size=hidden)
    c = rng.normal(size=(hidden, dout)); d = rng.normal(size=dout)
    np.testing.assert_allclose((xx@a+b)@c+d, xx@(a@c)+(b@c+d), atol=1e-12)
    assert np.linalg.matrix_rank(a@c) <= hidden
x = np.array([-2., -.5, 0, 2])
np.testing.assert_allclose(3*(2*x+1)-4, 6*x-1)
np.testing.assert_allclose(3*np.maximum(0, 2*x+1)-4, np.where(x <= -.5, -4, 6*x-1))

# Hidden coordinates can be permuted without changing the computed function.
for perm in permutations(range(3)):
    ids = list(perm)
    torch.testing.assert_close(H[:, ids] @ W2[ids, :], H @ W2)

# FFN preserves leading axes, shares 17 parameters, and never mixes positions.
tokens = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)/5
with torch.no_grad():
    whole = model(tokens)
    separate = torch.stack([model(row) for row in tokens.reshape(-1, 2)]).reshape(2, 3, 2)
    torch.testing.assert_close(whole, separate)
    changed = tokens.clone(); changed[1, 2] += torch.tensor([3., -4.])
    after = model(changed)
    torch.testing.assert_close(after[:1], whole[:1])
    torch.testing.assert_close(after[1, :2], whole[1, :2])
    assert not torch.allclose(after[1, 2], whole[1, 2])
    order = [2, 0, 1]
    torch.testing.assert_close(model(tokens[:, order]), whole[:, order])
assert whole.shape == (2, 3, 2) and sum(p.numel() for p in model.parameters()) == 17
original_sequence = torch.tensor([[1., 2.], [2., -1.], [1., 2.]], dtype=torch.float64)
torch.testing.assert_close((original_sequence @ W1 + b1).relu() @ W2,
    torch.tensor([[1.2, -.8], [-1., 1.], [1.2, -.8]], dtype=torch.float64))
replaced = original_sequence.clone(); replaced[1] = replaced[0]
torch.testing.assert_close((replaced @ W1 + b1).relu() @ W2,
    torch.tensor([[1.2, -.8], [1.2, -.8], [1.2, -.8]], dtype=torch.float64))
np.testing.assert_allclose(scope["after"], .098018, atol=.5e-6)

# Backward accumulates without updating; zero_grad clears; ReLU chooses zero at its kink.
replica = copy.deepcopy(model)
before_params = [p.detach().clone() for p in replica.parameters()]
replica.zero_grad()
replica(X).sum().backward()
first_grads = [p.grad.clone() for p in replica.parameters()]
replica(X).sum().backward()
for param, original, first_grad in zip(replica.parameters(), before_params, first_grads):
    torch.testing.assert_close(param, original)
    torch.testing.assert_close(param.grad, 2*first_grad)
replica.zero_grad()
assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in replica.parameters())
zero = torch.tensor(0., requires_grad=True)
zero.relu().backward()
assert zero.grad.item() == 0.
assert scope["after"] < scope["before"]

# Module registration is separate from participation in autograd and an optimizer.
CustomLayer = scope['CustomLayer']
probe = scope['state_probe']
assert repr(scope['model']) == '''MLP(
  (fc1): Linear(in_features=2, out_features=3, bias=True)
  (fc2): Linear(in_features=3, out_features=2, bias=True)
)'''
assert sum(p.numel() for p in scope['layer'].parameters()) == 220
assert list(dict(scope['layer'].named_buffers())) == ['running_mean']
assert sum(b.numel() for b in scope['layer'].buffers()) == 20
assert list(dict(probe.named_parameters())) == ['weight', 'bias']
assert list(dict(probe.named_buffers())) == ['running_mean', 'scratch']
assert scope['probe_loss'].item() == 11
torch.testing.assert_close(probe.weight.grad, torch.tensor([[1., 2.], [1., 2.]], dtype=torch.float64))
assert probe.running_mean.grad is None and torch.count_nonzero(probe.running_mean) == 0
assert {id(p) for g in scope['probe_optimizer'].param_groups for p in g['params']} == \
       {id(probe.weight), id(probe.bias)}

# A buffer can require a gradient without becoming a parameter.
buffer_probe = CustomLayer(2, 2).double()
buffer_probe.running_mean.requires_grad_()
buffer_optimizer = torch.optim.SGD(buffer_probe.parameters(), lr=.1)
buffer_probe.running_mean.sum().backward()
torch.testing.assert_close(buffer_probe.running_mean.grad, torch.ones(2, dtype=torch.float64))
buffer_optimizer.step()
torch.testing.assert_close(buffer_probe.running_mean, torch.zeros(2, dtype=torch.float64))

# Freezing does not erase a gradient computed earlier.
stale = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
stale_optimizer = torch.optim.SGD([stale], lr=.1)
(3 * stale).backward()
stale.requires_grad_(False)
stale_optimizer.step()
torch.testing.assert_close(stale, torch.tensor(1.7, dtype=torch.float64))
stale.grad = None
stale_optimizer.step()
torch.testing.assert_close(stale, torch.tensor(1.7, dtype=torch.float64))

# Unregistered layers still run and differentiate. Recursive module APIs miss them.
LayerStack = scope['LayerStack']
unregistered = LayerStack([CustomLayer(2, 2)], register=False)
with torch.no_grad():
    unregistered.blocks[0].weight.fill_(1)
    unregistered.blocks[0].bias.fill_(1)
unregistered(torch.ones(1, 2)).sum().backward()
torch.testing.assert_close(unregistered.blocks[0].weight.grad, torch.ones(2, 2))
assert list(unregistered.parameters()) == []
unregistered.eval().double()
assert unregistered.blocks[0].training and unregistered.blocks[0].weight.dtype == torch.float32
try:
    torch.optim.SGD(unregistered.parameters(), lr=.1)
except ValueError as error:
    assert 'empty parameter list' in str(error)
else:
    raise AssertionError('unregistered parameters unexpectedly reached SGD')
registered = LayerStack([CustomLayer(2, 2)], register=True).eval().double()
assert not registered.blocks[0].training
assert all(t.dtype == torch.float64 for t in registered.state_dict().values())
assert all(a is b for a, b in zip(scope['plain_stack'].blocks, scope['registered_stack'].blocks))

# Reusing one registered layer twice does not create another set of parameters.
shared_layer = CustomLayer(2, 2)
shared_stack = LayerStack([shared_layer, shared_layer], register=True)
assert sum(p.numel() for p in shared_stack.parameters()) == 6
assert shared_stack.blocks[0].weight is shared_stack.blocks[1].weight

# Tensor values restore; construction, grad flags, modes and temporary attributes do not.
restored = scope['restored_layer']
assert restored.training and restored.bias.requires_grad
assert not probe.bias.requires_grad and not hasattr(restored, 'scratch')
assert list(restored.state_dict()) == list(probe.state_dict())
for name, value in restored.state_dict().items():
    torch.testing.assert_close(value, probe.state_dict()[name])
try:
    CustomLayer(3, 2).double().load_state_dict(scope['loaded_state'], strict=False)
except RuntimeError as error:
    assert 'size mismatch' in str(error)
else:
    raise AssertionError('strict=False incorrectly accepted a different weight shape')
missing = dict(scope['loaded_state']); missing.pop('bias'); missing['extra'] = torch.zeros(1)
keys = CustomLayer(2, 2).double().load_state_dict(missing, strict=False)
assert keys.missing_keys == ['bias'] and keys.unexpected_keys == ['extra']
live = restored.state_dict()
snapshot = copy.deepcopy(live)
with torch.no_grad():
    restored.weight.add_(1)
torch.testing.assert_close(live['weight'], restored.weight)
torch.testing.assert_close(snapshot['weight'] + 1, restored.weight)

# The standard module call executes hooks; a direct forward call bypasses them.
hook_calls = []
handle = scope['mode_probe'].register_forward_hook(lambda *args: hook_calls.append('forward'))
scope['mode_probe'](scope['mode_input'])
scope['mode_probe'].forward(scope['mode_input'])
handle.remove()
assert hook_calls == ['forward']

print("PASS: all seven published blocks; unchanged accepted MLP body; verbatim Harvard example; "
      "registration/buffers/dtype/frozen and stale gradients; ModuleList/list/autograd; "
      "shared parameters; checkpoint roundtrip/strict/shallow state; eval/no_grad/hooks; "
      "XOR impossibility/construction; full forward/CE/SGD; "
      "finite difference; affine collapse/rank; parameter counts; hidden permutations; "
      "token-wise FFN; gradient accumulation and ReLU at zero.")
print('Chapter SHA-256:', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
print("PyTorch:", torch.__version__)

