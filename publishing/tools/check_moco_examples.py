"""Check reader code and the pinned MoCo class with explicit CPU adapters."""
import ast
import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder.md"
SOURCE = ROOT / "05 Источники/Originals/moco-2026-10-28/moco/builder.py"
text = CHAPTER.read_text()
blocks = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
assert len(blocks) == 6
reader = {}
for code in blocks:
    exec(compile(code, str(CHAPTER), "exec"), reader)
enqueue, logits_fn = reader["enqueue"], reader["moco_logits"]

source_scope = {}
exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), source_scope)
OriginalMoCo = source_scope["MoCo"]
source_tree = ast.parse(SOURCE.read_text())
source_class = next(n for n in source_tree.body if isinstance(n, ast.ClassDef))
original_ema = next(n for n in source_class.body if getattr(n, "name", "") == "_momentum_update_key_encoder")
reader_class = next(n for n in ast.parse(blocks[0]).body if isinstance(n, ast.ClassDef))
reader_ema = next(n for n in reader_class.body if getattr(n, "name", "") == "_momentum_update_key_encoder")
assert ast.dump(original_ema, include_attributes=False) == ast.dump(reader_ema, include_attributes=False)

# Only the execution environment is adapted: one local gather, identity
# shuffle/unshuffle, and .cuda() retaining CPU placement. The source forward,
# EMA and enqueue methods execute unchanged. There is no distributed/BN test.
source_scope["concat_all_gather"] = lambda values: values
torch.manual_seed(47)
source_steps = 0
for batch in [1, 2, 3]:
    for dim in [2, 3, 4]:
        for temperature in [.07, .2, 1.]:
            for momentum in [0., .9, .999]:
                model = OriginalMoCo(
                    lambda num_classes: torch.nn.Linear(2, num_classes, bias=False),
                    dim=dim, K=6, m=momentum, T=temperature,
                ).double()
                model._batch_shuffle_ddp = lambda x: (x, torch.arange(len(x)))
                model._batch_unshuffle_ddp = lambda x, index: x
                with torch.no_grad():
                    model.encoder_q.weight.add_(.03 * torch.randn_like(model.encoder_q.weight))
                for _ in range(3):
                    xq = torch.randn(batch, 2, dtype=torch.float64)
                    xk = xq + .05 * torch.randn_like(xq)
                    before_q = model.encoder_q.weight.detach().clone()
                    before_k = model.encoder_k.weight.detach().clone()
                    before_queue = model.queue.detach().clone()
                    before_pointer = int(model.queue_ptr.item())
                    expected_k = momentum * before_k + (1-momentum) * before_q
                    reference_q = before_q.clone().requires_grad_()
                    q = F.normalize(xq @ reference_q.T, dim=1)
                    k = F.normalize(xk @ expected_k.T, dim=1)
                    expected_logits = logits_fn(q, k, before_queue, temperature)
                    reference_loss = F.cross_entropy(expected_logits, torch.zeros(batch, dtype=torch.long))
                    model.zero_grad(set_to_none=True)
                    with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
                        actual_logits, targets = model(xq, xk)
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-10, atol=1e-10)
                    assert targets.tolist() == [0] * batch
                    actual_loss = F.cross_entropy(actual_logits, targets)
                    actual_loss.backward()
                    reference_loss.backward()
                    torch.testing.assert_close(model.encoder_q.weight.grad, reference_q.grad, rtol=1e-9, atol=1e-9)
                    assert model.encoder_k.weight.grad is None
                    torch.testing.assert_close(model.encoder_k.weight, expected_k, rtol=1e-12, atol=1e-12)
                    expected_queue = before_queue.clone()
                    next_pointer = enqueue(expected_queue, before_pointer, k)
                    torch.testing.assert_close(model.queue, expected_queue, rtol=1e-12, atol=1e-12)
                    assert int(model.queue_ptr.item()) == next_pointer
                    with torch.no_grad():
                        model.encoder_q.weight.add_(model.encoder_q.weight.grad, alpha=-.03)
                    source_steps += 1

# Exercise the original optional v2 MLP constructor and forward branch too.
class SmallBackbone(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Linear(2, 4)
        self.fc = torch.nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).tanh())

v2 = OriginalMoCo(SmallBackbone, dim=3, K=6, m=.9, T=.2, mlp=True).double()
assert isinstance(v2.encoder_q.fc, torch.nn.Sequential) and len(v2.encoder_q.fc) == 3
for q, k in zip(v2.encoder_q.parameters(), v2.encoder_k.parameters()):
    torch.testing.assert_close(q, k)
    assert not k.requires_grad
v2._batch_shuffle_ddp = lambda x: (x, torch.arange(len(x)))
v2._batch_unshuffle_ddp = lambda x, index: x
with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
    value, target = v2(torch.randn(2, 2, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64))
F.cross_entropy(value, target).backward()
assert value.shape == (2, 7) and int(v2.queue_ptr.item()) == 2
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in v2.encoder_q.parameters())
assert all(p.grad is None for p in v2.encoder_k.parameters())

restricted = OriginalMoCo(lambda num_classes: torch.nn.Linear(2, num_classes), dim=3, K=5)
before = restricted.queue.clone()
try:
    restricted._dequeue_and_enqueue(torch.ones(2, 3))
except AssertionError:
    pass
else:
    raise AssertionError("source enqueue should reject K not divisible by global batch")
torch.testing.assert_close(restricted.queue, before)
assert restricted.queue_ptr.item() == 0

ring_cases = 0
for capacity in [1, 2, 5, 7]:
    for dim in [1, 3]:
        for pointer in range(capacity):
            for batch in range(1, capacity+1):
                initial = torch.arange(dim*capacity, dtype=torch.float64).reshape(dim, capacity)
                keys = 100 + torch.arange(batch*dim, dtype=torch.float64).reshape(batch, dim)
                actual, expected = initial.clone(), initial.clone()
                end = enqueue(actual, pointer, keys)
                for j in range(batch):
                    expected[:, (pointer+j) % capacity] = keys[j]
                torch.testing.assert_close(actual, expected)
                assert end == (pointer+batch) % capacity
                ring_cases += 1

ema_cases = 0
for momentum in [0., .5, .9, .999]:
    state = torch.zeros(5, dtype=torch.float64)
    history = torch.randn(8, 5, dtype=torch.float64)
    for t, value in enumerate(history, start=1):
        state = momentum * state + (1-momentum) * value
        expanded = (1-momentum) * sum(momentum**r * history[t-r-1] for r in range(t))
        torch.testing.assert_close(state, expanded, rtol=1e-11, atol=1e-12)
        ema_cases += 1

permutation_cases = 0
for count in [2, 4, 6]:
    ids = torch.arange(count)
    for order in itertools.permutations(range(count)):
        permutation = torch.tensor(order)
        assert torch.equal(ids[permutation][permutation.argsort()], ids)
        permutation_cases += 1

assert 128*65536*4 == 33554432 and 256*65537*4 == 67109888
assert abs(math.log(.5)/math.log(.999) - 692.80055) < .001
assert abs(.999**256 - .774) < .001
assert 65536//256 == 256
print(json.dumps({
    "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
    "programs_executed_in_order": len(blocks),
    "source_forward_loss_gradient_queue_steps": source_steps,
    "source_v2_mlp_forward_backward_cases": 1,
    "source_divisibility_assertion_cases": 1,
    "generalized_ring_cases": ring_cases,
    "ema_expansion_cases": ema_cases,
    "shuffle_inverse_permutations": permutation_cases,
    "original_ema_method_ast": "unchanged",
    "source_adapters": ["local gather", "identity shuffle/unshuffle", "cuda placement -> CPU"],
    "distributed_bn_or_vision_training": "not executed",
    "torch": torch.__version__,
}, indent=2))
