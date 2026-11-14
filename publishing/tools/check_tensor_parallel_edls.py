"""Execute the exact published EDLS TP classes on two real CPU/Gloo ranks.

Small deterministic modules only: no model download, CUDA, NCCL or timing.
The copied course classes remain unchanged; fixtures live in this checker.
"""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
REL = "00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md"
SOURCE = "05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part2.ipynb.md"
text = (ROOT / REL).read_text()
code = re.search(r"<!-- example:tp-edls-mlp -->\s*```python\n(.*?)```", text, re.S).group(1)
original = next(block for block in re.findall(r"```python\n(.*?)```", (ROOT / SOURCE).read_text(), re.S)
                if block.startswith("%%writefile tensor_parallel_mlp.py"))
original = original.split("\n", 1)[1].split('\n\nif __name__')[0]
assert code == original
assert hashlib.sha256(code.encode()).hexdigest() == "92414271c13361505eaa4929140e5c452b05a986808c22676c600ea3b1e29cb8"
namespace = {"__name__": __name__}
exec(compile(code, REL + "#tp-edls-mlp", "exec"), namespace)
LlamaMLP = namespace["LlamaMLP"]
AllReduceModule = namespace["AllReduceModule"]


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


def shard_mlp(full, rank, size):
    hidden = full.up_proj.in_features
    width = full.up_proj.out_features // size
    part = LlamaMLP(hidden, width).double()
    interval = slice(rank * width, (rank + 1) * width)
    with torch.no_grad():
        part.up_proj.weight.copy_(full.up_proj.weight[interval])
        part.gate_proj.weight.copy_(full.gate_proj.weight[interval])
        part.down_proj.weight.copy_(full.down_proj.weight[:, interval])
    return part, interval


def ordinary_case(rank, size, *, seed, rows, hidden, intermediate, input_grad):
    torch.manual_seed(seed)
    norm = torch.nn.RMSNorm(hidden, eps=1e-5).double()
    full = LlamaMLP(hidden, intermediate).double()
    part, interval = shard_mlp(full, rank, size)
    local_norm = torch.nn.RMSNorm(hidden, eps=1e-5).double()
    local_norm.load_state_dict(norm.state_dict())
    model = torch.nn.Sequential(local_norm, AllReduceModule(part))
    value = torch.randn(rows, hidden, dtype=torch.float64)
    reference_input = value.clone().requires_grad_(input_grad)
    local_input = value.clone().requires_grad_(input_grad)
    incoming = torch.linspace(-0.7, 1.1, rows * hidden, dtype=torch.float64).reshape(rows, hidden)
    reference = full(norm(reference_input))
    reference.backward(incoming)
    calls = []
    hook = part.register_forward_hook(lambda module, args, output: calls.append(output.detach().clone()))
    with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reductions:
        actual = model(local_input)
        close(actual, reference)
        actual.backward(incoming)
        assert reductions.call_count == 2
    hook.remove()
    assert len(calls) == 2  # original forward and backward recomputation
    close(calls[0], calls[1])
    close(local_norm.weight.grad, norm.weight.grad)
    assert torch.count_nonzero(norm.weight.grad) > 0
    assert not torch.allclose(local_norm.weight.grad * size, norm.weight.grad)
    if input_grad:
        close(local_input.grad, reference_input.grad)
    else:
        assert local_input.grad is reference_input.grad is None
    for name in ("up_proj", "gate_proj", "down_proj"):
        actual_grad = getattr(part, name).weight.grad
        full_grad = getattr(full, name).weight.grad
        expected = full_grad[:, interval] if name == "down_proj" else full_grad[interval]
        close(actual_grad, expected)


def frozen_input_case(rank, size):
    torch.manual_seed(4412)
    full = LlamaMLP(4, 12).double()
    part, _ = shard_mlp(full, rank, size)
    value = torch.randn(3, 4, dtype=torch.float64)
    reference = full(value)
    assert reference.requires_grad
    reference.sum().backward()
    assert all(parameter.grad is not None for parameter in full.parameters())
    actual = AllReduceModule(part)(value)
    assert not actual.requires_grad  # no tensor argument of Function.apply requires grad
    try:
        actual.sum().backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError("the original boundary unexpectedly accepted a frozen input")
    assert all(parameter.requires_grad and parameter.grad is None for parameter in part.parameters())


def dropout_case(rank, size):
    torch.manual_seed(4413)
    full = LlamaMLP(4, 12).double()
    part, _ = shard_mlp(full, rank, size)
    dropout = torch.nn.Dropout(0.5)
    model = AllReduceModule(torch.nn.Sequential(dropout, part))
    value = torch.linspace(0.2, 1.7, 32, dtype=torch.float64).reshape(8, 4).requires_grad_()
    incoming = torch.linspace(-0.4, 1.2, 32, dtype=torch.float64).reshape(8, 4)
    masks = []
    hook = dropout.register_forward_hook(
        lambda module, args, output: masks.append((output / args[0]).detach().clone()))
    actual = model(value)
    # Differentiate the function that produced the observed forward: fixed FIRST mask.
    probe = value.detach().clone().requires_grad_()
    parameters = tuple(part.parameters())
    expected = torch.autograd.grad(part(probe * masks[0]), (*parameters, probe), incoming)
    actual.backward(incoming)
    hook.remove()
    assert len(masks) == 2 and not torch.equal(masks[0], masks[1])
    assert any(not torch.allclose(p.grad, correct) for p, correct in zip(parameters, expected[:-1]))


def worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=90))
    try:
        count = 0
        for seed in (4401, 4402):
            for rows, hidden, intermediate in ((3, 4, 12), (2, 6, 16)):
                for input_grad in (True, False):
                    ordinary_case(rank, size, seed=seed, rows=rows, hidden=hidden,
                                  intermediate=intermediate, input_grad=input_grad)
                    count += 1
        frozen_input_case(rank, size)
        dropout_case(rank, size)
        if rank == 0:
            print(json.dumps({"torch": torch.__version__, "backend": "gloo", "ranks": size,
                "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_class_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "ordinary_cases": count, "real_reductions_per_case": 2,
                "checked": ["outputs", "input gradients", "all three weight-shard gradients",
                            "replicated RMSNorm gradients", "two local forward calls"],
                "negative_controls": ["extra TP sum of full RMSNorm gradient",
                    "frozen input prevents the custom autograd boundary",
                    "dropout backward differentiates a different mask"],
                "limits": "Small CPU modules; no pretrained weights, CUDA, NCCL or timing."}, indent=2))
            print("PASS exact EDLS classes on two real CPU/Gloo ranks")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bookvar-edls-tp-") as temporary:
        mp.spawn(worker, args=(2, "file://" + temporary + "/rendezvous"), nprocs=2, join=True)
