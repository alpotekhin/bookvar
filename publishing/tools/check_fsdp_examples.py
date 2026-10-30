"""Execute chapter 44e examples on CPU/Gloo; no mocked FSDP or GPU claims.

Requires PyTorch 2.14. Eight Python fences are executed from the current
chapter. Four real worker processes exercise DeviceMesh, DTensor, FSDP2,
hybrid sharding, gradients, AdamW states, dtype policy and token weighting.
No throughput, allocator peak, NCCL or checkpoint portability is measured.
"""
from __future__ import annotations

import ast
import copy
import inspect
import hashlib
import json
import math
import re
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md"


def chapter_blocks() -> dict[str, str]:
    text = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    assert len(blocks) == 8, "Review test mapping after changing chapter code."
    for block in blocks:
        ast.parse(block)
    markers = {
        "shard": "def shard_model(",
        "precision": "policy = MixedPrecisionPolicy(\n",
        "mesh": 'mesh = init_device_mesh(',
        "layout": "values = torch.tensor(",
        "tokens": "def token_normalized_step(",
        "owned": "def owned_adam_step(",
        "nonlinear": "pair_mesh = mesh[",
        "hybrid": "row_sum_shard = gradient_shard.clone()",
    }
    selected = {}
    for name, marker in markers.items():
        matches = [block for block in blocks if marker in block]
        assert len(matches) == 1, (name, "ambiguous chapter-code mapping")
        selected[name] = matches[0]
    # Deliberately make the test sensitive to removal of the actual group.
    assert "group=data_group" in selected["tokens"]
    assert "get_world_size(data_group)" in selected["tokens"]
    assert inspect.signature(fully_shard).parameters["reshard_after_forward"].default is None
    return selected


def execute(source: str, namespace: dict) -> None:
    exec(compile(source, str(CHAPTER), "exec"), namespace)


def numerical_checks() -> None:
    count, n = 7_000_000_000, 64
    p, g, o = 2 * count, 2 * count, 12 * count
    stages = [p + g + o, p + g + o / n, p + (g + o) / n, (p + g + o) / n]
    assert [x / 1e9 for x in stages] == [112, 29.3125, 15.53125, 1.75]
    for n in (1, 2, 8, 64):
        for o_per_param, total in ((12, 16), (8, 12)):
            model = [2 + 2 + o_per_param,
                     4 + o_per_param / n,
                     2 + (2 + o_per_param) / n, total / n]
            assert model[0] >= model[1] >= model[2] >= model[3]
    torch.testing.assert_close(
        torch.tensor([1., 2., 3., 4.]) - .1 * torch.tensor([2., 4., 6., 8.]),
        torch.tensor([.8, 1.6, 2.4, 3.2]),
    )
    assert (7 / 8) * (2 * 2 + 4) == 7
    assert (7 / 8) * (2 * 2 + 2) == 5.25
    assert max(6, 8) == 8 and 6 + 8 == 14
    assert (6 / 8, 8 / 8) == (.75, 1)
    assert (6 * 4 / 2 / 8, 8 * 4 / 2 / 8) == (1.5, 2)
    assert 8192 * 32768 * 2 / 2**20 == 512
    assert (8192 // 4, 32768 // 8) == (2048, 4096)
    assert 2048 * 4096 * 2 / 2**20 == 16
    assert 8192 * 4096 * 2 / 2**20 == 64
    assert math.isclose(1.12e12 / 200e9, 5.6)
    # HSDP state scales with S, while its default gradient average covers R*S.
    assert 16 * count / 8 == 14e9
    assert (2 * 8) / 6 == 16 / 6


def owned_optimizer_checks(source):
    namespace = {}
    execute(source, namespace)
    update = namespace["owned_adam_step"]
    torch.testing.assert_close(torch.cat([s[1] for s in namespace["owned"]]),
                               torch.tensor([.2, .4, .6, .8], dtype=torch.float64))
    torch.testing.assert_close(torch.cat([s[2] for s in namespace["owned"]]),
                               torch.tensor([.04, .16, .36, .64], dtype=torch.float64))
    gradients = [[2., 4., 6., 8.], [-2., 4., 0., -8.], [0., -1., .5, 1.], [1., 2., 3., 4.]]
    for owners in [1, 2, 4]:
        reference = nn.Parameter(torch.tensor([1., 2., 3., 4.], dtype=torch.float64))
        optimizer = torch.optim.Adam([reference], lr=.1, betas=(.9, .99), eps=1e-8)
        states = [(s.clone(), torch.zeros_like(s), torch.zeros_like(s))
                  for s in reference.detach().chunk(owners)]
        for step, gradient in enumerate(gradients, 1):
            grad = torch.tensor(gradient, dtype=torch.float64)
            reference.grad = grad.clone()
            optimizer.step()
            states = [update(w, g, first, second, step)
                      for (w, first, second), g in zip(states, grad.chunk(owners))]
            for index, expected in [(0, reference.detach()),
                                    (1, optimizer.state[reference]["exp_avg"]),
                                    (2, optimizer.state[reference]["exp_avg_sq"])]:
                torch.testing.assert_close(torch.cat([s[index] for s in states]), expected,
                                           rtol=1e-12, atol=1e-12)
    print("PASS actual owned Adam code: four steps, 1/2/4 owners, weights and both moments", flush=True)


class ToyModel(nn.Module):
    """Two independently sharded blocks plus root-owned output projection."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.Tanh()),
            nn.Sequential(nn.Linear(8, 8), nn.Tanh()),
        ])
        self.head = nn.Linear(8, 4)

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(inputs)


def full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def close(actual, expected, *, atol=2e-10, rtol=2e-8):
    torch.testing.assert_close(full(actual), expected, rtol=rtol, atol=atol)


def compare(model, reference, optimizer, reference_optimizer):
    for (name, actual), (other_name, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        assert isinstance(actual, DTensor), (name, type(actual))
        close(actual, expected)
        assert actual.grad is not None and expected.grad is not None
        close(actual.grad, expected.grad)
        a_state, e_state = optimizer.state[actual], reference_optimizer.state[expected]
        assert a_state.keys() == e_state.keys()
        for key in a_state:
            close(a_state[key], e_state[key])


def batch(rank, step=0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(130 + rank + 10 * step)
    inputs = torch.randn(4, 8, generator=generator, dtype=dtype)
    targets = (torch.arange(4) + rank + step) % 4
    # Rank zero has no valid target; others have one/two/three tokens.
    targets[rank:] = -100
    return inputs, targets


def reference_step(model, optimizer, step):
    inputs, targets = zip(*(batch(rank, step) for rank in range(4)), strict=True)
    inputs, targets = torch.cat(inputs), torch.cat(targets)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        model(inputs), targets, ignore_index=-100, reduction="sum"
    ) / (targets != -100).sum()
    loss.backward()
    optimizer.step()


def layout_checks(rank, namespace, blocks):
    execute(blocks["mesh"], namespace)
    mesh = namespace["mesh"]
    assert dist.get_process_group_ranks(namespace["shard_group"]) == [
        2 * (rank // 2), 2 * (rank // 2) + 1
    ]
    assert dist.get_process_group_ranks(namespace["replicate_group"]) == [
        rank % 2, rank % 2 + 2
    ]
    execute(blocks["layout"], namespace)
    expected = torch.tensor([6., 8.]) if rank < 2 else torch.tensor([10., 12.])
    torch.testing.assert_close(namespace["local_values"], expected)
    execute(blocks["nonlinear"], namespace)
    torch.testing.assert_close(namespace["correct_relu"], torch.tensor([0., 2.]))
    torch.testing.assert_close(namespace["wrong_relu"], torch.tensor([1., 4.]))
    assert not torch.equal(namespace["correct_relu"], namespace["wrong_relu"])
    execute(blocks["hybrid"], namespace)
    torch.testing.assert_close(namespace["row_sum_shard"], torch.tensor([[4.], [6.], [12.], [14.]][rank]))
    torch.testing.assert_close(namespace["gradient_shard"], torch.tensor([4. if rank % 2 == 0 else 5.]))
    assert not torch.equal(namespace["row_sum_shard"] / 2, namespace["gradient_shard"])
    # Partial is a pending SUM, not a slice to concatenate.
    shard_mesh = mesh["shard"]
    local = torch.tensor([1., 2.]) if rank % 2 == 0 else torch.tensor([3., 4.])
    partial = DTensor.from_local(local, shard_mesh, (Partial("sum"),))
    replicated = partial.redistribute(placements=(Replicate(),))
    torch.testing.assert_close(replicated.to_local(), torch.tensor([4., 6.]))
    sharded = partial.redistribute(placements=(Shard(0),))
    torch.testing.assert_close(
        sharded.to_local(), torch.tensor([4. if rank % 2 == 0 else 6.])
    )
    return mesh


def training_checks(rank, namespace, mesh, label):
    torch.manual_seed(321)
    model = ToyModel().double()
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    model, optimizer = namespace["shard_model"](model, mesh)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    for step in range(3):
        inputs, targets = batch(rank, step)
        namespace["token_normalized_step"](
            model, optimizer, inputs, targets, dist.group.WORLD
        )
        reference_step(reference, reference_optimizer, step)
        compare(model, reference, optimizer, reference_optimizer)

    # All ranks reject the empty global denominator, before collective forward.
    previous = [full(p).detach().clone() for p in model.parameters()]
    inputs, targets = batch(rank)
    targets.fill_(-100)
    try:
        namespace["token_normalized_step"](
            model, optimizer, inputs, targets, dist.group.WORLD
        )
    except ValueError as exc:
        assert str(exc) == "Нет учитываемых целевых токенов"
    else:
        raise AssertionError("Globally empty batch was not rejected")
    for actual, expected in zip(model.parameters(), previous, strict=True):
        close(actual, expected)
    if rank == 0:
        print(f"PASS {label}: three AdamW steps, unequal/zero-local tokens, "
              "full parameters/gradients/moments, empty-global rejection", flush=True)


def lifetime_and_accumulation(rank, namespace, mesh):
    torch.manual_seed(92)
    model = ToyModel().double()
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    model, optimizer = namespace["shard_model"](model, mesh)
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    # All four ranks have different batches and different token counts.
    window = [batch(rank, step) for step in (4, 5)]
    total = sum(int((batch(r, step)[1] != -100).sum()) for r in range(4) for step in (4, 5))
    for index, (inputs, targets) in enumerate(window):
        model.set_requires_gradient_sync(index == 1)
        logits = model(inputs)
        assert isinstance(model.blocks[0][0].weight, DTensor)
        assert not isinstance(model.head.weight, DTensor)
        loss = nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction="sum"
        ) * 4 / total
        loss.backward()
        if index == 0:
            # Registered parameters are sharded again. FSDP keeps unsynchronised
            # gradients separately; their private storage is not a public .grad.
            # Numerical equality after the second microbatch proves accumulation,
            # not a measurement of that temporary buffer or its allocator peak.
            assert all(isinstance(p, DTensor) for p in model.parameters())
    optimizer.step()
    inputs, targets = zip(*(batch(r, step) for r in range(4) for step in (4, 5)), strict=True)
    reference_loss = nn.functional.cross_entropy(
        reference(torch.cat(inputs)), torch.cat(targets), ignore_index=-100, reduction="sum"
    ) / total
    reference_loss.backward()
    reference_optimizer.step()
    compare(model, reference, optimizer, reference_optimizer)
    if rank == 0:
        print("PASS default child/root lifetime and two-microbatch accumulation", flush=True)


def dtype_checks(rank, namespace, mesh):
    torch.manual_seed(17)
    model = ToyModel()
    execute(namespace["_precision_source"], namespace)
    model, optimizer = namespace["shard_model"](model, mesh, namespace["policy"])
    assert all(p.dtype == torch.float32 for p in model.parameters())
    observed = []

    def observe(module, inputs):
        observed.append((module.weight.dtype, inputs[0].dtype))
    handle = model.blocks[0][0].register_forward_pre_hook(observe)
    inputs, _ = batch(rank, dtype=torch.float32)
    output = model(inputs)
    assert output.dtype == torch.bfloat16
    assert observed == [(torch.bfloat16, torch.bfloat16)]
    assert model.head.weight.dtype == torch.bfloat16
    output.float().square().mean().backward()
    handle.remove()
    assert all(isinstance(p, DTensor) and p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad.dtype == torch.float32 for p in model.parameters())
    optimizer.step()
    for parameter in model.parameters():
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    if rank == 0:
        print("PASS real CPU BF16 forward / FP32 originals, gradients and AdamW moments", flush=True)


def worker(rank, rendezvous, blocks):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank,
        world_size=4, timeout=timedelta(seconds=40),
    )
    try:
        namespace = {"torch": torch}
        execute(blocks["shard"], namespace)
        execute(blocks["tokens"], namespace)
        namespace["_precision_source"] = blocks["precision"]
        hybrid_mesh = layout_checks(rank, namespace, blocks)
        if rank == 0:
            print("PASS four-rank DeviceMesh, actual DTensor Partial/ReLU and HSDP SUM→SUM/4", flush=True)
        full_mesh = init_device_mesh("cpu", (4,))
        training_checks(rank, namespace, full_mesh, "FSDP2 1D")
        training_checks(rank, namespace, hybrid_mesh, "FSDP2 2x2 hybrid")
        lifetime_and_accumulation(rank, namespace, full_mesh)
        dtype_checks(rank, namespace, full_mesh)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    assert tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 14), torch.__version__
    blocks = chapter_blocks()
    numerical_checks()
    owned_optimizer_checks(blocks["owned"])
    text = CHAPTER.read_text()
    for anchor in ["cs336-distributed-fsdp", "peak-memory-synchronization", "контракт-корректности-custom-fsdp"]:
        assert f'<a id="{anchor}"></a>' in text
    assert len(re.findall(r"!\[\[", text)) == 6
    for source in ["lecture-08-zero-sharding", "lecture-08-zero-memory-accounting",
                   "assignment-02-task-optimizer-state-sharding", "assignment-02-task-optimizer-state-sharding-accounting",
                   "assignment-02-task-fsdp", "assignment-02-task-fsdp-accounting",
                   "assignment-02-task-fsdp-calcs", "assignment-02-task-fsdp-tp-calcs"]:
        assert source in text
    for row in ["| ZeRO-1 | $4+12/N$ | $4+8/N$ |", "| ZeRO-2 | $2+14/N$ | $2+10/N$ |"]:
        assert row in text
    print(f"PASS eight chapter fences parsed and numerical calculations; torch={torch.__version__}",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="bookvar-fsdp-") as directory:
        context = mp.spawn(worker, args=(str(Path(directory) / "store"), blocks),
                           nprocs=4, join=False)
        deadline = time.monotonic() + 150
        try:
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    raise TimeoutError("FSDP CPU checks exceeded 150 seconds")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
    print("PASS chapter 44e CPU/Gloo examples. CUDA/NCCL, peak memory and DCP restore NOT tested.")
    print(json.dumps({"torch": torch.__version__, "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      "actual_python_fences": len(blocks), "figure_count": 6, "processes": 4}, indent=2))


if __name__ == "__main__":
    main()

