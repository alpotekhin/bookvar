"""Check chapter 44a on CPU/Gloo; this is not a CUDA/NCCL benchmark.

Run from any directory with PyTorch installed. Training and flat-gradient fences
are extracted from the current chapter and executed verbatim. GPU setup is
only syntax-checked. Worker timeouts are bounded and no external service is used.
"""

from __future__ import annotations

import ast
import copy
import math
import re
import runpy
import tempfile
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md"
COURSE = ROOT / "05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment2-systems"
RTOL = 1e-9
ATOL = 1e-11


def chapter_blocks() -> dict[str, str]:
    text = CHAPTER.read_text()
    fences = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
    assert len(fences) == 15, "Review the test mapping when chapter code changes"
    for source in fences:
        ast.parse(source)
    markers = {
        "collective": "values = torch.arange",
        "equal": "loss = criterion(predictions, targets) / len(window)",
        "tokens": "global_count = torch.tensor",
        "flat": "flat = torch.cat",
    }
    selected = {}
    for name, marker in markers.items():
        matches = [source for source in fences if marker in source]
        assert len(matches) == 1, (name, "chapter source mapping is ambiguous")
        selected[name] = matches[0]
    assert "group=group" in selected["tokens"]
    assert "group_size = dist.get_world_size(group)" in selected["tokens"]
    assert "no_sync()" in selected["equal"] and "no_sync()" in selected["tokens"]
    return selected


def extended_arithmetic_checks() -> None:
    text = CHAPTER.read_text()
    before = (ROOT / "docs/audit/snapshots/2026-10-23-ddp-before.md").read_text()
    ids = lambda s: s.split("source_unit_id:\n", 1)[1].split("\nprimary_sources:", 1)[0]
    assert ids(text) == ids(before)
    assert set(re.findall(r'<a id="([^"]+)"', before)) <= set(re.findall(r'<a id="([^"]+)"', text))
    for value in ("[2,4,6,8,10,12]", "на 16 ms", "до 13 ms", "до 11 ms",
                  "alternate_ring_all_reduce", "5,285 ms", "4,624 ms", "5,395 ms",
                  "F_{\\mathrm{backward}}=\\frac{12bdh}{N}", "r(m+n)=mn=4"):
        assert value in text, value

    def finish(messages):
        end = 0
        for ready, size in messages:
            end = max(end, ready) + 1 + size
        return end
    assert finish([(8, 1)] * 4) == 16
    assert finish([(8, 4)]) == 13
    assert finish([(4, 2), (8, 2)]) == 11
    gossip = [0., 0., 0., 4.]
    for expected in ([4/3, 0, 4/3, 4/3], [8/9, 8/9, 8/9, 4/3]):
        gossip = [(gossip[(i-1) % 4] + gossip[i] + gossip[(i+1) % 4])/3
                  for i in range(4)]
        assert all(math.isclose(a, b) for a,b in zip(gossip, expected))
        assert math.isclose(sum(gossip), 4)
        assert not all(math.isclose(x, 1) for x in gossip)

    # Simultaneous rounds use the old buffers, never an in-round updated value.
    for n in range(2, 10):
        initial = [[rank + j for j in range(n)] for rank in range(n)]
        current = copy.deepcopy(initial)
        for step in range(1, n):
            previous = copy.deepcopy(current)
            for rank in range(n):
                chunk = (rank - step - 1) % n
                current[rank][chunk] += previous[(rank - 1) % n][chunk]
            if n == 4:
                owner = (step + 1) % n
                assert current[owner][0] == [3, 6, 6][step - 1]
        expected = [sum(row[j] for row in initial) for j in range(n)]
        assert [current[rank][rank] for rank in range(n)] == expected
        known = [{rank: current[rank][rank]} for rank in range(n)]
        for step in range(1, n):
            previous = copy.deepcopy(known)
            for rank in range(n):
                chunk = (rank - step) % n
                known[rank][chunk] = previous[(rank - 1) % n][chunk]
        assert all([parts[j] for j in range(n)] == expected for parts in known)
        naive = copy.deepcopy(initial)
        for step in range(1, n):
            for rank in range(n):
                naive[rank] = [a+b for a,b in zip(naive[rank], initial[(rank-step) % n])]
        assert naive == [expected] * n
        assert math.isclose((n - 1) / (2 * (n - 1) / n), n / 2)

    b, d, h, f, w = 10752, 7168, 7168, 1.792e14, 1e11
    assert b*w/f == 6 and b % 7 == b % 8 == 0
    compute = lambda n: 12*b*d*h/(n*f)
    network = lambda n, byte_factor=1: byte_factor*12*d*h*(n-1)/(n*w)
    assert math.isclose(compute(7), network(7))
    assert round(compute(7)*1000, 3) == 5.285
    assert round(compute(8)*1000, 3) == 4.624
    assert round(network(8)*1000, 3) == 5.395
    assert network(8) > compute(8)
    assert math.isclose(network(4, 2), compute(4))
    assert network(7, 2) > compute(7)
    print("PASS ring chunk rounds, whole-tensor negative comparison, bucket schedule and DP capacity", flush=True)


def flat_and_power_checks(rank: int, source: str) -> None:
    model = nn.ParameterList([nn.Parameter(torch.zeros(2, dtype=torch.float64)),
                              nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))])
    model[0].grad = torch.tensor([1., 3.], dtype=torch.float64) + 2*rank
    model[1].grad = torch.tensor([[5., 7.], [9., 11.]], dtype=torch.float64) + 2*rank
    namespace = {"model": model, "torch": torch, "dist": dist}
    exec(compile(source, str(CHAPTER) + ":flat", "exec"), namespace)
    assert_close(model[0].grad, torch.tensor([2., 4.], dtype=torch.float64))
    assert_close(model[1].grad, torch.tensor([[6., 8.], [10., 12.]], dtype=torch.float64))
    assert namespace["offset"] == 6

    torch.manual_seed(195)
    model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    reference = copy.deepcopy(model)
    data = torch.randn(2, 3, 4, dtype=torch.float64)
    targets = torch.randn(2, 3, 2, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    nn.functional.mse_loss(reference(data), targets).backward()
    nn.functional.mse_loss(model(data[rank]), targets[rank]).backward()
    exec(compile(source, str(CHAPTER) + ":flat-model", "exec"),
         {"model": model, "torch": torch, "dist": dist})
    optimizer.step()
    ref_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert_close(actual.grad, expected.grad)
        assert_close(actual, expected)

    # Two genuine reductions for the chapter's rank-1 illustration, not the
    # production PowerSGD hook, convergence, or a compression benchmark.
    matrix = torch.diag(torch.tensor([3., 1.] if rank == 0 else [1., 3.], dtype=torch.float64))
    q0 = torch.tensor([[1.], [0.]], dtype=torch.float64)
    p = matrix @ q0
    dist.all_reduce(p)
    p /= 2
    assert_close(p, torch.tensor([[2.], [0.]], dtype=torch.float64))
    p /= torch.linalg.vector_norm(p)
    q = matrix.T @ p
    dist.all_reduce(q)
    q /= 2
    approximation = p @ q.T
    assert_close(approximation, torch.diag(torch.tensor([2., 0.], dtype=torch.float64)))
    assert not torch.equal(approximation, 2*torch.eye(2, dtype=torch.float64))
    assert ((1+3)/2)*((3+1)/2) == 4 != (1*3+3*1)/2


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def numerical_and_sampler_checks() -> None:
    # Four-rank example from the chapter, independently of the two-rank run.
    assert torch.equal(sum(torch.arange(4) + rank for rank in range(4)),
                       torch.tensor([6, 10, 14, 18]))
    assert (1 + 3) / 2 == 2 and (2 * 1 + 6 * 3) / 8 == 2.5
    assert (2 * (2 / 8) + 18 * (2 / 8)) / 2 == 2.5
    payload, bandwidth, count = 2**30, 25e9, 8
    assert math.isclose(2 * (count - 1) / count * payload / bandwidth * 1e3,
                        75.16192768)
    assert math.isclose(2 * (count - 1) * 5e-6 * 1e3, 0.07)
    assert math.isclose(2 * (count - 1) / count * 4096 / bandwidth * 1e6,
                        0.28672)
    assert 180 + 70 - 55 == 195 and 100 + 70 - 30 == 140
    assert math.isclose(195 / 140, 1.3928571428571428)
    assert 4096**2 == 16777216 and 8 * (4096 + 4096) == 65536
    assert 4096**2 / (8 * (4096 + 4096)) == 256
    residual, sent = 0.0, []
    for expected_residual in (0.4, -0.2, 0.2):
        value = 0.4 + residual
        sent.append(round(value))
        residual = value - sent[-1]
        assert math.isclose(residual, expected_residual, abs_tol=1e-15)
    assert sent == [0, 1, 0] and math.isclose(sum(sent) + residual, 1.2)

    def parts(size: int, ranks: int, drop: bool, shuffle: bool = False,
              epoch: int = 0) -> list[list[int]]:
        result = []
        for rank in range(ranks):
            sampler = DistributedSampler(range(size), num_replicas=ranks,
                                         rank=rank, drop_last=drop,
                                         shuffle=shuffle, seed=42)
            sampler.set_epoch(epoch)
            result.append(list(sampler))
        return result

    assert parts(8, 2, False) == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert parts(10, 3, False) == [[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 1]]
    assert parts(10, 3, True) == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    shuffled = parts(10, 3, False, True)
    permutation = torch.randperm(10, generator=torch.Generator().manual_seed(42)).tolist()
    merged = [shuffled[rank][i] for i in range(4) for rank in range(3)]
    assert merged == permutation + permutation[:2]
    assert shuffled == parts(10, 3, False, True)
    assert shuffled != parts(10, 3, False, True, epoch=1)
    sampler = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False,
                                 drop_last=False)
    assert len(list(sampler)) == 4
    assert sum(len(x) for x in DataLoader(range(10), batch_size=3, sampler=sampler,
                                        drop_last=True)) == 3


def collective_checks(rank: int, source: str) -> None:
    namespace = {"torch": torch, "dist": dist, "device": torch.device("cpu"),
                 "rank": rank}
    exec(compile(source, str(CHAPTER) + ":collective", "exec"), namespace)
    assert_close(namespace["values"], torch.tensor([1., 3., 5., 7.]))
    output = torch.empty(2)
    dist.reduce_scatter_single(output, torch.arange(4, dtype=torch.float32) + rank)
    assert_close(output, torch.tensor([1., 3.]) if rank == 0 else torch.tensor([5., 7.]))
    reconstructed = torch.empty(4)
    dist.all_gather_single(reconstructed, output)
    assert_close(reconstructed, torch.tensor([1., 3., 5., 7.]))
    received = torch.empty(2)
    dist.all_to_all_single(received, torch.arange(2, dtype=torch.float32) + 10 * rank)
    assert_close(received, torch.tensor([float(rank), 10. + rank]))

    # A mismatched *meaning* with matching shapes does not cause a transport error.
    # Both ranks issue two SUMs, but A/B are crossed: 1+20=21 and 10+2=12.
    first = torch.tensor([1. if rank == 0 else 20.])
    second = torch.tensor([10. if rank == 0 else 2.])
    dist.all_reduce(first)
    dist.all_reduce(second)
    assert first.item() == 21 and second.item() == 12
    assert first.item() != 3 and second.item() != 30

    subgroup = dist.new_group([1])  # Both original ranks create it.
    if rank == 1:
        assert dist.get_rank(subgroup) == 0 and dist.get_world_size(subgroup) == 1
        assert dist.get_global_rank(subgroup, 0) == 1
        dist.destroy_process_group(subgroup)
    dist.barrier()


def check_gradients_parameters_optimizer(ddp: DDP, reference: nn.Module,
                                        optimizer: torch.optim.Optimizer,
                                        reference_optimizer: torch.optim.Optimizer) -> None:
    for actual, expected in zip(ddp.module.parameters(), reference.parameters(), strict=True):
        assert_close(actual, expected)
        if actual.requires_grad:
            assert actual.grad is not None and expected.grad is not None
            assert_close(actual.grad, expected.grad)
        assert optimizer.state[actual].keys() == reference_optimizer.state[expected].keys()
        for key, value in optimizer.state[actual].items():
            assert_close(value, reference_optimizer.state[expected][key])


def execute_training_source(source: str, ddp: DDP, optimizer: torch.optim.Optimizer,
                            window: list[tuple[torch.Tensor, torch.Tensor]],
                            criterion: nn.Module) -> None:
    exec(compile(source, str(CHAPTER) + ":training", "exec"), {
        "torch": torch, "dist": dist, "nullcontext": nullcontext,
        "ddp": ddp, "optimizer": optimizer, "window": window, "criterion": criterion,
    })


def training_checks(rank: int, blocks: dict[str, str]) -> None:
    # These are the two actual architectures used in the pinned CS336 test file.
    # We reuse their definitions, not the assignment's student adapters/test suite.
    definitions = runpy.run_path(str(COURSE / "tests/common.py"))
    for name in ("ToyModel", "ToyModelWithTiedWeights"):
        torch.manual_seed(17 + rank)
        base = definitions[name]().double()
        before = [value.detach().clone() for value in base.parameters() if value.requires_grad]
        ddp = DDP(base)
        if rank == 1:
            assert any(not torch.equal(a, b) for a, b in zip(
                before, (x for x in base.parameters() if x.requires_grad), strict=True))
        reference = copy.deepcopy(base)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=0.003)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(2026)
        for microbatches in (2, 2, 1):  # Includes a shortened final window.
            data = torch.randn(2, microbatches, 3, 10, generator=generator, dtype=torch.float64)
            targets = torch.randn(2, microbatches, 3, 10, generator=generator, dtype=torch.float64)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_loss = criterion(reference(data.reshape(-1, 10)), targets.reshape(-1, 10))
            expected_loss.backward()
            reference_optimizer.step()
            # Explicitly verify that global loss logging requires an extra collective.
            with torch.no_grad():
                local_loss = criterion(ddp.module(data[rank].reshape(-1, 10)),
                                       targets[rank].reshape(-1, 10))
                dist.all_reduce(local_loss)
                assert_close(local_loss / 2, expected_loss.detach())
            execute_training_source(blocks["equal"], ddp, optimizer,
                                    list(zip(data[rank], targets[rank], strict=True)), criterion)
            check_gradients_parameters_optimizer(ddp, reference, optimizer, reference_optimizer)
        del ddp

    # A small deterministic token classifier isolates masking and normalization.
    torch.manual_seed(333)
    ddp = DDP(nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).double())
    reference = copy.deepcopy(ddp.module)
    optimizer = torch.optim.Adam(ddp.parameters(), lr=0.002)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(444)
    for step in range(3):
        microbatches = 1 if step == 2 else 2
        data = torch.randn(2, microbatches, 3, 4, generator=generator, dtype=torch.float64)
        labels = torch.randint(3, (2, microbatches, 3), generator=generator)
        if step == 0:
            labels[0, 0, 2] = -100
            labels[0, 1, :] = -100
            labels[1, 1, 1:] = -100  # Local valid-token totals are 2 versus 4.
        elif step == 1:
            labels[0] = -100  # One rank has zero targets throughout the window.
            labels[1, 1, 2] = -100
        else:
            labels[0, 0, 1:] = -100  # Shortened window, still unequal token counts.
        reference_optimizer.zero_grad(set_to_none=True)
        expected_loss = nn.functional.cross_entropy(reference(data).reshape(-1, 3), labels.reshape(-1))
        expected_loss.backward()
        reference_optimizer.step()
        with torch.no_grad():
            loss_sum = nn.functional.cross_entropy(ddp.module(data[rank]).reshape(-1, 3),
                                                   labels[rank].reshape(-1), reduction="sum")
            dist.all_reduce(loss_sum)
            assert_close(loss_sum / (labels != -100).sum(), expected_loss.detach())
        execute_training_source(blocks["tokens"], ddp, optimizer,
                                list(zip(data[rank], labels[rank], strict=True)), nn.CrossEntropyLoss())
        check_gradients_parameters_optimizer(ddp, reference, optimizer, reference_optimizer)
    before_parameters = [p.detach().clone() for p in ddp.parameters()]
    before_state = copy.deepcopy(optimizer.state_dict())
    empty_window = [(torch.zeros(3, 4, dtype=torch.float64), torch.full((3,), -100))]
    try:
        execute_training_source(blocks["tokens"], ddp, optimizer, empty_window, nn.CrossEntropyLoss())
    except ValueError as error:
        assert "В окне нет" in str(error)
    else:
        raise AssertionError("Globally empty targets must not update the optimizer")
    for current, previous in zip(ddp.parameters(), before_parameters, strict=True):
        assert_close(current, previous)
    after_state = optimizer.state_dict()
    for parameter, state in before_state["state"].items():
        for key, value in state.items():
            assert_close(after_state["state"][parameter][key], value)
    dist.barrier()  # All ranks reached the same early-exit branch.


def join_checks(rank: int) -> None:
    for initial_divisor, expected_weight in ((True, -3.5), (False, -5.0)):
        ddp = DDP(nn.Linear(1, 1, bias=False).double())
        with torch.no_grad():
            ddp.module.weight.zero_()
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1)
        with ddp.join(divide_by_initial_world_size=initial_divisor):
            for _ in range(rank + 1):
                optimizer.zero_grad(set_to_none=True)
                ddp(torch.ones(1, 1, dtype=torch.float64)).sum().mul(1 + rank * 2).backward()
                optimizer.step()
        assert math.isclose(ddp.module.weight.item(), expected_weight)
    ddp = DDP(nn.Linear(1, 1, bias=False).double())
    optimizer = torch.optim.Adam(ddp.parameters(), lr=0.01)
    with ddp.join():
        for _ in range(rank + 1):
            optimizer.zero_grad(set_to_none=True)
            ddp(torch.ones(1, 1, dtype=torch.float64)).sum().mul(1 + rank * 2).backward()
            optimizer.step()
    state = optimizer.state[ddp.module.weight]
    evidence = torch.tensor([ddp.module.weight.item(), state["step"].item(),
                             state["exp_avg"].item()], dtype=torch.float64)
    gathered = [torch.empty_like(evidence) for _ in range(2)]
    dist.all_gather(gathered, evidence)
    assert gathered[0][0] == gathered[1][0], "join must sync final model weights"
    assert gathered[0][1] == 1 and gathered[1][1] == 2
    assert gathered[0][2] != gathered[1][2], "join does not sync Adam moments"
    ddp = DDP(nn.Linear(1, 1, bias=False).double())
    raised = False
    try:
        with ddp.join(throw_on_early_termination=True):
            for _ in range(rank + 1):
                ddp(torch.ones(1, 1, dtype=torch.float64)).sum().backward()
    except RuntimeError as error:
        assert "exhausted" in str(error).lower()
        raised = True
    assert raised, "Both ranks must receive early-termination notification"
    dist.barrier()


def counted_allreduce(state, bucket):
    state["calls"] += 1
    return default_hooks.allreduce_hook(state["group"], bucket)


def no_sync_communication_check(rank: int, source: str) -> None:
    # Diagnostic run only: a wrapper counts invocations of the official averaging
    # hook. The reference tests above use ordinary DDP without custom hooks.
    for accumulation, expected_calls in ((True, 1), (False, 2)):
        torch.manual_seed(10)
        ddp = DDP(nn.Linear(4, 3).double())
        state = {"calls": 0, "group": ddp.process_group}
        ddp.register_comm_hook(state, counted_allreduce)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        data = torch.full((2, 3, 4), 0.1 + rank, dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
        code = source if accumulation else source.replace("ddp.no_sync()", "nullcontext()")
        execute_training_source(code, ddp, optimizer,
                                list(zip(data, labels, strict=True)), nn.CrossEntropyLoss())
        assert state["calls"] == expected_calls


def worker(rank: int, store_uri: str, blocks: dict[str, str]) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        collective_checks(rank, blocks["collective"])
        if rank == 0:
            print("PASS real Gloo collectives, subgroup ranks and crossed-order negative control", flush=True)
        flat_and_power_checks(rank, blocks["flat"])
        if rank == 0:
            print("PASS actual flat-gradient fence, full-model SGD reference and two-reduction rank-1 example", flush=True)
        training_checks(rank, blocks)
        if rank == 0:
            print("PASS chapter accumulation code, two CS336 models, masked tokens and Adam reference", flush=True)
        no_sync_communication_check(rank, blocks["equal"])
        if rank == 0:
            print("PASS no_sync diagnostic: one bucket reduction versus two without suppression", flush=True)
        join_checks(rank)
        if rank == 0:
            print("PASS join divisors, final-weight sync, unsynced Adam state and early termination", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        raise SystemExit("This test requires a PyTorch build with CPU/Gloo; no checks were skipped as passed.")
    torch.set_num_threads(1)
    blocks = chapter_blocks()
    numerical_and_sampler_checks()
    extended_arithmetic_checks()
    print(f"PASS chapter syntax, arithmetic and actual DistributedSampler; torch={torch.__version__}", flush=True)
    with tempfile.TemporaryDirectory(prefix="bookvar-ddp-") as directory:
        context = mp.spawn(worker, args=((Path(directory) / "store").as_uri(), blocks),
                           nprocs=2, join=False)
        deadline = time.monotonic() + 90
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("DDP CPU validation exceeded the 90-second process deadline")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
    print("PASS all chapter 44a CPU checks. Not measured: CUDA/NCCL, overlap speed, network faults.")


if __name__ == "__main__":
    main()
