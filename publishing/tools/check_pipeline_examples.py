"""Validate actual chapter timelines and CPU autograd, not a GPU benchmark."""
from __future__ import annotations
import copy
import contextlib
import hashlib
import io
import itertools
import json
import math
import re
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism.md"


def chapter_examples(text):
    matches = re.findall(r"<!-- example: (pipeline-[\w-]+) -->\n```python\n(.*?)^```", text, re.M | re.S)
    assert [name for name, _ in matches] == ["pipeline-boundaries", "pipeline-partition", "pipeline-deferred-dw"]
    examples = {}
    for name, code in matches:
        namespace = {}
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, f"{CHAPTER}:{name}", "exec"), namespace)
        examples[name] = namespace
        if stream.getvalue():
            print(f"PASS actual {name}: {stream.getvalue().strip()}")
    return examples


def timelines(text):
    blocks = re.findall(r"^" + chr(96) * 3 + r"text\n(.*?)^" + chr(96) * 3, text, re.M | re.S)
    result = []
    for block in blocks:
        if not block.startswith("slot "):
            continue
        rows = block.splitlines()
        ticks = list(map(int, rows[0].split()[1:]))
        assert ticks == list(range(len(ticks)))
        grid = []
        for stage, row in enumerate(rows[1:]):
            cells = row.split()
            assert cells[0] == f"S{stage}" and len(cells[1:]) == len(ticks)
            grid.append(cells[1:])
        result.append(grid)
    assert len(result) == 2
    return result


def validate(grid, m):
    p = len(grid)
    assert p and m > 0 and all(len(row) == len(grid[0]) for row in grid)
    times = {}
    for s, row in enumerate(grid):
        for t, op in enumerate(row):
            if op == ".":
                continue
            assert re.fullmatch(r"[FB]\d+", op), op
            assert 0 <= int(op[1:]) < m
            assert (s, op) not in times, "duplicate operation"
            times[s, op] = t
    expected = {(s, f"{kind}{mu}") for s in range(p) for mu in range(m) for kind in "FB"}
    assert set(times) == expected, "missing operation"
    for s in range(p):
        for mu in range(m):
            f, b = times[s, f"F{mu}"], times[s, f"B{mu}"]
            assert b > f, "backward before local forward"
            if s:
                assert f > times[s - 1, f"F{mu}"], "forward dependency"
            if s < p - 1:
                assert b > times[s + 1, f"B{mu}"], "backward dependency"
    peaks = []
    for row in grid:
        live = peak = 0
        for op in row:
            live += 1 if op.startswith("F") else -1 if op.startswith("B") else 0
            assert live >= 0
            peak = max(peak, live)
        assert live == 0
        peaks.append(peak)
    return peaks


def strict_schedule(p, m):
    queues = []
    for s in range(p):
        warmup = min(p - s - 1, m)
        order = [f"F{i}" for i in range(warmup)]
        for i in range(m - warmup):
            order.extend([f"F{i + warmup}", f"B{i}"])
        order.extend(f"B{i}" for i in range(m - warmup, m))
        queues.append(order)
    done = set()
    grid = [[] for _ in range(p)]
    while len(done) < 2 * p * m:
        ready = []
        for s in range(p):
            op = queues[s][0] if queues[s] else "."
            possible = op != "."
            if op.startswith("F") and s:
                possible &= (s - 1, op) in done
            if op.startswith("B"):
                possible &= (s, "F" + op[1:]) in done
                if s < p - 1:
                    possible &= (s + 1, op) in done
            grid[s].append(op if possible else ".")
            if possible:
                ready.append((s, op))
        assert ready, "schedule deadlock"
        for s, op in ready:
            queues[s].pop(0)
            done.add((s, op))
    return grid


def check_autograd(grid, wrong_mean=False, public_step=None):
    torch.manual_seed(19)
    dims = [3, 5, 6, 4, 3]
    reference = torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.Tanh())
        for i in range(4)
    ]).double()
    stages = copy.deepcopy(reference)
    x_ref = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    x = x_ref.detach().clone()
    target = torch.tensor([0, 2, -100, -100, 1, -100, 2, 0])
    count = int((target != -100).sum())
    ref_output = reference(x_ref)
    ref_loss = F.cross_entropy(ref_output, target, ignore_index=-100, reduction="sum") / count
    ref_loss.backward()
    inputs, outputs, loss_values = {}, {}, {}
    for t in range(len(grid[0])):
        for s, row in enumerate(grid):
            op = row[t]
            if op == ".":
                continue
            mu = int(op[1:])
            if op.startswith("F"):
                value = x[2 * mu:2 * mu + 2] if s == 0 else outputs[s - 1, mu]
                inputs[s, mu] = value.detach().clone().requires_grad_(True)
                outputs[s, mu] = stages[s](inputs[s, mu])
            elif s == 3:
                y = target[2 * mu:2 * mu + 2]
                denom = max(int((y != -100).sum()), 1) * 4 if wrong_mean else count
                loss = F.cross_entropy(outputs[s, mu], y, ignore_index=-100, reduction="sum") / denom
                loss_values[mu] = loss.detach()
                loss.backward()
            else:
                outputs[s, mu].backward(inputs[s + 1, mu].grad)
    actual = torch.cat([outputs[3, mu].detach() for mu in range(4)])
    torch.testing.assert_close(actual, ref_output.detach(), rtol=1e-12, atol=1e-12)
    grad_error = max(float((a.grad - b.grad).abs().max())
                     for a, b in zip(stages.parameters(), reference.parameters()))
    input_error = float((torch.cat([inputs[0, mu].grad for mu in range(4)]) - x_ref.grad).abs().max())
    if not wrong_mean:
        assert grad_error < 1e-12 and input_error < 1e-12
        torch.testing.assert_close(sum(loss_values.values()), ref_loss.detach(), rtol=1e-12, atol=1e-12)
        for a, b in zip(stages.parameters(), reference.parameters()):
            torch.testing.assert_close(a.detach() - 0.01 * a.grad, b.detach() - 0.01 * b.grad,
                                       rtol=1e-12, atol=1e-12)
        if public_step is not None:
            public_parts = copy.deepcopy(reference)
            public_output, public_loss, public_dx = public_step(public_parts, x, target, grid)
            torch.testing.assert_close(public_output, ref_output.detach(), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(public_loss, ref_loss.detach(), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(public_dx, x_ref.grad, rtol=1e-12, atol=1e-12)
            for a, b in zip(public_parts.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(a.detach() - .01 * a.grad, b.detach() - .01 * b.grad,
                                           rtol=1e-12, atol=1e-12)
    return grad_error, input_error


def main():
    torch.set_num_threads(1)
    text = CHAPTER.read_text()
    examples = chapter_examples(text)
    public_step = examples["pipeline-boundaries"]["pipeline_step"]
    gpipe, one_f_one_b = timelines(text)
    assert validate(gpipe, 4) == [4, 4, 4, 4]
    assert validate(one_f_one_b, 4) == [4, 3, 2, 1]
    assert one_f_one_b == strict_schedule(4, 4), "table and pseudocode disagree"
    for p in range(1, 9):
        for m in range(1, 33):
            grid = strict_schedule(p, m)
            assert validate(grid, m) == [min(m, p - s) for s in range(p)]
            assert len(grid[0]) == 2 * (m + p - 1)
    broken = copy.deepcopy(gpipe)
    broken[2][6], broken[2][8] = broken[2][8], broken[2][6]
    try:
        validate(broken, 4)
    except AssertionError as error:
        assert "backward dependency" in str(error)
    else:
        raise AssertionError("validator accepted unavailable output gradient")
    missing = copy.deepcopy(one_f_one_b)
    missing[0][0] = "."
    try:
        validate(missing, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing forward")
    for grid in [gpipe, one_f_one_b]:
        parameter_error, input_error = check_autograd(grid, public_step=public_step)
        print(f"PASS CPU detached-boundary autograd: params={parameter_error:.3g}, input={input_error:.3g}")
    bad_parameter, bad_input = check_autograd(one_f_one_b, wrong_mean=True)
    assert bad_parameter > 1e-4 and bad_input > 1e-4
    partition = examples["pipeline-partition"]["partition_costs"]
    cases = 0
    for n in range(1, 7):
        for costs in itertools.product([1, 2, 4], repeat=n):
            for p in range(1, n + 1):
                optimum, pieces = partition(list(costs), p)
                oracle = min(max(sum(costs[a:b]) for a, b in zip((0,) + cuts, cuts + (n,)))
                             for cuts in itertools.combinations(range(1, n), p - 1))
                assert optimum == oracle
                assert len(pieces) == p and pieces[0][0] == 0 and pieces[-1][1] == n
                assert all(a < b for a, b in pieces)
                assert all(pieces[i][1] == pieces[i + 1][0] for i in range(p - 1))
                assert max(sum(costs[a:b]) for a, b in pieces) == optimum
                cases += 1
    assert partition([2, 2, 8, 2, 2, 2], 3) == (8, [(0, 2), (2, 3), (3, 6)])
    for costs, p in [([], 1), ([1], 0), ([1], 2), ([0, 2], 1), ([-1], 1)]:
        try:
            partition(costs, p)
        except ValueError:
            pass
        else:
            raise AssertionError("partition accepted invalid inputs")
    print(f"PASS actual partition code against all contiguous cuts: {cases} inputs")
    deferred = examples["pipeline-deferred-dw"]
    assert deferred["y"].detach().tolist() == [[5, 11]]
    assert deferred["dx"].tolist() == [[-2, -2]]
    assert deferred["dw"].tolist() == [[1, -1], [2, -2]]
    for name in ["x", "w"]:
        value = deferred[name].detach().clone()
        actual = deferred["dx" if name == "x" else "dw"]
        flat = value.reshape(-1)
        for index in range(value.numel()):
            old = float(flat[index])
            values = []
            for sign in [1, -1]:
                flat[index] = old + sign * 1e-5
                x = value if name == "x" else deferred["x"].detach()
                w = value if name == "w" else deferred["w"].detach()
                values.append(float(((x @ w) * deferred["dy"]).sum()))
            flat[index] = old
            assert abs((values[0] - values[1]) / 2e-5 - float(actual.reshape(-1)[index])) < 1e-8
    print("PASS actual deferred-dW code, analytic values and x/W finite differences")
    assert math.isclose(math.hypot(3, 4), 5)
    assert 4 / math.hypot(3, 4) == .8
    assert (6 + 4) / (3 + 1) == 2.5 and (6 / 3 + 4 / 1) / 2 == 3
    bn_x = torch.tensor([[0.], [2.], [10.], [12.]], dtype=torch.float64)
    bn = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=False).double()
    assert not torch.allclose(bn(bn_x), torch.cat([bn(v) for v in bn_x.chunk(2)]))
    for p, m, expected in [(4, 10, True), (4, 9, False), (4, 3, True)]:
        rounds = max(1, m // p)
        assert (m % rounds == 0) is expected
    assert 3 * 128 == 384 and 7 * 128 == 896
    assert 2 * 16 * 384 / 1024 == 12 and 2 * 16 * 896 / 1024 == 28
    assert 2 * 8192 * (8192 + 8192) / 2**20 == 256
    for phrase in ["384 MiB", "896 MiB", "12 GiB", "28 GiB", "4/8/6", "3,52", "0,88"]:
        assert phrase in text
    assert [round(n * 900 / 1024, 2) for n in [16, 4, 3, 2, 1]] == [14.06, 3.52, 2.64, 1.76, 0.88]
    for row in ["| S0 | 16 / 14,06 | 4 / 3,52 |", "| S1 | 16 / 14,06 | 3 / 2,64 |",
                "| S2 | 16 / 14,06 | 2 / 1,76 |", "| S3 | 16 / 14,06 | 1 / 0,88 |"]:
        assert row in text
    print("PASS BatchNorm counterexample, clipping, token loss, flex_pp contract and exact traffic counts")
    message = 2 * 2 * 4096 * 8192
    assert message == 134_217_728 and message / 2**20 == 128
    assert math.isclose(message / 25e9 * 1000, 5.36870912)
    assert 2 * 16 * message / 2**30 == 4
    assert 8 * 16 * 2 == 256
    assert math.isclose(19 * (0.09 + 0.18), 5.13)
    assert math.isclose(16 * (0.09 + 0.18), 4.32)
    assert math.isclose(3 / 19 * 100, 15.789473684210526)
    assert math.isclose(3 / 35 * 100, 8.571428571428571)
    assert [round(n * 900 / 1024, 2) for n in [16, 4, 3, 2, 1]] == [14.06, 3.52, 2.64, 1.76, 0.88]
    for source in ["lecture-07-parallelism-strategy-comparison", "lecture-08-pipeline-parallelism",
                   "lecture-08-pipeline-bubble-failure"]:
        assert source in text
    assert '<a id="cs336-distributed-pipeline"></a>' in text
    for anchor in ["gpipe-complete-p4-m4-schedule", "1f1b-warm-up-steady-state-и-drain",
                   "stage-balance-bubble-и-память", "trade-offs-и-верификация",
                   "что-меняют-zero-bubble-и-dualpipe"]:
        assert f'<a id="{anchor}"></a>' in text
    # The first Device 1 row is transcribed from the complete author-published
    # ZB2P figure. This checks its worked count, not a distributed ZB runtime.
    zb2p_first = ([f"F{mu}" for mu in range(1, 8)] + ["B1", "W1", "F8"]
                 + [f"{kind}{mu}" for mu in range(2, 9) for kind in "BW"])
    assert len(zb2p_first) == 24
    assert all(sum(op.startswith(kind) for op in zb2p_first) == 8 for kind in "FBW")
    live = peak = 0
    for op in zb2p_first:
        live += 1 if op.startswith("F") else -1 if op.startswith("B") else 0
        peak = max(peak, live)
        assert live >= 0
    assert live == 0 and peak == 7
    v_devices = [0, 1, 2, 3, 3, 2, 1, 0]
    assert sum(a != b for a, b in zip(v_devices, v_devices[1:])) == 6
    assert len(re.findall(r"!\[\[", text)) == 9
    for name in ["dependency.png", "zb1p.png", "zb2p.png", "zbv.png", "optimizer.png"]:
        assert f"zero-bubble-2026-10-23/{name}" in text
    assert "harvard/distributed/pipeline-parallelism.svg" not in text
    print("PASS actual tables, 256 schedule configurations, negative controls, token weighting, arithmetic and sources")
    print(json.dumps({"torch": torch.__version__, "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      "actual_python_fences": len(examples), "figure_count": 9,
                      "partition_oracle_inputs": cases}, indent=2))
    print("CPU single-process simulation only; no process-group, NCCL, GPU memory or throughput measurement.")


if __name__ == "__main__":
    main()
