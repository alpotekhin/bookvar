#!/usr/bin/env python3
"""Check actual chapter 44b code/calculations on CPU; not a CUDA benchmark."""
import ast
from contextlib import redirect_stdout
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload.md"
TEXT = CHAPTER.read_text(encoding="utf-8")
FENCE = chr(96) * 3
RESULTS = {"torch": torch.__version__, "device": "cpu", "cuda_tested": False}
RTOL = ATOL = 1e-12


def need(pattern, text=TEXT):
    found = re.search(pattern, text, re.S | re.M)
    assert found, f"Missing chapter expression: {pattern}"
    return found


def block(name):
    return need(r"<!-- example:" + re.escape(name) + r" -->\s*" +
                FENCE + r"python\n(.*?)\n" + FENCE).group(1)


def execute(name):
    namespace, output = {}, io.StringIO()
    with redirect_stdout(output):
        exec(compile(block(name), f"{CHAPTER}#{name}", "exec"), namespace)
    return namespace, output.getvalue().strip()


def table(header):
    lines = TEXT.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().replace("*", "")
                     for cell in line.strip("|").split("|")])
    assert rows, header
    return rows


def close(a, b, tolerance=1e-12):
    assert abs(a - b) <= tolerance, (a, b, tolerance)


def tensor_close(a, b):
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)


def saved_tensors():
    ns, output = execute("checkpoint-saved-tensors")
    events = ns["saved"]
    assert len(events) == 2
    logical = sum(x["logical_bytes"] for x in events)
    unique = sum(ns["unique_storage"].values())
    displayed = tuple(map(int, need(
        r"равна \*\*(\d+) байтам\*\*.*?занимает \*\*(\d+) байта\*\*").groups()))
    assert (logical, unique) == displayed
    grad = ast.literal_eval(need(r"градиент равен " + chr(96) + r"(\[[\d, ]+\])").group(1))
    tensor_close(ns["x"].grad, torch.tensor(grad, dtype=torch.float64))
    RESULTS["saved_tensors"] = {"events": len(events), "logical_bytes": logical,
                               "unique_storage_bytes": unique, "stdout": output}


def view_storage_and_policy_budget():
    ns, output = execute("checkpoint-view-storage")
    base, view, compact = (ns[key] for key in ("base", "view", "compact"))
    assert view.numel() * view.element_size() == compact.numel() * compact.element_size() == 4
    assert view.untyped_storage().nbytes() == base.untyped_storage().nbytes() == 2**22
    assert compact.untyped_storage().nbytes() == 4
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert compact.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    for tensor in (view, compact):
        tensor_close(tensor.grad, torch.tensor([2.], dtype=torch.float32))
    RESULTS["view_storage"] = {"logical_bytes": 4, "view_storage_bytes": 2**22,
                               "clone_storage_bytes": 4, "stdout": output}
    policy, output = execute("checkpoint-policy-budget")
    expected = [(15, 0, "none"), (8, 40, "BC"), (7, 64, "A"),
                (6, 72, "AC"), (1, 96, "AB"), (0, 104, "ABC")]
    assert [policy["choose_saved"](b) for b in (0, 40, 64, 72, 96, 104)] == expected
    # Independent bitmask oracle also checks every intermediate budget and ties.
    for budget in range(105):
        candidates = []
        for bits in range(8):
            chosen = [i for i in range(3) if bits & (1 << i)]
            size = sum((64, 32, 8)[i] for i in chosen)
            cost = 15 - sum((8, 6, 1)[i] for i in chosen)
            names = "".join("ABC"[i] for i in chosen) or "none"
            if size <= budget:
                candidates.append((cost, size, names))
        assert policy["choose_saved"](budget) == min(candidates)
    import torch._functorch.config as compiler_config
    assert compiler_config.activation_memory_budget == 1.0
    RESULTS["policy_budget"] = {"all_105_integer_budgets": "PASS", "stdout": output,
                                "compiler_config_exists": True, "compiled_execution_tested": False}


def memory_and_transfers():
    ns, _ = execute("checkpoint-memory")
    rows = table("| $k$ | Границы + пересчёт")
    for row in rows:
        assert len(row) == 3
        displayed = [float(value) for value in row[1].split("+")] + [float(row[2])]
        assert len(displayed) == 3
        for actual, expected in zip(ns["activation_estimate"](int(row[0])), displayed):
            close(actual, expected, 0.005)
    rejected = 0
    for k in (0, -1, 5, 49, 1.5, True, None):
        try:
            ns["activation_estimate"](k)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(("invalid segment accepted", k))
    divisors = [k for k in range(1, ns["layers"] + 1) if ns["layers"] % k == 0]
    best = min(ns["activation_estimate"](k)[2] for k in divisors)
    assert [k for k in divisors if ns["activation_estimate"](k)[2] == best] == [6, 8]
    close(ns["layers"] * ns["saved_mib"] / 1024,
          float(need(r"сохранения 48 блоков занимают (\d+) GiB").group(1)))
    dims = list(map(int, need(
        r"(\d+)\\cdot(\d+)\\cdot(\d+)\\cdot(\d+)\\ \\text\{байта\}=(\d+)\\ \\text\{MiB\}"
    ).groups()))
    close(math.prod(dims[:-1]) / 2**20, dims[-1])
    offload_rows = table("| Категория | CPU, GiB")
    values = [(float(row[1]), float(row[2])) for row in offload_rows]
    totals = tuple(sum(v[i] for v in values[:-1]) for i in (0, 1))
    assert totals == values[-1]
    pinned_mib = int(need(r"Два pinned-буфера по (\d+) MiB").group(1))
    gpu_mib = int(need(r"Два GPU-буфера по (\d+) MiB").group(1))
    boundaries, restored, _ = ns["activation_estimate"](6)
    assert values[0] == (boundaries, 0)
    assert values[1] == (2 * pinned_mib / 1024, 0)
    assert values[2] == (0, 2 * gpu_mib / 1024)
    assert values[3] == (0, restored)
    assert values[4] == (0, ns["temporary_mib"] / 1024)
    budget = float(need(r"\*\*(\d+) GiB GPU-памяти\*\*").group(1))
    assert totals[1] < budget
    transfer, output = execute("checkpoint-transfer")
    one_way = transfer["one_way_seconds"]
    close(one_way, float(need(r"=\s*(0\.\d+)\\ \\text\{с\}").group(1)), 0.00005)
    close(transfer["exposed_seconds"],
          float(need(r"-(?:0\.\d+)=(0\.\d+)\$ с").group(1)), 0.00005)
    close(2 * one_way, float(need(r"составляет около (\d+\.\d+) с").group(1)), 0.00005)
    boundary_transfer = 2 * boundaries * 2**30 / (transfer["bandwidth_gbs"] * 10**9)
    close(boundary_transfer,
          float(need(r"составила бы около (\d+\.\d+) с").group(1)), 0.00005)
    for row in table("| Схема из лекции"):
        assert len(row) == 2
        counts = list(map(int, re.findall(r"\d+", row[1])))
        assert len(counts) == 5
        assert sum(counts[:4]) == counts[4]
    RESULTS["memory"] = {"segment_table": rows, "invalid_k_rejected": rejected,
                         "offload_cpu_gpu_gib": totals, "input_mib": dims[-1]}
    RESULTS["transfer_seconds"] = {"one_way": one_way, "exposed": transfer["exposed_seconds"],
                                  "round_trip": 2 * one_way, "boundaries": boundary_transfer,
                                  "stdout": output}


def rng_and_selective():
    ns, output = execute("checkpoint-rng")
    for actual, expected in zip(ns["kept_rng"], ns["baseline"]):
        tensor_close(actual, expected)
    tensor_close(ns["changed_rng"][0], ns["baseline"][0])
    delta = (ns["changed_rng"][1] - ns["baseline"][1]).abs().max().item()
    assert delta > 0
    close(delta, float(need(r"получается " + chr(96) + r"0\.0 (\d+\.\d+)").group(1)))
    RESULTS["rng"] = {"preserved_gradient_max_error": 0, "changed_gradient_max_error": delta,
                       "stdout": output}
    ns, _ = execute("checkpoint-selective")

    class CountMM(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default:
                self.calls += 1
            return func(*args, **(kwargs or {}))

    results, counts = {}, {}
    for mode in ("eager", "plain", "selective"):
        with CountMM() as counter:
            results[mode] = ns["selective_run"](mode)
        counts[mode] = counter.calls
    assert counts == {"eager": 3, "plain": 4, "selective": 3}, counts
    errors = {}
    for mode in ("plain", "selective"):
        for a, b in zip(results[mode], results["eager"]):
            tensor_close(a, b)
        errors[mode] = max((a - b).abs().max().item()
                          for a, b in zip(results[mode], results["eager"]))
    # Finite differences use the inner function from the actual Markdown code.
    outer = next(n for n in ast.parse(block("checkpoint-selective")).body
                 if isinstance(n, ast.FunctionDef) and n.name == "selective_run")
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef))
    module = ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(CHAPTER), "exec"), scope)
    region = scope["region"]
    torch.manual_seed(7)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: checkpoint(region, a, b, use_reentrant=False),
        (x, w), eps=1e-6, atol=1e-5, rtol=1e-3)
    # SAC consumes its saved-result cache on backward. gradcheck reuses the
    # graph, so check numerical derivatives with fresh forward graphs instead.
    numerical = []
    eps = 1e-6
    def fresh_selective(a, b):
        contexts = partial(create_selective_checkpoint_contexts,
                           [torch.ops.aten.mm.default])
        return checkpoint(region, a, b, use_reentrant=False,
                          context_fn=contexts).sum().item()
    for position, tensor in enumerate((x, w)):
        numeric = torch.zeros_like(tensor)
        for index in range(tensor.numel()):
            plus = [x.detach().clone(), w.detach().clone()]
            minus = [x.detach().clone(), w.detach().clone()]
            plus[position].view(-1)[index] += eps
            minus[position].view(-1)[index] -= eps
            numeric.view(-1)[index] = (fresh_selective(*plus) - fresh_selective(*minus)) / (2 * eps)
        numerical.append(numeric)
        torch.testing.assert_close(numeric, results["selective"][position + 1],
                                   atol=1e-5, rtol=1e-3)
    RESULTS["selective"] = {"mm_forward_backward_counts": counts,
                            "gradient_max_error": errors, "finite_difference_gradcheck": "PASS",
                            "gradcheck_eps_atol_rtol": [1e-6, 1e-5, 1e-3],
                            "selective_fresh_graph_finite_difference_max_error": max(
                                (a - b).abs().max().item()
                                for a, b in zip(numerical, results["selective"][1:]))}
    seen = {}
    for respect in (False, True):
        shapes = []
        def pack(t):
            shapes.append(tuple(t.shape))
            return t.detach()
        contexts = partial(create_selective_checkpoint_contexts, [torch.ops.aten.mm.default])
        a, b = x.detach().requires_grad_(), w.detach().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = checkpoint(region, a, b, use_reentrant=False, context_fn=contexts,
                             respect_saved_tensors_hooks=respect)
        out.sum().backward()
        seen[str(respect)] = shapes
    assert (3, 5) not in seen["False"] and (3, 5) in seen["True"], seen
    RESULTS["selective_saved_tensor_hooks_shapes"] = seen


def semantics_and_cpu_context():
    ns, _ = execute("checkpoint-save-on-cpu")
    for a, b in zip(ns["restored"], ns["reference"]):
        tensor_close(a, b)
    assert ns["restored"][1].device.type == "cpu"
    RESULTS["save_on_cpu"] = {"gradient_max_error": 0, "actual_gpu_transfer": False}
    # Test the chapter's analytic derivatives, including frozen W.
    torch.manual_seed(11)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    g = torch.randn(3, 5, dtype=torch.float64)
    dx, dw = torch.autograd.grad((x @ w * g).sum(), (x, w))
    tensor_close(dx, g @ w.T)
    tensor_close(dw, x.T @ g)
    frozen_w = w.detach()
    (dx_frozen,) = torch.autograd.grad((x @ frozen_w * g).sum(), (x,))
    tensor_close(dx_frozen, dx)
    RESULTS["linear_derivatives_including_frozen_weight"] = "PASS"
    data, weight = x.detach(), w.detach().requires_grad_()
    fn = lambda input_data: torch.sin(input_data @ weight).sum()
    (baseline_grad,) = torch.autograd.grad(fn(data), (weight,))
    (checkpoint_grad,) = torch.autograd.grad(
        checkpoint(fn, data, use_reentrant=False), (weight,))
    tensor_close(checkpoint_grad, baseline_grad)
    RESULTS["non_reentrant_input_without_grad_and_autograd_grad"] = "PASS"
    # Diagnostic event recorder only, not a recommended side-effectful model.
    traces = {}
    for early in (True, False):
        events = []
        def fn(t):
            events.append("entry")
            out = t.sin().sin()
            events.append("exit")
            return out
        t = torch.ones(3, dtype=torch.float64, requires_grad=True)
        checkpoint(fn, t, use_reentrant=False, early_stop=early).sum().backward()
        traces[str(early)] = events
    assert traces["True"] == ["entry", "exit", "entry"]
    assert traces["False"] == ["entry", "exit", "entry", "exit"]
    RESULTS["early_stop_diagnostic"] = traces


def replay_prefixes():
    # Verify the prose's no-boundary algorithm by explicit local VJPs. CPU
    # numerical equivalence does not measure peak CUDA memory or runtime.
    torch.manual_seed(23)
    source = torch.randn(2, 4, dtype=torch.float64)
    weights = [(torch.randn(4, 4, dtype=torch.float64) / 4).requires_grad_()
               for _ in range(8)]
    value = source.detach().requires_grad_()
    reference_input = value
    for weight in weights:
        value = torch.sin(value @ weight)
    reference = torch.autograd.grad(value.sum(), (reference_input, *weights))
    with torch.no_grad():
        value = source
        for weight in weights:
            value = torch.sin(value @ weight)
    upstream = torch.ones_like(value)
    gradients = [None] * len(weights)
    forward_calls = len(weights)
    for index in reversed(range(len(weights))):
        with torch.no_grad():
            prefix = source
            for weight in weights[:index]:
                prefix = torch.sin(prefix @ weight)
                forward_calls += 1
        local_input = prefix.detach().requires_grad_()
        local_output = torch.sin(local_input @ weights[index])
        forward_calls += 1
        upstream, gradients[index] = torch.autograd.grad(
            local_output, (local_input, weights[index]), grad_outputs=upstream)
    for a, b in zip((upstream, *gradients), reference):
        tensor_close(a, b)
    assert forward_calls == 8 + sum(range(1, 9)) == 44
    RESULTS["prefix_replay"] = {"layers": 8, "forward_block_calls": forward_calls,
                                "input_and_all_weight_gradients": "PASS",
                                "gpu_memory_or_runtime_measured": False}


def chapter_integrity():
    for unit in ("lecture-05-activation-recomputation", "lecture-08-activation-memory",
                 "assignment-02-task-memory-profiling", "assignment-02-task-gradient-checkpointing"):
        assert "  - " + unit in TEXT
    assert '<a id="cs336-distributed-checkpointing"></a>' in TEXT
    assert '<a id="worked-configuration"></a>' in TEXT
    assert '<a id="offload--другая-граница"></a>' in TEXT
    assert TEXT.count(FENCE) % 2 == 0
    assert TEXT.count(FENCE + "python") == 8
    figures = re.findall(r"!\[\[(.*?)\]\]", TEXT)
    assert len(figures) == 5
    for link in figures:
        path = ROOT / link.removeprefix("02 Areas/ML & DL/")
        assert path.is_file(), path
    RESULTS["chapter_sha256"] = hashlib.sha256(TEXT.encode()).hexdigest()
    RESULTS["figure_count"] = len(figures)


if __name__ == "__main__":
    saved_tensors()
    view_storage_and_policy_budget()
    memory_and_transfers()
    rng_and_selective()
    semantics_and_cpu_context()
    replay_prefixes()
    chapter_integrity()
    print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    print("PASS: chapter examples and CPU semantics; CUDA memory/transfer/performance untested.")
