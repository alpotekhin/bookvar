"""Execute every RLVR chapter fence and independent CPU/numerical checks.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  python publishing/tools/check_rlvr_examples.py
No language model, sandboxed model code, downloads or performance claim.
"""
from __future__ import annotations

import ast
import contextlib
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md"


def rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 6, "execute every published Python fence"
    ns, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for i, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    parse, reward, passk = [ns[k] for k in ("parse_final", "numeric_reward", "pass_at_k")]
    for answer in (None, 4., Decimal("NaN"), Decimal("Infinity")):
        rejects(reward, "FINAL: 4", answer)
    for bad in ("FINAL: .4", "FINAL: 4.", "FINAL: ٤", "FINAL: 4_000",
                "FINAL: 4\r\n\r\n", "prefix FINAL: 4", "FINAL: 4 FINAL: 4"):
        assert parse(bad) is None, bad
    for value in ("-0", "+0004.000", "4" * 64 + "." + "5" * 64):
        assert parse("FINAL: " + value) == Decimal(value)
    assert parse("FINAL: 4\r\n") == Decimal(4)
    assert parse("FINAL: 4.0000000000000000000000000000000000000000000000000000000000000000") == 4
    assert reward("3x=18-6=10; x=4\nFINAL: 4", Decimal(4))["correct"] == 1

    # Independent exact objective differentiates p, unlike the REINFORCE surrogate.
    z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.1, 0., 1.1, .1], dtype=torch.float64)
    objective = lambda x: (x.softmax(-1) * rewards).sum()
    torch.autograd.gradcheck(objective, (z,))
    grad, = torch.autograd.grad(objective(z), z)
    torch.testing.assert_close(grad, ns["ascent_gradient"])
    assert ns["updated_reward"] > ns["mean_reward"]
    assert ns["updated_probabilities"][3] < .25  # positive but below-average reward
    for baseline in (0., .575, 100.):
        q = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        surrogate = (q.softmax(-1).detach() * (rewards-baseline) * q.log_softmax(-1)).sum()
        actual, = torch.autograd.grad(surrogate, q)
        torch.testing.assert_close(actual, grad)

    pass_cases = 0
    for n in range(1, 25):
        for c in range(n+1):
            for k in range(1, n+1):
                expected = 1 - math.comb(n-c, k) / math.comb(n, k)
                assert math.isclose(passk(n, c, k), expected, abs_tol=1e-14)
                pass_cases += 1
    for values in ((0,0,1), (4,-1,1), (4,5,1), (4,1,0), (4,1,5),
                   (4.,1,1), (True,1,1), (4,1.,1), (4,1,True)):
        rejects(passk, *values)
    assert math.isclose(passk(10**14, 1, 1), 1e-14, rel_tol=1e-14)
    # Expectation over all possible counts proves the estimator's behavior here.
    for p in (.01, .1, .5, .9):
        expected = math.fsum(math.comb(10,c)*p**c*(1-p)**(10-c)*passk(10,c,3)
                             for c in range(11))
        assert math.isclose(expected, 1-(1-p)**3, abs_tol=1e-14)
    assert math.isclose(1-.01**4-.99**4, .03940398)
    assert math.isclose(1-.01**32-.99**32, .2750196640421466)
    assert math.isclose(.9**8, .43046721)
    assert math.isclose(18/113, .1592920353982301)

    # Isolate the actual course function's format gate. This does not test SymPy.
    grader = ROOT / "05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment5-alignment/cs336_alignment/drgrpo_grader.py"
    tree = ast.parse(grader.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "r1_zero_reward_fn")
    calls = []
    def recording_grade(answer, gold, fast):
        calls.append((answer, gold, fast))
        return answer.strip() == gold  # fixture only, not a substitute symbolic verifier
    module = ast.Module(body=[function], type_ignores=[])
    isolated = {"grade": recording_grade}
    exec(compile(module, str(grader), "exec"), isolated)
    grade_format = isolated["r1_zero_reward_fn"]
    assert grade_format("4</think> <answer>4</answer>", "4")["reward"] == 1
    for response in ("4</think>\n<answer>4</answer>", "4</think> <answer>4"):
        before = len(calls)
        assert grade_format(response, "4")["format_reward"] == 0
        assert len(calls) == before
    assert grade_format("4</think> <answer>5</answer>", "4") == {
        "format_reward": 1., "answer_reward": 0., "reward": 0.}

    print(json.dumps({"python_fences":len(blocks), "torch":torch.__version__,
        "device":"cpu", "gradient":grad.tolist(), "reward_before":ns["mean_reward"].item(),
        "reward_after":ns["updated_reward"].item(), "pass_at_k_cases":pass_cases,
        "course_check":"format gate only; symbolic grading not executed",
        "example_stdout":output.getvalue().splitlines(),
        "chapter_sha256":hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print("PASS actual RLVR fences; no model training, unsafe code execution or quality claim")


if __name__ == "__main__":
    main()
