"""Execute the published tool-reward examples and compare to pinned ToolRL code.

Standard library only. No model, tool execution, sandbox or training claim.
"""
from __future__ import annotations

import ast
from collections import Counter
import contextlib
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md"
SOURCE = ROOT / "05 Источники/Originals/tool-rewards-2026-10-29/verl__utils__reward_score__rlla.py"


def main():
    text = CHAPTER.read_text()
    part = text.split("<!-- BEGIN REVIEW: tool rewards 2026-10-29 -->")[1].split(
        "<!-- END REVIEW: tool rewards 2026-10-29 -->")[0]
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", part, re.M | re.S)
    assert len(blocks) == 2
    namespace, captured = {}, io.StringIO()
    with contextlib.redirect_stdout(captured):
        for i, block in enumerate(blocks):
            exec(compile(block, f"{CHAPTER}:tool-example-{i+1}", "exec"), namespace)

    # Execute only the two reviewed pure scoring functions, with every variant off.
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
        "2ba68ec7e5e824a0ca73f5d0ac6eff2d8c7e94fcbd71f08296d345578590122d")
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("match_score", "compute_tool_call_reward")]
    assert len(functions) == 2

    class DefaultEnvironment:
        @staticmethod
        def getenv(name, default=None):
            return default

    source_ns = {"Counter": Counter, "os": DefaultEnvironment}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), source_ns)
    actual = source_ns["compute_tool_call_reward"]
    call = lambda params, name="f": {"name": name, "parameters": params}
    score = namespace["single_tool_reward"]
    comparisons = 0
    with contextlib.redirect_stdout(captured):
        for gold_values in itertools.product((0, 1), repeat=2):
            gold = call(dict(zip(("a", "b"), gold_values)))
            for key_mask in itertools.product((False, True), repeat=2):
                keys = [key for key, keep in zip(("a", "b"), key_mask) if keep]
                for values in itertools.product((0, 1), repeat=len(keys)):
                    for name in ("f", "wrong"):
                        predicted = call(dict(zip(keys, values)), name)
                        expected = actual([gold], [predicted], 3., -3.)
                        assert math.isclose(score(gold, predicted), expected, abs_tol=1e-14)
                        comparisons += 1

        assert source_ns["match_score"](["f", "f"], ["f"]) == .5
        gold = [call({"a": 0, "b": 0}), call({"a": 0, "b": 1})]
        predicted = [call({"a": 0, "b": 1}), call({"a": 1, "b": 0})]
        forward = actual(gold, predicted, 3., -3.)
        reversed_order = actual(gold, list(reversed(predicted)), 3., -3.)
        assert math.isclose(forward, 3/7, abs_tol=1e-14)
        assert math.isclose(reversed_order, 15/7, abs_tol=1e-14)
        assert forward != reversed_order

    efficiency = namespace["completion_efficiency_reward"]
    f1_cases = 0
    for required in range(1, 9):
        for solved in range(required + 1):
            for calls in range(solved, 12):
                eps = 1e-8
                recall = solved/required
                precision = solved/(calls+eps)
                expected = 2*recall*precision/(recall+precision) if solved else 0.
                assert math.isclose(efficiency(required, solved, calls), expected, abs_tol=1e-14)
                f1_cases += 1
    for arguments in ((0,0,0), (4,-1,2), (4,5,5), (4,2,1), (True,1,1), (4.,1,1)):
        try:
            efficiency(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(arguments)
    for eps in (0., -1., float("nan"), float("inf")):
        try:
            efficiency(4, 2, 2, eps)
        except ValueError:
            pass
        else:
            raise AssertionError(eps)

    print(json.dumps({
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "new_published_python_fences": len(blocks),
        "pinned_code_comparisons": comparisons,
        "f1_cases": f1_cases,
        "greedy_candidate_order_rewards": [forward, reversed_order],
        "illustration_sha256": hashlib.sha256((ROOT/
            "00 Учебник/Assets/Figures/curated/toolrl/reward-design.png").read_bytes()).hexdigest(),
        "example_stdout": captured.getvalue().splitlines()[:2],
        "model_calls": 0, "environment_actions": 0,
    }, indent=2))
    print("PASS tool reward arithmetic and pinned code comparison; not agent training")


if __name__ == "__main__":
    main()
