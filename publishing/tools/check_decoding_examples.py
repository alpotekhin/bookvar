"""Execute all actual chapter fences and independent CPU checks.

uv run --offline --with torch==2.14.1 --with transformers==4.57.1 \
  --with numpy==2.3.5 python publishing/tools/check_decoding_examples.py
No pretrained weights, endpoint, benchmark, or GPU measurements.
"""
from __future__ import annotations

import contextlib
from fractions import Fraction
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re

import torch
from transformers.generation import beam_search, logits_process, stopping_criteria, utils

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена.md"
ARCHIVE = ROOT / "05 Источники/Originals/decoding-2026-10-25"
SOURCES = [
    (logits_process, "transformers-logits-process.py", "6481506629ff6f021157e893fc64c4e7b5b06a72072771809d679387cb54a4b1"),
    (stopping_criteria, "transformers-stopping-criteria.py", "495628e4c877f667fbbd9ed4cc83b680de9eac2abc97ec3903f6e9219af3ecd2"),
    (beam_search, "transformers-beam-search.py", "918320831d0312b70d709ca1424e64d5971da9eb99690e0ec6a8ed76978a6612"),
    (utils, "transformers-generation-utils.py", "a20024b1e82ed5361a524d238d2197be5407abc91297dd9888c57e8284d63fef"),
]


def children(prefix):
    """The explicitly labelled editorial beam counterexample, not an LLM."""
    if not prefix:
        return [("A", Fraction(1, 2)), ("B", Fraction(2, 5)), ("C", Fraction(1, 10))]
    if len(prefix) == 1:
        count = {"A": 5, "B": 2, "C": 1}[prefix[0]]
        return [(str(i), Fraction(1, count)) for i in range(count)]
    if prefix[0] == "A":
        return [("x", Fraction(99, 100)), ("y", Fraction(1, 100))]
    count = 10 if prefix[0] == "B" else 1
    return [(str(i), Fraction(1, count)) for i in range(count)]


def tree_search(width):
    paths = [((), Fraction(1))]
    for _ in range(3):
        expanded = []
        for prefix, probability in paths:
            choices = children(prefix)
            assert sum(p for _, p in choices) == 1
            expanded.extend(((*prefix, token), probability*p) for token, p in choices)
        paths = sorted(expanded, key=lambda item: item[1], reverse=True)
        if width is not None:
            paths = paths[:width]
    return paths


def main():
    torch.set_num_threads(1)
    assert importlib.metadata.version("transformers") == "4.57.1"
    for module, name, expected in SOURCES:
        archived = (ARCHIVE/name).read_bytes()
        assert hashlib.sha256(archived).hexdigest() == expected
        assert Path(module.__file__).read_bytes() == archived, name
    text = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~\s*$", text, re.M | re.S)
    assert len(blocks) == 6
    ns, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, code in enumerate(blocks, 1):
            exec(compile(code, f"{CHAPTER}:python-{index}", "exec"), ns)
    f64 = torch.float64
    p = ns["probabilities"]
    torch.testing.assert_close(torch.softmax(torch.tensor([2., 1., 0.], dtype=f64), -1),
                               torch.softmax(torch.tensor([1002., 1001., 1000.], dtype=f64), -1))
    for temperature in (.1, .5, 1., 2., 10.):
        q = ns["probabilities_at"](temperature)
        expected = p.pow(1/temperature)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(q, expected)
        assert q.argmax(-1).item() == p.argmax(-1).item()
    for invalid in (0, -1, float("nan"), float("inf")):
        try:
            ns["probabilities_at"](invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid temperature accepted")
    # The upstream pedagogical image is not an exact common-temperature rescale.
    pair_temperatures = (math.log(.5/.4)/math.log(.75/.23),
                         math.log(.5/.1)/math.log(.75/.02))
    assert math.isclose(pair_temperatures[0], .189, abs_tol=.0005)
    assert math.isclose(pair_temperatures[1], .444, abs_tol=.0005)
    assert not math.isclose(*pair_temperatures, rel_tol=.01)
    expected = {
        "top-k=2": [[4/7, 3/7, 0, 0, 0]],
        "top-p=0.8": [[4/9, 3/9, 2/9, 0, 0]],
        "min-p=0.15": [[20/49, 15/49, 10/49, 4/49, 0]],
    }
    for name, values in expected.items():
        torch.testing.assert_close(ns["filtered_probabilities"][name], torch.tensor(values, dtype=f64))
    torch.testing.assert_close(ns["k_then_p"], torch.tensor([[1, 0, 0, 0, 0]], dtype=f64))
    torch.testing.assert_close(ns["p_then_k"], torch.tensor(expected["top-k=2"], dtype=f64))
    tied = torch.tensor([[2., 1., 1., 0.]])
    assert torch.isfinite(logits_process.TopKLogitsWarper(2)(ns["history"], tied)).sum() == 3
    for probabilities, count in [([.9, .04, .03, .02, .01], 1),
                                 ([.24, .22, .20, .18, .16], 5)]:
        scores = torch.tensor([probabilities], dtype=f64).log()
        assert torch.isfinite(logits_process.MinPLogitsWarper(.1)(ns["history"], scores)).sum() == count
    for temperature in (.5, 1., 2.):
        for threshold in (.05, .15, .4):
            z = p.log()/temperature
            result = logits_process.MinPLogitsWarper(threshold)(ns["history"], z)
            expected_keep = z >= z.max(-1, keepdim=True).values + math.log(threshold)
            assert torch.equal(torch.isfinite(result), expected_keep)
    torch.testing.assert_close(ns["penalized"], torch.tensor([[-2., 2/1.2, .5, -.6, 1.]], dtype=f64))
    torch.testing.assert_close(ns["without_prompt"], torch.tensor([[-2., 2., .5, -.6, 1.]], dtype=f64))
    # Duplicate history entries do not compound the multiplicative penalty.
    once = logits_process.RepetitionPenaltyLogitsProcessor(1.2)(
        torch.tensor([[1, 3]]), ns["penalty_logits"])
    torch.testing.assert_close(once, ns["penalized"])
    shifted = logits_process.RepetitionPenaltyLogitsProcessor(1.2)(
        ns["penalty_history"], ns["penalty_logits"]+2)
    assert not torch.allclose(shifted.softmax(-1), ns["penalized"].softmax(-1))
    assert math.isclose(2*.3+.2, .8)
    assert torch.isfinite(ns["ngram_scores"]).tolist() == [[True, False, True, True, True]]
    assert ns["criterion"](torch.tensor([[ns["vocabulary"]["stop"]]]), None).item()
    assert not ns["criterion"](torch.tensor([[ns["vocabulary"]["op"]]]), None).item()
    # Genuine generation output scores match the configured processor chain.
    first_scores = ns["samples"].scores[0]
    expected_first = logits_process.TopPLogitsWarper(.9)(
        ns["prompt"].repeat(3, 1), ns["first_logits"].repeat(3, 1)/.8)
    torch.testing.assert_close(first_scores, expected_first)
    for scores in ns["samples"].scores:
        torch.testing.assert_close(scores.softmax(-1).sum(-1), torch.ones(3))
    assert ns["greedy"].shape == (1, 7) and ns["stopped"].shape == (1, 4)
    assert ns["samples"].sequences.shape == (3, 7)
    assert ns["beams"].sequences.shape == (2, 7)
    # Exact probability arithmetic: beam pruning need not dominate greedy.
    greedy, beam, exhaustive = tree_search(1), tree_search(2), tree_search(None)
    assert greedy[0][1] == Fraction(99, 1000)
    assert beam[0][1] == Fraction(1, 50)
    assert len(exhaustive) == 31 and exhaustive[0][1] == Fraction(1, 10)
    assert math.log(.3) > math.log(.2)
    assert math.log(.3)/2 < math.log(.2)/3
    assert Fraction(6, 10)*Fraction(1, 10) == Fraction(6, 100)
    assert Fraction(4, 10)*Fraction(9, 10) == Fraction(36, 100)
    assert Fraction(6, 100) / Fraction(42, 100) == Fraction(1, 7)
    assert math.isclose(1-.99**100, .634, abs_tol=.0005)
    print(json.dumps({
        "python_fences": len(blocks),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
        "random_model_parameters": sum(p.numel() for p in ns["model"].parameters()),
        "beam_counterexample": {"greedy": float(greedy[0][1]), "beam_2": float(beam[0][1]),
                                "exhaustive": float(exhaustive[0][1]), "paths": len(exhaustive)},
        "stdout": output.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }, indent=2))
    print("PASS actual chapter fences, pinned Transformers, beam example and stopping; not a quality benchmark")


if __name__ == "__main__":
    main()
