"""Execute the preference chapter's real fences; stdlib only, no LLM API calls."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/02 Preference data.md"


def fails(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("negative case was silently accepted")


def main():
    text = CHAPTER.read_text()
    assert "\n$\n" not in text, "display equations must retain both dollar delimiters"
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 5, "every published Python fence must execute"
    ns = {}
    example_stdout = []
    for number, code in enumerate(blocks):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, f"{CHAPTER}:python-{number+1}", "exec"), ns)
        if stream.getvalue():
            example_stdout.append(stream.getvalue().strip())

    ranking, row = ns["ranking_to_pairs"], ns["row"]
    pairs = ranking(row)
    assert {p["group_id"] for p in pairs} == {"q1"}
    assert len(pairs) == 6 and len(ranking(row, adjacent_only=True)) == 3
    assert ns["strict_pairs"] == [("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D")]
    for mode in (False, True):
        for pair in ranking(row, adjacent_only=mode):
            assert pair["chosen_policy"][-1] == pair["chosen"][-1]
            assert pair["rejected_policy"][-1] == pair["rejected"][-1]
    for replacement in ([["A"], ["B", "C"], ["D"]], ["A", "B", "C", "C"], ["A", "B", "C"]):
        fails(ranking, {**row, "ranking": replacement})
    fails(ranking, {**row, "responses": row["responses"] + [row["responses"][0]]})
    fails(ranking, {"ranking": ["A"], "responses": [row["responses"][0]]})
    assert len(ns["train_pairs"]) == 12 and len(ns["valid_pairs"]) == 6
    assert {p["group_id"] for p in ns["train_pairs"]}.isdisjoint(
        {p["group_id"] for p in ns["valid_pairs"]})

    final_pair = ns["final_turn_pair"]
    chosen, rejected = ns["chosen"], ns["rejected"]
    assert final_pair(chosen, rejected)["prompt"] == ns["history"]
    different = copy.deepcopy(rejected)
    different[1]["content"] = "Другое предыдущее объяснение"
    fails(final_pair, chosen, different)
    fails(final_pair, chosen, chosen)
    fails(final_pair, [], rejected)
    wrong_role = copy.deepcopy(rejected)
    wrong_role[-1]["role"] = "user"
    fails(final_pair, chosen, wrong_role)

    verdict = ns["normalize_verdict"]
    assert verdict('{"winner":"first"}', ["X", "Y"])["winner_id"] == "X"
    assert verdict('{"winner":"second"}', ["Y", "X"])["winner_id"] == "X"
    assert verdict('{"winner":"first"}', ["Y", "X"])["winner_id"] == "Y"
    assert verdict('{"winner":"tie"}', ["X", "Y"]) == {"status": "tie", "winner_id": None}
    for invalid in ('B, хотя A тоже неплох', '{"winner":"A"}', '{"winner":true}',
                    '{"winner":null}', '{"winner":["first"]}', '[]',
                    '{"winner":"first","extra":"second"}'):
        fails(verdict, invalid, ["X", "Y"])
    fails(verdict, '{"winner":"first"}', ["X", "X"])

    json_blocks = re.findall(r"^\x60{3}json\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(json_blocks) == 1
    original = json.loads((ROOT / "05 Источники/Originals/preference-2026-10-23/hh-helpful-test-line1.jsonl").read_text())
    delimiter = "\n\nAssistant:"
    cp, _, ca = original["chosen"].rpartition(delimiter)
    rp, _, ra = original["rejected"].rpartition(delimiter)
    assert cp == rp and ca != ra
    assert "I don’t know anything" in ca
    assert "> " + cp.removeprefix("\n\nHuman: ") in text
    assert "> " + ca.strip() in text and "> " + ra.strip() in text
    # This delimiter check covers the pinned one-turn excerpt, not an arbitrary chat parser.

    probability = 1 / (1 + math.exp(-0.9))
    assert math.isclose(probability, 0.7109495026, abs_tol=1e-10)
    assert math.isclose(math.log((2/3)/(1/3)), math.log(2))
    observed, expected = (45+35)/100, 0.55**2 + 0.45**2
    kappa = (observed-expected)/(1-expected)
    assert math.isclose(kappa, 0.59595959596, abs_tol=1e-11)
    assert math.comb(4, 2) == 6 and math.comb(8, 2) == 28
    assert math.isclose(28/34, 0.8235294118, abs_tol=1e-10)
    assert math.isclose(sum([1/6]*6), sum([1/28]*28))
    pair_mean = (6 * .2 + 28 * .8) / 34
    group_mean = (.2 + .8) / 2
    assert math.isclose(pair_mean, .6941176470588236)
    assert group_mean == .5 and not math.isclose(pair_mean, group_mean)
    assert r"L_{\mathrm{pairs}}" in text and r"L_{\mathrm{groups}}" in text
    cycle = ns["cycle_probability"]
    assert math.isclose(cycle, .09453439083199781)
    probability_of = ns["preference_probability"]
    assert all(math.isclose(probability_of(a, b) + probability_of(b, a), 1)
               for a, b in [(1.2, .3), (.3, 0), (0, 1.2)])
    rankings = [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")]
    for a, b in [("A", "B"), ("B", "C"), ("C", "A")]:
        assert sum(order.index(a) < order.index(b) for order in rankings) == 2

    def schema(value):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return (isinstance(parsed, dict) and set(parsed) == {"sum", "count"}
                and all(type(v) is int for v in parsed.values()))
    assert schema('{"sum":4,"count":2}')
    assert not schema('Результат: {"sum":4,"count":2}')
    assert not schema('{"sum":true,"count":2}')
    assert not schema('{"sum":4.0,"count":2}')
    assert not schema('{"sum":4,"count":2,"extra":0}')

    print(json.dumps({"chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "published_python_fences": len(blocks),
                      "ranking_pairs": len(pairs), "tied_ranking_strict_pairs": len(ns["strict_pairs"]),
                      "train_pairs": len(ns["train_pairs"]), "validation_pairs": len(ns["valid_pairs"]),
                      "bradley_terry_probability": probability, "kappa": kappa,
                      "independent_cycle_probability": cycle,
                      "pair_weighted_loss": pair_mean, "group_weighted_loss": group_mean,
                      "majority_cycle_votes": [2, 2, 2],
                      "example_stdout": example_stdout}, indent=2))
    print("PASS actual preference-data examples; ranking, ties, grouping, histories, verdicts, original source row, arithmetic")


if __name__ == "__main__":
    main()
