#!/usr/bin/env python3
"""Check chapter 67 fixtures and source crops; no LLM/API/agent benchmark."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import product
import json
from math import isclose, log, sqrt
from pathlib import Path
import re

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md"
ENGLISH = ROOT / "en/00 Textbook/17 Tools and Agents/67 Agent memory, planning, and orchestration.md"
ARCHIVE = ROOT / "05 Источники/Originals/agent-memory-planning-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_plan():
    # A deliberately deterministic STRIPS-like toy, not a PDDL planner.
    rules = {
        "book_room": ({"room_available"}, {"room_confirmed"}),
        "send_invitation": ({"room_confirmed", "time_confirmed"}, {"invitation_sent"}),
    }
    def execute(actions):
        state = {"room_available", "time_confirmed"}
        for action in actions:
            preconditions, effects = rules[action]
            if not preconditions <= state:
                return False, state
            state |= effects
        return True, state
    ok, state = execute(["book_room", "send_invitation"])
    assert ok and "invitation_sent" in state
    ok, state = execute(["send_invitation", "book_room"])
    assert not ok and "invitation_sent" not in state


def check_tot():
    # Exact arithmetic on the two branches explained next to original Figure 2.
    a, b = Fraction(7), Fraction(9)
    values = {a+b, a-b, b-a, a*b, a/b, b/a}
    assert values == {16, -2, 2, 63, Fraction(7, 9), Fraction(9, 7)}
    assert Fraction(24) not in values
    assert (10-4)*(13-9) == 24
    assert isclose(0.7+sqrt(log(10)/8), 1.2364915065723368)
    assert isclose(0.5+sqrt(log(10)/2), 1.5729830131446736)


def run_examples(path, text):
    blocks = re.findall(
        r'^<details id="memory-code-([^"]+)">\n<summary>[^\n]+</summary>\n\n'
        r'^(~~~|```)(python|json)\n(.*?)^\2\n\n</details>$',
        text, re.M | re.S,
    )
    assert len(blocks) == 5 == len({name for name, _, _, _ in blocks})
    scopes, shared, asts, records = {}, {}, {}, {}
    for name, _, kind, block in blocks:
        if kind == "json":
            records[name] = json.loads(block)
            continue
        code = compile(block, f"{path}:{name}", "exec")
        scope = {}
        exec(code, scope)
        exec(code, shared)
        scopes[name] = scope
        asts[name] = ast.dump(ast.parse(block))
    assert set(scopes) == {"ranking", "ppr", "reflexion", "merge"}
    assert set(records) == {"record"}
    return scopes, asts, records["record"]


def main():
    text, english = CHAPTER.read_text(), ENGLISH.read_text()
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    scopes, asts, record = run_examples(CHAPTER, text)
    en_scopes, en_asts, en_record = run_examples(ENGLISH, english)
    assert asts == en_asts and record == en_record
    source_pattern = (r'<details id="psl-source-([^"]+)">\n<summary>[^\n]+</summary>\n\n'
                      r'(.*?)\n\n</details>')
    source_excerpt = re.findall(source_pattern, text, re.S)
    assert len(source_excerpt) == 6
    assert source_excerpt == re.findall(source_pattern, english, re.S)
    assert record["supersedes"] == "seminar-42-start-v1"
    assert record["value"] == "2026-10-09T19:00:00+03:00"

    ranking, graph, reflection, merge = (
        scopes[name] for name in ("ranking", "ppr", "reflexion", "merge")
    )
    assert ranking["minmax"]([1, 3, 5]) == [0.0, 0.5, 1.0]
    expected = [2.712486074129683, 1.0, 1.875]
    assert all(isclose(a, b) for a, b in zip(ranking["scores"], expected))
    assert "2,712" in text and "0.712" in text
    assert all(isclose(sum(row), 1.0) for row in graph["P"])
    assert all(isclose(a, b) for a, b in zip(
        graph["ppr_step"]([0.075, 0.075, 0.85]), [0.43625, 0.43625, 0.1275]))

    run = reflection["try_candidates"]
    bad, good, cases = (reflection[k] for k in ["counts_only", "prefix_checked", "cases"])
    assert run([bad, good], cases, 0)["attempts"] == 0
    assert run([bad, good], cases, 1)["status"] == "budget_exhausted"
    assert run([good, bad], cases)["attempts"] == 1
    assert run([bad], [("()", True)])["status"] == "passed_given_cases"
    for invalid in [-1, 1.5, True]:
        try:
            run([bad, good], cases, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid attempt limit accepted")
    for length in range(9):
        for chars in product("()", repeat=length):
            value = "".join(chars)
            reduced = value
            while "()" in reduced:
                reduced = reduced.replace("()", "")
            assert good(value) == (reduced == "")

    accept = merge["accept_result"]
    current, room, papers = (deepcopy(merge[k]) for k in ["current", "room", "papers"])
    before = deepcopy((current, room, papers))
    assert accept(current, room)["status"] == "stale"
    assert accept(current, papers)["status"] == "accepted_for_review"
    assert (current, room, papers) == before
    thursday = {**current, "people": 30, "start": "2026-10-08T18:00:00+03:00"}
    thursday_room = {**room, "depends_on": {"people": 30, "start": thursday["start"]}}
    friday = {**thursday, "start": record["value"]}
    assert accept(thursday, thursday_room)["status"] == "accepted_for_review"
    assert accept(friday, thursday_room)["status"] == "stale"
    assert accept(friday, papers)["status"] == "accepted_for_review"
    del current["topic"]
    assert accept(current, papers)["status"] == "stale"
    assert accept({}, {"depends_on": {"missing": None}})["status"] == "stale"
    assert accept({}, {"depends_on": {}})["status"] == "stale"
    check_plan()
    check_tot()

    refs = json.loads((ARCHIVE / "references.json").read_text())
    # Retain the original accepted source record. The bounded PSL addition has
    # its own pinned-source check; all earlier prose/code must still match it.
    from import_planseqlearn_excerpt import baseline, main as check_psl
    assert refs["chapter_sha256"] == hashlib.sha256(baseline(text).encode()).hexdigest()
    check_psl()
    for row in refs["downloaded"] + refs["derived"] + refs["existing_sources"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    for figure in refs["figures"]:
        for field, digest in [("asset", "sha256"), ("source_asset", "source_asset_sha256"),
                              ("source_page_snapshot", "source_page_sha256"),
                              ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / figure[field]) == figure[digest]
        assert "License: CC BY 4.0" in (ROOT / figure["license_evidence"]).read_text()
        with Image.open(ROOT / figure["source_page_snapshot"]) as page:
            with Image.open(ROOT / figure["asset"]) as image:
                assert image.size == (figure["width"], figure["height"])
                assert page.size == tuple(figure["render_page_size_px"])
                assert ImageChops.difference(page.crop(figure["crop_box_px"]), image).getbbox() is None
    print(f"PASS: {len(scopes)} Python blocks per edition / {2*(len(scopes)+len(en_scopes))} executions; matching JSON, retrieval scores and PPR;")
    print("      both readers preserve the six original source blocks; room result invalidated by date and attendance.")
    print("      511 bracket strings; retry limits; dependency invalidation; toy plan/ToT/UCT;")
    print("      pinned source hashes and 2 pixel-identical original crops.")
    print("No LLM inference, framework integration, PDDL solver or quality benchmark executed.")


if __name__ == "__main__":
    main()
