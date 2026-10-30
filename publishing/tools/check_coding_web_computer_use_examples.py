#!/usr/bin/env python3
"""Validate chapter 69 fixtures and source provenance; no browser, agent or LLM run."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import product
import json
from math import isclose
from pathlib import Path
import re

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md"
ARCHIVE = ROOT / "05 Источники/Originals/coding-web-computer-use-2026-10-21"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = CHAPTER.read_text()
    blocks = [m.group(2) for m in re.finditer(
        r"^(```|~~~)python\n(.*?)^\1\s*$", text, re.M | re.S)]
    assert len(blocks) == 4
    scopes, shared = [], {}
    for index, block in enumerate(blocks):
        code = compile(block, f"{CHAPTER}:example-{index+1}", "exec")
        scope = {}
        exec(code, scope)
        exec(code, shared)
        scopes.append(scope)
    branch, search, broken, fixed = scopes
    assert branch["root"]["snapshot"] == {"x": 10}
    assert [child["snapshot"]["x"] for child in branch["children"]] == [11, 20]
    branch["children"][0]["snapshot"]["x"] = 99
    assert branch["children"][1]["snapshot"]["x"] == 20
    assert branch["root"]["snapshot"]["x"] == 10

    run, replay = search["best_first_fixture"], search["replay"]
    before = deepcopy((search["transitions"], search["values"]))
    assert run()["visited"] == [
        ("start", 0.5), ("advert", 0.8), ("dead_end", 0.0),
        ("product", 0.6), ("answer", 1.0)]
    assert run(budget=2)["path"] == ("open_first",)
    assert run(threshold=0.7)["state"] == "advert"
    assert run(depth=0)["state"] == "start"
    assert run(budget=0)["path"] is None
    search_cases = 0
    for budget, depth, threshold in product(range(9), range(5), [0, 0.7, 0.95, 1.1]):
        result = run(budget, depth, threshold)
        assert len(result["visited"]) <= budget
        if result["path"] is not None:
            assert len(result["path"]) <= depth
            assert replay(result["path"]) == result["state"]
            assert result["value"] == max(value for _, value in result["visited"])
            assert result["value"] == search["values"][result["state"]]
        else:
            assert budget == 0 and result["value"] is None
        if result["reason"] == "threshold":
            assert result["visited"][-1][1] >= threshold
        search_cases += 1
    for keyword in ["budget", "depth"]:
        for invalid in [-1, 1.5, True, "2", None]:
            try:
                run(**{keyword: invalid})
            except ValueError:
                pass
            else:
                raise AssertionError((keyword, invalid))
    assert before == (search["transitions"], search["values"])

    old, repaired = broken["parse_assignment"], fixed["parse_assignment"]
    try:
        old("token=a=b")
    except ValueError as error:
        assert "too many values" in str(error)
    else:
        raise AssertionError("original failure was not reproduced")
    parser_cases = 0
    for length in range(8):
        for chars in product(" a=", repeat=length):
            line = "".join(chars)
            if "=" in line:
                split = line.index("=")
                expected = (line[:split].strip(), line[split+1:].strip())
                assert repaired(line) == expected
                if line.count("=") == 1:
                    assert old(line) == expected
            else:
                try:
                    repaired(line)
                except ValueError:
                    pass
                else:
                    raise AssertionError(line)
            parser_cases += 1
    assert isclose(0.9**10, 0.3486784401)
    assert (4 + 12*Fraction(1, 2)) / 20 == Fraction(1, 2)
    assert Fraction(264-189, 189) == Fraction(25, 63)
    assert isclose((26.4-18.9)/18.9*100, 39.682539682539684)
    assert Fraction(146, 1000)*150_000 == 21_900
    assert Fraction(167, 1000)*150_000 == 25_050
    assert [2*abs(score-Fraction(1, 2)) for score in [0, 1]] == [1, 1]
    assert (7463-500) == 6963
    assert (Fraction(50, 100)-Fraction(35, 100))*100 == 15
    assert not (Fraction(45, 100)-Fraction(35, 100) > Fraction(10, 100))
    assert (1170*Fraction(960, 1440), 210*Fraction(600, 900)) == (780, 140)

    refs = json.loads((ARCHIVE / "references.json").read_text())
    assert sha(CHAPTER) == refs["chapter_sha256"]
    for row in refs["downloaded"] + refs["derived"] + refs["existing_sources"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert len(refs["figures"]) == 1 and len(refs["external_figures"]) == 3
    for row in refs["figures"]:
        for key, digest in [("asset", "sha256"), ("source_asset", "source_asset_sha256"),
                            ("source_page_snapshot", "source_page_sha256"),
                            ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / row[key]) == row[digest]
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        asset = Image.open(ROOT / row["asset"]).convert("RGB")
        crop = page.crop(row["crop_box_px"])
        assert list(page.size) == row["render_page_size_px"]
        assert asset.size == crop.size == (2520, 2235)
        assert ImageChops.difference(asset, crop).getbbox() is None
        assert 'href="http://creativecommons.org/licenses/by/4.0/"' in (
            ROOT / row["license_evidence"]).read_text()
    for row in refs["external_figures"]:
        assert text.count("]("+row["url"]+")") == 2
        assert row["mode"] == "external-original-no-local-copy"
    assert refs["excluded_course_pages"]["meeting-06-slides"] == [117, 118, 119, 120, 121]
    assert "last_verified: 2026-10-21" in text
    print(f"PASS: 4 code blocks individually/shared (8 executions); {search_cases} search cases; "
          f"{parser_cases} parser strings; branch independence; arithmetic and coordinates; "
          "source hashes; exact original TACO crop. No LLM/agent/browser benchmark or exploit run.")


if __name__ == "__main__":
    main()
