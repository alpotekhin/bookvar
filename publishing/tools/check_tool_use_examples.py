#!/usr/bin/env python3
"""Check chapter 65's deterministic fixtures and original ReAct crops.

Run the fixtures in both editions without a model, network or filesystem tools.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md"
ENGLISH = ROOT / "en/00 Textbook/17 Tools and Agents/65 How a function call becomes an action.md"
ARCHIVE = ROOT / "05 Источники/Originals/tool-use-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_examples(path, text):
    blocks = re.findall(
        r'^<details id="tool-use-code-([^"]+)">\n<summary>[^\n]+</summary>\n\n'
        r'^(~~~|```)(python|json)\n(.*?)^\2\n\n</details>$',
        text, re.M | re.S,
    )
    assert len(blocks) == 11 == len({name for name, _, _, _ in blocks})
    scopes, shared, sources, wire = {}, {}, {}, {}
    for name, _, kind, block in blocks:
        if kind == "json":
            wire[name] = json.loads(block)
            assert isinstance(wire[name], dict)
        else:
            namespace = {}
            compiled = compile(block, f"{path}:{name}", "exec")
            exec(compiled, namespace)
            exec(compiled, shared)
            scopes[name] = namespace
            sources[name] = ast.dump(ast.parse(block))
    assert set(wire) == {"schema", "proposal", "rejected", "observation", "mcp-request"}
    assert set(scopes) == {"loss", "sampling", "mcp-correlation", "react-vote", "retry", "episode"}
    return scopes, sources, wire


def main():
    text, english = CHAPTER.read_text(), ENGLISH.read_text()
    refs = json.loads((ARCHIVE / "references.json").read_text())
    preceding = before_topic_consolidation(text, CHAPTER)
    assert hashlib.sha256(preceding.encode()).hexdigest() == refs["chapter_sha256"], "Unreviewed chapter change"
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    scopes, sources, wire = run_examples(CHAPTER, text)
    translated, en_sources, en_wire = run_examples(ENGLISH, english)
    assert sources == en_sources and wire == en_wire

    for ns in (scopes, translated):
        fixture = ns["episode"]
        success, history = fixture["run_episode"](fixture["proposals"])
        assert success and history[0][0]["arguments"] == wire["proposal"]
        assert history[0][1]["error"] == wire["rejected"]["error"]["type"]
        assert not history[0][1]["executed"]
        assert history[1][1] == wire["observation"]
        assert history[1][0]["arguments"] == wire["mcp-request"]["params"]["arguments"]
        assert wire["mcp-request"]["id"] == 17
        schema = wire["schema"]["parameters"]
        assert schema["properties"]["start_line"]["minimum"] == wire["rejected"]["error"]["minimum"]

    losses, sampling, mcp, voting, writing, episode = (
        scopes[name] for name in ("loss", "sampling", "mcp-correlation", "react-vote", "retry", "episode")
    )
    correlate, rpc, pending = mcp["correlate_response"], mcp["rpc"], mcp["pending"]
    assert not pending
    pending.update({17: "current-read", 18: "failed-read"})
    invalid_responses = [
        {"id": 17, "result": {}},
        rpc(True, result={}),
        rpc(999, result={}),
        rpc(17),
        rpc(17, result={}, error={}),
        rpc(17, result="not an object"),
        rpc(17, result={"isError": 1}),
    ]
    for response in invalid_responses:
        assert correlate(response) is None
        assert pending == {17: "current-read", 18: "failed-read"}
    unavailable = {"code": -32602, "message": "Unknown tool"}
    assert correlate(rpc(18, error=unavailable)) == {
        "call_id": "failed-read", "kind": "rpc_error", "payload": unavailable,
    }
    result = {"structuredContent": {"lines": ["timeout=30"]}}
    assert correlate(rpc(17, result=result)) == {
        "call_id": "current-read", "kind": "tool_result", "payload": result,
    }
    assert correlate(rpc(17, result=result)) is None and not pending
    assert sampling["insertion_sites"]([.01, .08, .07, .04, .06, .03], .05, 2) == [1, 2]
    assert sampling["api_start_allowed"](["x", "[API]"], 2)
    assert not sampling["api_start_allowed"](["x", "[API]"], 2, calls_already_made=1)
    assert voting["fallback_to_react"](["A", "A", "B", "B", "C"])
    assert not voting["fallback_to_react"](["A", "A", "B", "B"])
    assert not voting["fallback_to_react"](["wrong"] * 5)
    try:
        voting["fallback_to_react"]([])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty sample group must be rejected")
    assert np.isclose(losses["baseline"], 1.5474484614172663)
    useful_delta = losses["baseline"] - losses["useful_result"]
    irrelevant_delta = losses["baseline"] - losses["irrelevant_result"]
    assert np.isclose(useful_delta, 1.3025041102867874)
    assert np.isclose(irrelevant_delta, -1.4482838121367247)
    assert losses["baseline"] == min(losses["no_call"], losses["call_without_result"])

    create = writing["create_note"]
    create("alice", "new-intent", "ReAct", "Read the paper")
    assert len(writing["notes"]) == 2
    create("bob", "task-7/create-1", "ReAct", "Read the paper")
    assert len(writing["notes"]) == 3  # caller is part of the scope

    valid = deepcopy(episode["proposals"][1])
    read = episode["read_tool"]
    result = read(valid)
    assert result["lines"] == ["timeout=30", "retries=2"]
    assert episode["verify"]("30", result)
    rejected = [
        ({"name": "delete_file"}, "unknown_tool"),
        ({"arguments": []}, "schema_error"),
        ({"arguments": {"path": "notes.txt", "start_line": True, "end_line": 2}}, "schema_error"),
        ({"arguments": {"path": "notes.txt", "start_line": 1, "end_line": 0}}, "schema_error"),
        ({"arguments": {"path": "notes.txt", "start_line": 2, "end_line": 1}}, "range_error"),
        ({"arguments": {"path": "notes.txt", "start_line": 1, "end_line": 101}}, "range_error"),
        ({"arguments": {"path": "../notes.txt", "start_line": 1, "end_line": 2}}, "permission_denied"),
        ({"arguments": {"path": "notes.txt", "start_line": 1, "end_line": 2, "extra": 0}}, "schema_error"),
    ]
    for changes, error in rejected:
        proposal = deepcopy(valid)
        proposal.update(changes)
        observation = read(proposal)
        assert observation["error"] == error
        assert observation["executed"] is False
    episode["READABLE"].add("missing.txt")
    missing = deepcopy(valid)
    missing["arguments"]["path"] = "missing.txt"
    assert read(missing)["error"] == "not_found"

    for changes in [{"truncated": True}, {"revision": "stale"},
                    {"lines": ["timeout=300", "retries=2"]},
                    {"executed": False}, {"path": "other.txt"}]:
        observation = deepcopy(result)
        observation.update(changes)
        assert not episode["verify"]("30", observation)
    assert not episode["verify"]("300", result)

    run = episode["run_episode"]
    assert run(episode["proposals"])[0]
    duplicate = deepcopy(episode["proposals"])
    duplicate[1]["call_id"] = "c1"
    assert not run(duplicate)[0]
    denied = deepcopy(episode["proposals"])
    denied[0]["arguments"]["path"] = "secret.txt"
    denied[0]["arguments"]["start_line"] = 1
    ok, history = run(denied)
    assert not ok and len(history) == 1
    assert not run([{"kind": "unexpected", "call_id": "x"}])[0]
    assert not run([], max_steps=0)[0]

    for row in refs["downloaded"] + refs["reused"] + refs["derived"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert len(refs["figures"]) == 2
    for row in refs["figures"]:
        for field, digest in [("asset", "sha256"),
                              ("source_asset", "source_asset_sha256"),
                              ("source_page_snapshot", "source_page_sha256"),
                              ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / row[field]) == row[digest]
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        crop = page.crop(tuple(row["crop_box_px"]))
        actual = Image.open(ROOT / row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert crop.size == actual.size
        assert ImageChops.difference(crop, actual).getbbox() is None
        assert row["license"] == "CC BY 4.0"
        assert "creativecommons.org/licenses/by/4.0/" in (ROOT / row["license_evidence"]).read_text()

    new_figures = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/assets.json").read_text())
    assert len(new_figures) == 2
    for row in new_figures:
        for field, digest in [("asset", "sha256"),
                              ("source_asset", "source_asset_sha256"),
                              ("source_page_snapshot", "source_page_sha256"),
                              ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / row[field]) == row[digest]
        with Image.open(ROOT / row["source_page_snapshot"]) as source:
            with Image.open(ROOT / row["asset"]) as actual:
                crop = source.crop(tuple(row["crop_box_px"])).convert("RGB")
                assert source.size == (2550, 3300)
                assert crop.size == actual.size
                assert ImageChops.difference(crop, actual.convert("RGB")).getbbox() is None

    print(json.dumps({
        "chapter_sha256": [sha(CHAPTER), sha(ENGLISH)],
        "python_blocks_per_locale": len(scopes),
        "standalone_and_shared_runs": 2 * (len(scopes) + len(translated)),
        "json_examples_parsed": len(wire) + len(en_wire),
        "toolformer_useful_delta": useful_delta,
        "toolformer_irrelevant_delta": irrelevant_delta,
        "schema_policy_and_missing_cases": len(rejected) + 1,
        "additional_rejected_final_observations": 5,
        "mcp_invalid_and_duplicate_responses": len(invalid_responses) + 1,
        "source_downloads_verified": len(refs["downloaded"]),
        "reused_course_objects_verified": len(refs["reused"]),
        "original_figure_crops_verified": len(refs["figures"]) + len(new_figures),
        "llm_run": False, "network_or_real_mutations": False,
        "agent_quality_benchmark": False
    }, indent=2))


if __name__ == "__main__":
    main()
