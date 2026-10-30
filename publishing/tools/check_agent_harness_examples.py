#!/usr/bin/env python3
"""Check chapter 66's fixtures, pinned pure condenser methods and source crops.

No SDK integration, LLM API, actual tokenizer, sandbox or performance benchmark.
The OpenHands methods are read from the pinned MIT-licensed source archive.
"""
import ast
from copy import deepcopy
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md"
ARCHIVE = ROOT / "05 Источники/Originals/agent-harness-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned_condenser():
    source = ast.parse((ARCHIVE / "openhands-llm-summarizing.py").read_text())
    base = ast.parse((ARCHIVE / "openhands-condenser-base.py").read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef)
                    and n.name == "LLMSummarizingCondenser")
    names = {"_effective_max_tokens", "get_condensation_reasons",
             "condensation_requirement", "_get_forgotten_events"}
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef)
                and n.name in names]
    assert len(selected) == len(names)
    code_class = ast.ClassDef(name=original.name, bases=[], keywords=[],
                             body=selected, decorator_list=[])
    body = [ast.ImportFrom(module="__future__",
                           names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {"Reason", "_leading_system_prompt_index"}]
    body += [n for n in base.body if isinstance(n, ast.ClassDef)
             and n.name == "CondensationRequirement"]
    body.append(code_class)

    class SystemPromptEvent:
        id, tokens = 0, 1

    namespace = {
        "Enum": Enum, "SystemPromptEvent": SystemPromptEvent,
        "logger": logging.getLogger("bookvar-condenser-fixture"),
        "get_total_token_count": lambda events, llm: sum(e.tokens for e in events),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 "pinned-openhands-pure-methods", "exec"), namespace)
    condenser = namespace["LLMSummarizingCondenser"]()
    condenser.max_size, condenser.keep_first, condenser.max_tokens = 10, 2, None

    class Boundaries:
        def __init__(self, values):
            self.values = values

        def find_next(self, value):
            return next(i for i in self.values if i >= value)

    class View:
        def __init__(self, count, boundaries=None, request=False):
            self.events = [SystemPromptEvent()] + [
                SimpleNamespace(id=i, tokens=1) for i in range(1, count)
            ]
            self.manipulation_indices = Boundaries(
                range(count+1) if boundaries is None else boundaries
            )
            self.unhandled_condensation_request = request

        def __len__(self):
            return len(self.events)

        def __getitem__(self, item):
            return self.events[item]

    Reason, Requirement = namespace["Reason"], namespace["CondensationRequirement"]
    view = View(11)
    original_ids = [event.id for event in view.events]
    assert condenser.get_condensation_reasons(view) == {Reason.EVENTS}
    assert condenser.condensation_requirement(view) == Requirement.SOFT
    forgotten, offset = condenser._get_forgotten_events(view)
    assert offset == 2 and [event.id for event in forgotten] == list(range(2, 9))
    assert [event.id for event in view.events] == original_ids
    assert condenser.condensation_requirement(View(10)) is None
    assert condenser.condensation_requirement(View(5, request=True)) == Requirement.HARD

    atomic = View(11, boundaries=[0, 1, 2, 4, 6, 10, 11])
    forgotten, offset = condenser._get_forgotten_events(atomic)
    assert offset == 2 and [event.id for event in forgotten] == list(range(2, 10))
    condenser.keep_first = 0
    forgotten, offset = condenser._get_forgotten_events(view)
    assert offset == 1 and all(event.id != 0 for event in forgotten)
    condenser.keep_first, condenser.max_tokens = 2, 100
    fake_llm = SimpleNamespace(effective_max_input_tokens=8)
    assert condenser._effective_max_tokens(fake_llm) == 8
    assert Reason.TOKENS in condenser.get_condensation_reasons(view, fake_llm)
    assert condenser.condensation_requirement(view, fake_llm) == Requirement.HARD


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    assert len(blocks) == 5
    shared, scopes = {}, []
    for i, block in enumerate(blocks):
        code = compile(block, f"{CHAPTER}:example-{i+1}", "exec")
        namespace = {}
        exec(code, namespace)
        exec(code, shared)
        scopes.append(namespace)
    json_blocks = re.findall(r"^(~~~|```)json\n(.*?)^\1", text, re.M | re.S)
    assert len(json_blocks) == 1
    for _, block in json_blocks:
        assert isinstance(json.loads(block), dict)

    budget, prefix, checkpoint, reminder, runner = scopes
    assert budget["after_compaction"] == 6960
    assert prefix["common_prefix_length"]([], [1]) == 0
    assert prefix["common_prefix_length"]([1, 2], [1]) == 1
    for key in checkpoint["trusted"]:
        missing = deepcopy(checkpoint["candidate"])
        missing.pop(key)
        assert not checkpoint["valid_checkpoint"](missing), key
    slots = {"other": {"id": "keep"}}
    original = deepcopy(slots)
    state = {"revision": 3, "tested_revision": 2, "test_passed": True}
    updated = reminder["update_reminders"](slots, state)
    assert slots == original and updated["other"] == original["other"]
    assert updated["verification"]["id"] == "verify-revision-3"

    run, proposals = runner["run_episode"], runner["proposals"]
    fresh = lambda: {"timeout": 30, "revision": 0}
    env = fresh()
    denied = run(proposals, env, can_write=False)
    assert denied["status"] == "incomplete" and denied["reason"] == "permission_denied"
    assert env == fresh()
    env = fresh()
    limited = run(proposals, env, tool_limit=1)
    assert limited["reason"] == "tool_budget" and env == fresh()
    env = fresh()
    limited = run(proposals, env, model_limit=2)
    assert limited["reason"] == "model_budget" and limited["model_steps"] == 2
    assert env == {"timeout": 45, "revision": 1}
    duplicate = deepcopy(proposals)
    duplicate[1]["call_id"] = "c1"
    env = fresh()
    assert run(duplicate, env)["reason"] == "invalid_call_id"
    assert env == fresh()
    malformed = deepcopy(proposals)
    malformed[1]["arguments"]["expected_revision"] = False
    env = fresh()
    assert run(malformed, env)["status"] == "incomplete"
    assert env == fresh()

    env = fresh()
    partial = run(proposals[:4], env)
    assert partial["tested_revision"] == 1 and partial["test_passed"]
    preserved = deepcopy(partial)
    env["revision"] = 2
    stale = run([proposals[-1]], env, checkpoint=partial)
    assert stale["status"] == "incomplete"
    assert partial == preserved
    assert stale["events"][-1]["kind"] == "finish_check"
    assert stale["events"][-1]["passed"] is False
    wrong_answer = run([{"kind": "final", "timeout": 44}],
                       {"timeout": 45, "revision": 1}, checkpoint=partial)
    assert wrong_answer["status"] == "incomplete"
    assert [e["number"] for e in stale["events"]] == list(range(1, len(stale["events"])+1))
    for event in stale["events"]:
        if event["kind"] == "model_input":
            assert re.fullmatch(r"[0-9a-f]{64}", event["sha256"])

    check_pinned_condenser()
    refs = json.loads((ARCHIVE / "references.json").read_text())
    assert sha(CHAPTER) == refs["chapter_sha256"]
    for row in refs["downloaded"] + refs["reused"] + refs["derived"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    for row in refs["figures"]:
        for field, digest in [("asset", "sha256"),
                              ("source_asset", "source_asset_sha256"),
                              ("source_page_snapshot", "source_page_sha256"),
                              ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / row[field]) == row[digest]
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        crop = page.crop(tuple(row["crop_box_px"]))
        actual = Image.open(ROOT / row["asset"]).convert("RGB")
        assert crop.size == actual.size
        assert ImageChops.difference(crop, actual).getbbox() is None
        assert row["license"] == "CC BY 4.0"

    print(json.dumps({
        "chapter_sha256": sha(CHAPTER), "python_blocks": len(blocks),
        "standalone_and_shared_runs": 2*len(blocks), "json_examples_parsed": len(json_blocks),
        "token_budget_after_compaction": budget["after_compaction"],
        "protected_checkpoint_fields_checked": len(checkpoint["trusted"]),
        "fresh_revision_and_budget_failures_checked": True,
        "pinned_openhands_pure_methods_executed": 4,
        "pinned_openhands_commit": refs["openhands_commit"],
        "source_downloads_verified": len(refs["downloaded"]),
        "original_figure_crops_verified": len(refs["figures"]),
        "llm_or_sdk_integration": False, "actual_tokenizer_run": False,
        "network_or_real_mutations": False, "performance_benchmark": False,
    }, indent=2))


if __name__ == "__main__":
    main()
