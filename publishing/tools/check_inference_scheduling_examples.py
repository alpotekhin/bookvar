#!/usr/bin/env python3
"""Execute 55b's actual Markdown examples and check their CPU-only models.

No inference engine, GPU, wall-clock benchmark or cryptographic proof is claimed.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import random
import re
import struct

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / (
    "00 Учебник/14 Inference и оптимизация/"
    "55b Scheduling — continuous batching, chunked prefill и prefix caching.md"
)
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/inference-scheduling-review-2026-10"
ORIGINALS = ROOT / "05 Источники/Originals/scheduling-2026-10-19"
OLD_HEADINGS = [
    "Почему в статическом пакете простаивают места",
    "Orca: планирование на уровне итерации",
    "Что выбирает планировщик на каждом шаге",
    "Fairness, preemption и перегрузка",
    "Почему длинный prefill мешает decode",
    "Prefix caching: не вычислять общий вход повторно",
    "Как механизмы взаимодействуют",
    "Устойчивый механизм и меняющийся API",
    "Практика и первоисточники",
]


def expect_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError(f"{function.__name__}{args!r} must reject input")


def reference_completions(values, slots, continuous):
    """Independent event-time model rather than the chapter's iteration loop."""
    if continuous:
        available, result = [0] * slots, {}
        for i, length in enumerate(values):
            result[str(i)] = heapq.heappop(available) + length
            heapq.heappush(available, result[str(i)])
        return result
    start, result = 0, {}
    for offset in range(0, len(values), slots):
        group = values[offset:offset + slots]
        for index, length in enumerate(group, offset):
            result[str(index)] = start + length
        start += max(group)
    return result


def check_batching(ns):
    f, count = ns["finish_steps"], 0
    for size in range(1, 7):
        for values in itertools.product((1, 2, 3), repeat=size):
            jobs = {str(i): n for i, n in enumerate(values)}
            for slots in range(1, 5):
                for continuous in (False, True):
                    assert f(jobs, slots, continuous) == reference_completions(
                        values, slots, continuous
                    )
                    assert jobs == {str(i): n for i, n in enumerate(values)}
                    count += 1
    assert f({}, 4, True) == {}
    expect_value_error(f, {"A": 0}, 1, True)
    expect_value_error(f, {"A": 1}, 0, True)
    assert (2 + 4 + 6 + 8) / (4 * 8) == 0.625
    print(f"PASS batching: {count} traces vs independent event-time reference")


def check_admission(ns):
    f, rng = ns["plan_step"], random.Random(20261014)
    for _ in range(3000):
        block = rng.randrange(1, 17)
        budget, free = rng.randrange(0, 40), rng.randrange(0, 9)
        requests = [
            (str(i), rng.randrange(0, 65), rng.randrange(0, 25))
            for i in range(rng.randrange(0, 9))
        ]
        expected, remaining, blocks = {}, budget, free
        for name, computed, wanted in requests:
            old_blocks = (computed + block - 1) // block
            feasible = [
                q for q in range(min(wanted, remaining) + 1)
                if (computed + q + block - 1) // block - old_blocks <= blocks
            ]
            q = max(feasible)
            expected[name] = q
            remaining -= q
            blocks -= (computed + q + block - 1) // block - old_blocks
        assert f(requests, budget, free, block) == (expected, remaining, blocks)
        assert sum(expected.values()) <= budget and min(remaining, blocks) >= 0
    for bad in [([], -1, 0, 4), ([], 1, -1, 4), ([], 1, 0, 0),
                ([("A", -1, 1)], 1, 1, 4), ([("A", 0, -1)], 1, 1, 4),
                ([("A", 0, 0), ("A", 1, 1)], 4, 4, 4)]:
        expect_value_error(f, *bad)
    print("PASS admission: 3000 plans vs exhaustive per-request feasible counts")


def check_chunks(ns):
    f, count = ns["prefill_timeline"], 0
    for prompt in range(1, 65):
        for chunk in range(1, 17):
            for decodes in (0, 1, 4, 9):
                events = f(prompt, chunk, decodes)
                passes = (prompt + chunk - 1) // chunk
                assert len(events) == passes
                assert sum(q for q, _ in events) == prompt
                assert all(0 < q <= chunk for q, _ in events)
                assert events[-1][1] == prompt + passes * (2 + 0.5 * decodes)
                previous = 0
                for q, elapsed in events:
                    assert elapsed - previous == 2 + q + 0.5 * decodes
                    previous = elapsed
                count += 1
    assert 4 * 16 + 4 * 5 // 2 == 74
    assert 4 * 16000 + 4 * 5 // 2 == 64010
    for bad in [(0, 4, 1), (4, 0, 1), (4, 4, -1)]:
        expect_value_error(f, *bad)
    print(f"PASS chunking: {count} traces; conditional model, not measured ms")


def check_prefix(ns):
    f = ns["prefix_keys"]
    namespace = {"model": "r1", "adapter": None, "tenant": "A"}
    tokens = list(range(19))
    keys = f(tokens, 4, namespace)
    assert len(keys) == 4
    assert f(tokens[:-3], 4, namespace) == keys
    assert f(tokens[:16] + [99, 98, 97], 4, namespace) == keys
    for field in namespace:
        assert f(tokens, 4, {**namespace, field: "different"})[0] != keys[0]
    assert f(tokens, 4, dict(reversed(list(namespace.items())))) == keys
    changed = tokens.copy()
    changed[5] = 999
    different = f(changed, 4, namespace)
    assert keys[0] == different[0]
    assert all(x != y for x, y in zip(keys[1:], different[1:]))
    assert f([], 4, namespace) == [] and f([1, 2, 3], 4, namespace) == []
    expect_value_error(f, tokens, 0, namespace)
    full_hit, count = ns["local_full_block_hit"], 0
    for prompt in range(1, 65):
        for cached in range(0, 65):
            for block in (1, 2, 4, 8, 16):
                reuse, work = full_hit(prompt, cached, block)
                possible = [
                    n for n in range(0, min(cached, prompt - 1) + 1)
                    if n % block == 0
                ]
                assert reuse == max(possible)
                assert work >= 1 and reuse + work == prompt
                count += 1
    for bad in [(0, 4, 4), (4, -1, 4), (4, 4, 0)]:
        expect_value_error(full_hit, *bad)

    # Independent prose check: one retained-prefix slot; no decode simulation.
    def prefix_work(order):
        cached, work = None, 0
        for family in order:
            if family != cached:
                work += 8
                cached = family
        return work

    assert prefix_work(["A", "B", "A"]) == 24
    assert prefix_work(["A", "A", "B"]) == 16
    print(f"PASS prefix: namespace/parent/tail checks and {count} block-hit cases")
    print("NOTE hash tests establish example behavior, not collision resistance")


def check_packed_masks(ns):
    function = ns["packed_causal_mask"]
    count = 0
    for size in range(1, 4):
        for requests in itertools.product(itertools.product(range(4), range(1, 4)), repeat=size):
            keys = [(owner, position) for owner, (cached, queries) in enumerate(requests)
                    for position in range(cached + queries)]
            rows = [(owner, position) for owner, (cached, queries) in enumerate(requests)
                    for position in range(cached, cached + queries)]
            expected = [[owner == key_owner and key_position <= position
                         for key_owner, key_position in keys] for owner, position in rows]
            assert function(requests) == expected
            count += 1
    for requests in ([], [(-1, 1)], [(0, 0)], [(0, -1)]):
        expect_value_error(function, requests)

    dynamic = function([(8, 1), (0, 6)])
    assert (len(dynamic), len(dynamic[0]), sum(map(sum, dynamic))) == (7, 15, 30)
    second = function([(5, 1), (0, 4)])
    third = function([(4, 3), (0, 2)])
    assert sum(map(sum, second)) == 16
    assert sum(map(sum, third)) == 21

    def output(mask, values):
        result = []
        for row, allowed in enumerate(mask):
            weights = [math.exp(0.01 * (row + 1) * (key + 1)) if valid else 0.0
                       for key, valid in enumerate(allowed)]
            result.append(sum(w * v for w, v in zip(weights, values)) / sum(weights))
        return result

    values = list(range(1, 11))
    changed = [value + 1000 if key < 6 else value for key, value in enumerate(values)]
    baseline, isolated = output(second, values), output(second, changed)
    assert math.isclose(isolated[0] - baseline[0], 1000)
    assert isolated[1:] == baseline[1:]
    # A single lower-right causal diagonal preserves row 0 but leaks request 0
    # into the four query rows of request 1.
    wrong = [[key <= row + 5 for key in range(10)] for row in range(5)]
    assert wrong[0] == second[0] and wrong != second
    assert all(abs(a - b) > 100 for a, b in
               zip(output(wrong, values)[1:], output(wrong, changed)[1:]))
    square = [[key <= row for key in range(5)] for row in range(5)]
    assert len(square[0]) != len(second[0])
    print(f"PASS packed masks: {count} layouts vs owner/position reference; 7x15, 5x10, 5x9")
    print("PASS scalar attention isolation; wrong global diagonal leaks across requests")


def image_dimensions(binary):
    if binary[:8] == b"\x89PNG\r\n\x1a\n":
        return list(struct.unpack(">II", binary[16:24]))
    assert binary[:2] == b"\xff\xd8" and binary[-2:] == b"\xff\xd9"
    offset = 2
    while offset < len(binary):
        assert binary[offset] == 0xFF
        while binary[offset] == 0xFF:
            offset += 1
        marker = binary[offset]
        offset += 1
        size = int.from_bytes(binary[offset:offset + 2], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", binary[offset + 3:offset + 7])
            return [width, height]
        assert size >= 2
        offset += size
    raise AssertionError("missing JPEG frame header")


def check_structure_and_assets(text):
    assert '<a id="почему-статический-пакет-теряет-вместимость"></a>' in text
    for heading in OLD_HEADINGS:
        if heading == "Устойчивый механизм и меняющийся API":
            assert '<a id="устойчивый-механизм-и-меняющийся-api"></a>' in text
            assert "\n## Версии движков и проверка производительности\n" in text
        else:
            assert f"\n## {heading}\n" in text, f"lost anchor: {heading}"
    for source_id in (
        "lecture-10-section-381-handling-dynamic-workloads",
        "lecture-10-dynamic-workload-failure-mode",
        "lecture-10-figure-step-392-rendering-1",
    ):
        assert f"  - {source_id}\n" in text
    assert "```mermaid" not in text
    embeds = re.findall(r"!\[\[(.*?)\]\]", text)
    assert len(embeds) == 7, embeds
    for embed in embeds:
        target = embed.split("|")[0].removeprefix("02 Areas/ML & DL/")
        assert (ROOT / target).is_file(), target
    records = json.loads((ASSETS / "assets.json").read_text())["new_assets"]
    assert len(records) == 3
    for entry in records:
        binary = (ROOT / entry["asset"]).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == entry["sha256"]
        assert image_dimensions(binary) == entry["dimensions"]
        assert re.fullmatch(r"curated-[a-z0-9-]+", entry["id"])
        assert "undefined" not in entry["review_note"]
        assert entry["commit"] in entry["source_url"]
        assert entry["provenance_confirmation"] == "pinned-upstream-repository"
        assert entry["derivation"] == "source-raster-copy"
        license_data = (ROOT / entry["license_copy"]).read_bytes()
        assert hashlib.sha256(license_data).hexdigest() == entry["license_sha256"]
    print(f"PASS IDs, old H2 anchors, {len(embeds)} embeds, {len(records)} asset hashes")
    source_root = ROOT / "05 Источники/Originals/scheduling-2026-10-25"
    output_root = ROOT / "00 Учебник/Assets/Figures/curated/scheduling-2026-10-25"
    for name, digest, size in (
        ("dynamic", "1721080f5c46eed465bcf3d6e19f276037c7fea144acd3ec7b386a76f60ee641", [2801, 1030]),
        ("continuous", "c57b1deeacea8c2e591687d1eb105722fd352492d86cdfd74c05f166fc96068d", [2722, 2569]),
    ):
        original = (source_root / f"{name}_batching.png").read_bytes()
        actual = (output_root / f"hf-{name}-batching.png").read_bytes()
        assert original == actual
        assert hashlib.sha256(actual).hexdigest() == digest
        assert image_dimensions(actual) == size
    card = (source_root / "hf-images-README.md").read_bytes()
    assert hashlib.sha256(card).hexdigest() == "5e6a73b7ff47813e90e2ed21d14807256b2e89fa55e5075ec69fdf01cde801ce"
    assert b"license: cc-by-nc-sa-4.0" in card
    print("PASS two unmodified HF figures, pinned dataset license evidence and dimensions")


def check_source_excerpts(text):
    manifest = json.loads((ORIGINALS / "excerpts.json").read_text())
    assert ROOT / manifest["chapter"] == CHAPTER
    sources = {}
    for item in manifest["sources"]:
        binary = (ORIGINALS / item["file"]).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == item["sha256"], item["file"]
        assert manifest["upstream_commit"] in item["url"]
        sources[item["file"]] = binary.decode("utf-8")
    license_bytes = (ROOT / manifest["license_copy"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    assert len(manifest["excerpts"]) == 2
    for item in manifest["excerpts"]:
        start = f"<!-- source-excerpt:{item['id']}:start -->"
        end = f"<!-- source-excerpt:{item['id']}:end -->"
        assert text.count(start) == text.count(end) == 1, item["id"]
        actual = text.split(start)[1].split(end)[0].strip()
        actual = re.sub(r"^```[^\n]*\n|\n```$", "", actual)
        expected = item["raw"].split()
        source_words = " ".join(sources[item["source"]].split())
        assert " ".join(expected) in source_words, f"source changed: {item['id']}"
        assert actual.split() == expected, f"chapter excerpt changed: {item['id']}"
    print("PASS 2 original source excerpts, pinned source hashes and license")


def main():
    text = CHAPTER.read_text(encoding="utf-8")
    examples = re.findall(r"^```python\n(.*?)^```$", text, flags=re.M | re.S)
    assert len(examples) == 6, f"expected six examples, got {len(examples)}"
    ns = {"__name__": "bookvar_scheduling_examples"}
    for i, source in enumerate(examples, 1):
        exec(compile(source, f"{CHAPTER}:example-{i}", "exec"), ns)
    print("PASS executed all 6 actual Markdown Python blocks")
    check_batching(ns)
    check_admission(ns)
    check_chunks(ns)
    check_prefix(ns)
    check_packed_masks(ns)
    check_structure_and_assets(text)
    check_source_excerpts(text)
    print(f"CHAPTER_SHA256 {hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}")
    print("PASS inference-scheduling CPU checks; no vLLM/SGLang/CUDA execution claimed")


if __name__ == "__main__":
    main()
