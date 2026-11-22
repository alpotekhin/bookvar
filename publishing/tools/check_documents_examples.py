#!/usr/bin/env python3
"""Chapter 64d examples and source objects, not an OCR or VLM benchmark."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md"
ENGLISH = ROOT / "en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding.md"
SOURCE = ROOT / "05 Источники/Originals/documents-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_blocks(path, fence):
    blocks = re.findall(rf"^{fence}python\n(.*?)^{fence}", path.read_text(), re.M | re.S)
    assert len(blocks) == 12, len(blocks)
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        code = compile(block, f"{path}:block-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)
    return spaces


def main():
    chapter = CHAPTER.read_text()
    english = ENGLISH.read_text()
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    spaces = run_blocks(CHAPTER, "~~~")
    english_spaces = run_blocks(ENGLISH, "`" * 3)
    row_cases = [next(s for s in edition if "items" in s)
                 for edition in [spaces, english_spaces]]
    assert row_cases[0]["items"] == row_cases[1]["items"]
    assert row_cases[0]["answer_box"] == row_cases[1]["answer_box"]

    # Execute only the two reviewed serialization methods, not the downloaded module.
    model = SOURCE / "donut-model.py"
    tree = ast.parse(model.read_text())
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)
               and node.name in {"json2token", "token2json"}]
    assert len(methods) == 2
    env = dict(Any=Any, re=re)
    for node in methods:
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(model), "exec"), env)
    Toy = type("Toy", (), {name: env[name] for name in ["json2token", "token2json"]})
    toy = Toy()
    toy.decoder = SimpleNamespace(tokenizer=SimpleNamespace(
        all_special_tokens=[], get_added_vocab=lambda: {}
    ))
    samples = [
        {"total": "12480", "currency": "RUB"},
        {"items": [{"name": "Кофе", "quantity": "2"}, {"name": "Чай", "quantity": "1"}]},
        {"total": "0"},
    ]
    for obj in samples:
        serialized = toy.json2token(obj, False, False)
        assert toy.token2json(serialized) == obj
    assert toy.json2token(samples[0], False, False) == (
        "<s_total>12480</s_total><s_currency>RUB</s_currency>"
    )
    assert "total" not in toy.token2json("<s_total>12480")
    # Original code stringifies numeric leaf values: no claim of preserving JSON number types.
    assert toy.token2json(toy.json2token({"total": 12480}, False, False)) == {"total": "12480"}

    # Reading-order grouping is invariant to the order OCR returned the four items.
    for permutation in itertools.permutations(row_cases[0]["items"]):
        groups = {}
        for ident, text, box in permutation:
            cy = (box[1] + box[3]) / 2
            row = 0 if cy < 1120 else 1
            groups.setdefault(row, []).append((box[0], ident))
        rows = [[item[1] for item in sorted(groups[k])] for k in sorted(groups)]
        assert rows == [["l_subtotal", "v_subtotal"], ["l_total", "v_total"]]

    rng = np.random.default_rng(20261020)
    for _ in range(100):
        s = rng.uniform(.1, 3)
        offset = rng.uniform(-1000, 1000, 2)
        matrix = np.array([[s, 0, offset[0]], [0, s, offset[1]], [0, 0, 1.]])
        point = np.r_[rng.uniform(0, 2000, 2), 1.]
        assert np.allclose(np.linalg.inv(matrix) @ (matrix @ point), point)
    assert (100 * 2, (700 - 100) * 2) == (200, 1200)

    # Binary relations in the completed panel of Figure 3, not model predictions.
    precedes = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0],
    ])
    assert np.array_equal(precedes + precedes.T, 1 - np.eye(5))
    predecessors = precedes.sum(axis=0)
    reading_order = predecessors.argsort() + 1
    assert predecessors.tolist() == [3, 4, 0, 2, 1]
    assert reading_order.tolist() == [3, 5, 4, 1, 2]
    invoice_regions = {1: "total", 2: "footnote", 3: "heading", 4: "tax", 5: "subtotal"}
    assert [invoice_regions[i] for i in reading_order] == [
        "heading", "subtotal", "tax", "total", "footnote",
    ]
    cyc = np.array([[0, .9, .4], [.1, 0, .8], [.6, .2, 0]])
    assert np.allclose(cyc.sum(axis=0), [.7, 1.1, 1.2])
    assert cyc.sum(axis=0).argsort().tolist() == [0, 1, 2]
    assert cyc[0, 1] > .5 and cyc[1, 2] > .5 and cyc[2, 0] > .5

    # A matching literal string can come from the wrong field.
    alternatives = {"return": ("14 days", "right"),
                    "processing": ("14 days", "other")}
    correct = alternatives["return"]
    assert alternatives["processing"][0] == correct[0]
    assert alternatives["processing"] != correct
    changed = {**alternatives, "return": ("7 days", "right")}
    moved = {"return": ("14 days", "other"),
             "processing": ("14 days", "right")}
    assert changed["return"][0] != correct[0] and changed["return"][1] == correct[1]
    assert moved["return"][0] == correct[0] and moved["return"][1] != correct[1]

    distance, anls, iou = [
        next(space[key] for space in spaces if key in space)
        for key in ["edit_distance", "anls_one", "box_iou"]
    ]
    strings = ["", "0", "О", "ab", "ac", "12 480", "12480"]
    for a, b in itertools.product(strings, repeat=2):
        assert distance(a, b) == distance(b, a)
        assert distance(a, a) == 0
        assert 0 <= anls(a, b) <= 1
    assert anls("ab", "ac") == 0
    assert anls("12480", "1248О") == .8
    assert distance("12480", "12840") == 2 and 12840 - (11000 + 1480) == 360
    for _ in range(100):
        lo = rng.uniform(0, 100, 2)
        hi = lo + rng.uniform(.01, 100, 2)
        box = np.r_[lo, hi]
        other = box + np.tile(rng.uniform(-20, 20, 2), 2)
        assert np.isclose(iou(box, box), 1)
        assert np.isclose(iou(box, other), iou(other, box))
        assert 0 <= iou(box, other) <= 1

    refs = json.loads((SOURCE/"references.json").read_text())
    for row in refs["downloaded"] + refs["reused"]:
        assert sha(ROOT/row["path"]) == row["sha256"]
    for row in refs["figures"] + refs["repairs"]:
        if "source_raster" in row:
            assert sha(ROOT/row["source_raster"]) == row["source_raster_sha256"]
        for path_key, hash_key in [
            ("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
        ]:
            assert sha(ROOT/row[path_key]) == row[hash_key]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        asset = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), asset).getbbox() is None
    figure_root = ROOT/"00 Учебник/Assets/Figures/curated/documents-review-2026-10"
    assert sha(figure_root/"donut-overview.png") == sha(SOURCE/"donut-overview.png")
    assert sha(ROOT/"site/public/licenses/donut-4cfcf972-MIT.txt") == sha(SOURCE/"donut-LICENSE.txt")
    assert sha(ROOT/"site/public/licenses/donut-4cfcf972-NOTICE.txt") == sha(SOURCE/"donut-NOTICE.txt")
    reader_figures = {
        link.split("|", 1)[0].rsplit("/", 1)[-1]
        for link in re.findall(r"!\[\[([^\]\n]+)\]\]", chapter)
    }
    assert reader_figures == {
        "layoutlmv3-objectives.png", "donut-overview.png", "pix2struct-screenshot-parsing.png",
        "pix2struct-variable-resolution.png",
        "grits-figure1-alignment.png", "grits-figure2-representations.png",
        "pp-doclayout-v2-reading-order.png",
    }
    assert "https://proceedings.mlr.press/v202/lee23g/lee23g.pdf#page=2" in chapter
    assert "https://arxiv.org/pdf/2409.12191v1#page=29" in chapter
    for block in re.findall(r"^~~~json\n(.*?)^~~~", chapter, re.M | re.S):
        json.loads(block)
    print(json.dumps({
        "chapter_sha256": [sha(CHAPTER), sha(ENGLISH)], "standalone_and_shared_code_runs": 48,
        "pinned_donut_serialization_cases": 5, "reading_order_permutations": 24,
        "paddleocr_figure3_reading_order": reading_order.tolist(),
        "pairwise_cycle_column_sums": cyc.sum(axis=0).round(2).tolist(),
        "inverse_transform_cases": 100, "string_pairs": len(strings)**2,
        "iou_cases": 100, "new_original_figures": 4,
        "repaired_historical_figures": 1, "unchanged_historical_figures": 1,
        "pretrained_model_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
