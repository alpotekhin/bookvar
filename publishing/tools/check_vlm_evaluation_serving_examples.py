#!/usr/bin/env python3
"""Chapter 64f worked examples and reviewed source objects; no model or serving benchmark."""
import ast
from collections import OrderedDict
import hashlib
import itertools
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md"
ENGLISH = ROOT / "en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving.md"
SOURCE = ROOT / "05 Источники/Originals/vlm-evaluation-serving-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_examples(path, chapter):
    blocks = re.findall(r"^(`{3}|~~~)python\n(.*?)^\1$", chapter, re.M | re.S)
    assert len(blocks) == 18, len(blocks)
    spaces, shared = [], {}
    for i, (_, block) in enumerate(blocks, 1):
        code = compile(block, f"{path}:block-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)
    requests = re.findall(r"^(`{3}|~~~)json\n(.*?)^\1$", chapter, re.M | re.S)
    assert len(requests) == 1
    for _, block in requests:
        request = json.loads(block)
        assert request["max_tokens"] == 200
        items = request["messages"][0]["content"]
        assert [item["type"] for item in items] == ["image_url", "image_url", "text"]
        assert all(isinstance(item["image_url"], dict) and "url" in item["image_url"]
                   for item in items[:2])
    return spaces, [ast.dump(ast.parse(block)) for _, block in blocks]


def main():
    chapter = CHAPTER.read_text()
    english = ENGLISH.read_text()
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    spaces, ru_ast = run_examples(CHAPTER, chapter)
    english_spaces, en_ast = run_examples(ENGLISH, english)
    assert en_ast == ru_ast

    for blocks in (spaces, english_spaces):
        ttft, critical = blocks[9], blocks[10]
        arrivals = [ttft["client_ttft"] + 40 * i for i in range(10)]
        assert arrivals[0] == 500 and arrivals[-1] == 860
        assert arrivals[0] <= 550 and arrivals[-1] > 800
        assert (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) == 40
        assert critical["critical_path"] == 475 and critical["all_work"] == 710

    wg = spaces[1]["winoground_scores"]
    outcomes = [wg(np.array(p).reshape(2, 2)) for p in itertools.permutations(range(4))]
    assert np.sum(outcomes, axis=0).tolist() == [6, 6, 4]
    assert wg(np.ones((2, 2))) == (False, False, False)

    # Run only the reviewed answer-parsing loop, not the source module or its file I/O.
    pope = SOURCE / "pope-evaluate.py"
    tree = ast.parse(pope.read_text())
    parser_loop = next(node for node in tree.body
                       if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                       and node.target.id == "answer")
    answers = [{"answer": text} for text in
               ["Yes.", "No.", "I am not sure.", "I am unsure.", "NO."]]
    env = {"answers": answers}
    exec(compile(ast.Module(body=[parser_loop], type_ignores=[]), str(pope), "exec"), env)
    assert [x["answer"] for x in answers] == ["yes", "no", "no", "yes", "yes"]

    # Selected bookkeeping methods only. No model runner, tensors, weights or GPU calls.
    cache_file = SOURCE / "vllm-encoder-cache.py"
    cache_tree = ast.parse(cache_file.read_text())
    cache_class = next(n for n in cache_tree.body
                       if isinstance(n, ast.ClassDef) and n.name == "EncoderCacheManager")
    names = {"__init__", "reset", "check_and_update_cache", "get_cached_num_encoder_embeds"}
    methods = [n for n in cache_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 4
    env = {"OrderedDict": OrderedDict, "Request": SimpleNamespace}
    for method in methods:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(cache_file), "exec"), env)
    ToyCache = type("ToyCache", (), {n: env[n] for n in names})
    cache = ToyCache(100)
    req = SimpleNamespace(request_id="r1", mm_features=[SimpleNamespace(identifier="image-A")])
    assert cache.check_and_update_cache(req, 0) is False
    assert cache.get_cached_num_encoder_embeds(req, 0) is None
    cache.cached["image-A"] = set()
    cache.num_embeds_by_hash["image-A"] = 10
    cache.freeable["image-A"] = 10
    assert cache.check_and_update_cache(req, 0) is True
    assert cache.cached["image-A"] == {"r1"} and cache.num_freeable_slots == 90
    assert cache.get_cached_num_encoder_embeds(req, 0) == 10
    assert cache.check_and_update_cache(req, 0) is True and cache.num_freeable_slots == 90
    cache.reset()
    assert cache.cached == {} and cache.num_free_slots == cache.num_freeable_slots == 100
    assert cache.check_and_update_cache(req, 0) is False

    # Exact counts, not language-model measurements.
    rng = np.random.default_rng(20261020)
    for _ in range(100):
        rows = [("a", str(int(x))) for x in rng.integers(0, 100000, 8)]
        order = rng.permutation(len(rows))
        total = spaces[4]["total"]
        assert total(rows) == total([rows[i] for i in order])
        q, t = rng.random((2, 100)) > .5
        both = (q & t).sum()
        assert max(0, q.sum()+t.sum()-len(q)) <= both <= min(q.sum(), t.sum())

    refs = json.loads((SOURCE / "references.json").read_text())
    for row in refs["downloaded"] + refs["reused"]:
        assert sha(ROOT/row["path"]) == row["sha256"], row["path"]
    for row in refs["figures"] + refs["repairs"]:
        for pk, hk in [("asset", "sha256"), ("source_asset", "source_asset_sha256"),
                       ("source_page_snapshot", "source_page_sha256")]:
            assert sha(ROOT/row[pk]) == row[hk], row[pk]
        if "source_raster" in row:
            assert sha(ROOT/row["source_raster"]) == row["source_raster_sha256"]
            assert list(Image.open(ROOT/row["source_raster"]).size) == row["source_raster_size_px"]
        if "license_evidence" in row:
            assert sha(ROOT/row["license_evidence"]) == row["license_evidence_sha256"]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        asset = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert list(asset.size) == [row["width"], row["height"]]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), asset).getbbox() is None
    assert len(re.findall(r"!\[\[", chapter)) == 7
    review = json.loads((ROOT / "docs/audit/2026-11-13-vlm-evaluation-contamination-additions.json").read_text())
    figure = review["figure"]
    assert sha(ROOT / figure["asset"]) == figure["sha256"]
    assert list(Image.open(ROOT / figure["asset"]).size) == figure["dimensions_px"]
    for path in figure["used_in"]:
        text = (ROOT / path).read_text()
        assert figure["asset"] in text and figure["figure_url"] in text
        assert "https://arxiv.org/pdf/2409.17146v2#page=19" in text
        assert "molmo-figure11-openness.png" not in text
    assert "&lt;AND&gt;" in chapter and "<AND>" not in chapter
    print(json.dumps({
        "chapter_sha256": [sha(CHAPTER), sha(ENGLISH)],
        "standalone_and_shared_code_runs": 2 * (len(spaces) + len(english_spaces)),
        "first_and_last_token_ms": [500, 860], "answer_deadline_ms": 800,
        "winoground_rankings": 24, "winoground_success_counts": [6, 6, 4],
        "original_pope_parser_cases": 5, "original_encoder_cache_methods": 4,
        "permutation_and_intersection_cases": 100,
        "new_original_figures": 5, "repaired_existing_figures": 2,
        "native_contamination_figure": figure["sha256"],
        "pretrained_model_run": False, "network_or_gpu_benchmark": False,
    }, indent=2))


if __name__ == "__main__":
    main()
