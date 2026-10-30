#!/usr/bin/env python3
"""Connector mechanisms and figure provenance, not pretrained VLM evaluation."""
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64a Connectors и fusion.md"
SOURCE = ROOT / "05 Источники/Originals/connectors-2026-10-20"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/connectors-review-2026-10"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^\x60\x60\x60python\n(.*?)^\x60\x60\x60", chapter, re.M | re.S)
    assert len(blocks) == 6
    shared, spaces = {}, []
    for number, block in enumerate(blocks, 1):
        local = {}
        code = compile(block, f"{CHAPTER}:block-{number}", "exec")
        exec(code, local)
        exec(code, shared)
        spaces.append(local)

    mask = spaces[1]["qformer_mask"]
    matching_logits = spaces[1]["matching_logits"]
    individual = np.exp(matching_logits - matching_logits.max(axis=1, keepdims=True))
    individual /= individual.sum(axis=1, keepdims=True)
    assert abs(individual[:, 1].mean() - .842) < .0005
    assert not np.isclose(individual[:, 1].mean(), spaces[1]["matching_prob"][1])
    mask_cases = 0
    for q in range(1, 5):
        for t in range(1, 6):
            itc, itm, itg = [mask(mode, q, t) for mode in ["ITC", "ITM", "ITG"]]
            assert itc.sum() == q*q + t*t
            assert itm.sum() == (q+t)**2
            assert itg.sum() == q*q + q*t + t*(t+1)//2
            assert not itg[:q, q:].any()
            assert itg[q:, :q].all()
            assert not np.triu(itg[q:, q:], 1).any()
            mask_cases += 1

    attn = spaces[3]["attention"]
    visual, latents = spaces[3]["visual"], spaces[3]["latents"]
    result = spaces[3]["result"]
    for permutation in itertools.permutations(range(4)):
        memory = np.concatenate([visual[list(permutation)], latents])
        changed, weights = attn(latents, memory, memory)
        assert np.allclose(latents+changed, result)
        assert np.allclose(weights.sum(axis=1), 1)
    assert np.allclose(result[0], [1.65186040, .46961883], atol=1e-8)

    # Gate derivatives: a zero gate blocks branch-weight gradients, not its own.
    rng = np.random.default_rng(20261020)
    for _ in range(50):
        theta, alpha, target = rng.normal(size=3)
        def loss(t, a):
            return .5 * (np.tanh(a)*t - target)**2
        value = np.tanh(alpha)*theta
        dt = (value-target)*np.tanh(alpha)
        da = (value-target)*theta*(1-np.tanh(alpha)**2)
        eps = 1e-5
        assert np.isclose((loss(theta+eps, alpha)-loss(theta-eps, alpha))/(2*eps), dt, atol=1e-8)
        assert np.isclose((loss(theta, alpha+eps)-loss(theta, alpha-eps))/(2*eps), da, atol=1e-8)

    assert 1024*4096 + 4096 == 4_198_400
    assert 1024*4096 + 4096 + 4096*4096 + 4096 == 20_979_712
    kv_mib = [2*32*n*32*128*2 / 2**20 for n in [576,64,32]]
    assert kv_mib == [288,32,16]
    assert 576+40 == 616 and 616+60 == 676

    records = json.loads((FIGURES/"assets.json").read_text())
    assert len(records) == 3
    for row in records:
        for path_key, hash_key in [
            ("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
            ("license_evidence", "license_evidence_sha256")
        ]:
            assert sha(ROOT/row[path_key]) == row[hash_key]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        crop = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), crop).getbbox() is None
        assert row["asset"] in chapter
    for row in json.loads((SOURCE/"references.json").read_text())["downloaded"]:
        assert sha(ROOT/row["path"]) == row["sha256"]

    repair = json.loads((SOURCE/"flamingo-repair.json").read_text())
    for path_key, hash_key in [("asset", "sha256"), ("source_raster", "source_raster_sha256"), ("reviewed_paper", "reviewed_paper_sha256")]:
        assert sha(ROOT/repair[path_key]) == repair[hash_key]
    before = Image.open(ROOT/repair["source_raster"]).convert("RGB")
    after = Image.open(ROOT/repair["asset"]).convert("RGB")
    assert list(before.size) == repair["source_raster_size_px"]
    assert ImageChops.difference(before.crop(repair["crop_box_px"]), after).getbbox() is None

    pinned = ROOT/"05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images"
    base = ROOT/"00 Учебник/Assets/Figures/curated"
    for original, target in [
        ("llava-architecture.png", "vlm-2026/llava-figure1-projector.png"),
        ("llava-onevision.png", "stanford-cs336-2026/multimodal/llava-onevision.png"),
        ("qwen3-vl.png", "stanford-cs336-2026/multimodal/qwen3-vl.png")
    ]:
        assert sha(pinned/original) == sha(base/target)
    assert len(re.findall(r"!\[\[", chapter)) == 7
    print(json.dumps({
        "chapter_sha256":sha(CHAPTER), "standalone_and_shared_code_runs":12,
        "qformer_mask_sizes":mask_cases, "resampler_permutations":24,
        "gate_gradient_samples":50, "visual_kv_mib":kv_mib,
        "pixel_identical_new_crops":3, "pixel_identical_existing_figure_repairs":1,
        "unchanged_pinned_course_figures":3,
        "pretrained_model_run":False
    }, indent=2))

if __name__ == "__main__":
    main()
