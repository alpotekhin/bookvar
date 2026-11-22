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
ENGLISH = ROOT / "en/00 Textbook/16 Multimodal Models/64a Connectors and fusion.md"
SOURCE = ROOT / "05 Источники/Originals/connectors-2026-10-20"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/connectors-review-2026-10"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_blocks(path, chapter):
    blocks = re.findall(r"^\x60\x60\x60python\n(.*?)^\x60\x60\x60", chapter, re.M | re.S)
    assert len(blocks) == 6
    shared, spaces = {}, []
    for number, block in enumerate(blocks, 1):
        local = {}
        code = compile(block, f"{path}:block-{number}", "exec")
        exec(code, local)
        exec(code, shared)
        spaces.append(local)
    return spaces

def check_examples(spaces):
    qformer = next(space for space in spaces if "qformer_mask" in space)
    mask = qformer["qformer_mask"]
    matching_logits = qformer["matching_logits"]
    individual = np.exp(matching_logits - matching_logits.max(axis=1, keepdims=True))
    individual /= individual.sum(axis=1, keepdims=True)
    assert abs(individual[:, 1].mean() - .842) < .0005
    assert not np.isclose(individual[:, 1].mean(), qformer["matching_prob"][1])
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

    resampler = next(space for space in spaces if "attention" in space)
    attn = resampler["attention"]
    visual, latents = resampler["visual"], resampler["latents"]
    result = resampler["result"]
    for permutation in itertools.permutations(range(4)):
        memory = np.concatenate([visual[list(permutation)], latents])
        changed, weights = attn(latents, memory, memory)
        assert np.allclose(latents+changed, result)
        assert np.allclose(weights.sum(axis=1), 1)
    assert np.allclose(result[0], [1.65186040, .46961883], atol=1e-8)

    patch_ids = np.arange(24 * 24).reshape(24, 24)
    grouped = patch_ids.reshape(12, 2, 12, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    assert grouped.shape == (144, 4)
    assert grouped[0].tolist() == [0, 1, 24, 25]
    assert grouped[-1].tolist() == [550, 551, 574, 575]
    assert np.array_equal(np.sort(grouped.ravel()), patch_ids.ravel())
    assert 32 * 576 == 18_432 and 64 * (576 + 64) == 40_960
    return mask_cases

def main():
    chapter, english = CHAPTER.read_text(), ENGLISH.read_text()
    assert f'canonical_revision: "{sha(CHAPTER)}"' in english
    mask_cases = sum(check_examples(run_blocks(path, text)) for path, text in [
        (CHAPTER, chapter), (ENGLISH, english)
    ])

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
    size = 2 * 32 * 128 * 2
    assert [n * 32 * size / 2**20 for n in (616, 72, 40)] == [308, 36, 20]
    assert 64 * 8 * size / 2**20 == 8
    scores = np.log([.8, .1, .1])
    scores_delta = np.array([0., 1e-5, 0.])
    for positives, expected in [([.5, .5, 0.], -.3833333333), ([1., 0., 0.], .0666666667)]:
        target = .9 * np.array(positives) + .1 / 3
        def row_loss(s):
            return np.logaddexp.reduce(s) - target @ s
        derivative = (row_loss(scores + scores_delta) - row_loss(scores - scores_delta)) / 2e-5
        assert np.isclose(derivative, expected, atol=1e-9)

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
    assert repair["asset"] not in chapter
    assert "https://arxiv.org/pdf/2204.14198v2#page=4" in chapter
    pheye = json.loads((ROOT/"docs/audit/2026-11-13-flamingo-molmo-figure-additions.json").read_text())["figure"]
    assert pheye["asset"] in chapter and pheye["figure_url"] in chapter
    assert sha(ROOT/pheye["asset"]) == pheye["sha256"]
    assert sha(ROOT/pheye["license_evidence"]) == pheye["license_evidence_sha256"]
    llava = ROOT / "00 Учебник/Assets/Figures/curated/connectors-2026-11-09/llava-figure1-projector.png"
    assert sha(llava) == "310746447f5ad4ae00a2e54407bed5845795221ca7b863779edacb3b86d80f44"
    assert llava.relative_to(ROOT).as_posix() in chapter

    pinned = ROOT/"05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images"
    base = ROOT/"00 Учебник/Assets/Figures/curated"
    for original, target in [
        ("llava-architecture.png", "vlm-2026/llava-figure1-projector.png"),
        ("llava-onevision.png", "stanford-cs336-2026/multimodal/llava-onevision.png"),
        ("qwen3-vl.png", "stanford-cs336-2026/multimodal/qwen3-vl.png")
    ]:
        assert sha(pinned/original) == sha(base/target)
    def figures(text):
        return [
            item.removeprefix("02 Areas/ML & DL/")
            for item in re.findall(r"!\[\[([^\]|]+)", text)
        ]
    assert len(figures(chapter)) == 7
    assert figures(chapter) == figures(english)
    print(json.dumps({
        "chapter_sha256":[sha(CHAPTER), sha(ENGLISH)], "standalone_and_shared_code_runs":24,
        "qformer_mask_sizes":mask_cases, "resampler_permutations":48,
        "gate_gradient_samples":50, "visual_kv_mib":kv_mib,
        "multi_caption_loss_gradient_controls":2,
        "common_scan_positions":{"llava":576, "blip2":32, "flamingo_cross":64, "qwen3_vl":144},
        "pixel_identical_new_crops":3, "pixel_identical_existing_figure_repairs":1,
        "native_pheye_figure":pheye["sha256"],
        "unchanged_pinned_course_figures":3,
        "pretrained_model_run":False
    }, indent=2))

if __name__ == "__main__":
    main()
