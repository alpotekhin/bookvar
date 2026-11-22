#!/usr/bin/env python3
"""Vision encoder mechanisms and original-source identity; not a trained-model benchmark."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md"
EN_CHAPTER = ROOT / "en/00 Textbook/16 Multimodal Models/64 Multimodal models.md"
SOURCE = ROOT / "05 Источники/Originals/vision-encoders-2026-10-20"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/vision-encoders-review-2026-10"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_blocks(path):
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)
    assert len(blocks) == 6
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        local = {}
        code = compile(block, f"{path}:block-{i}", "exec")
        exec(code, local)
        exec(code, shared)
        spaces.append(local)
    return spaces

def main():
    chapter = CHAPTER.read_text()
    english = EN_CHAPTER.read_text()
    assert f'canonical_revision: "{digest(CHAPTER)}"' in english
    spaces, en_spaces = run_blocks(CHAPTER), run_blocks(EN_CHAPTER)
    for i, names in enumerate([("patches", "projected"), ("original", "swapped"),
        ("loss",), ("before", "after"), ("chunked",), ("scores",)]):
        for name in names:
            if name in spaces[i]:
                np.testing.assert_array_equal(spaces[i][name], en_spaces[i][name])

    assert spaces[0]["patches"].shape == (4, 12)
    assert spaces[0]["projected"].shape == (4, 2)
    assert np.allclose(spaces[0]["projected"], spaces[0]["conv"])
    attend, z = spaces[1]["attend"], spaces[1]["z"]
    for p in itertools.permutations(range(1, 5)):
        perm = [0, *p]
        assert np.allclose(attend(z[perm]), attend(z)[perm])
    rng = np.random.default_rng(20261020)
    clip, sigmoid = spaces[2]["clip_loss"], spaces[4]["siglip_loss"]
    for _ in range(40):
        scores = rng.normal(size=(3, 3)) * 3
        a = np.exp(scores - scores.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        b = np.exp(scores - scores.max(axis=0, keepdims=True))
        b /= b.sum(axis=0, keepdims=True)
        gradient = (a + b - 2*np.eye(3)) / 6
        labels = 2*np.eye(3) - 1
        binary_gradient = ((1/(1+np.exp(-scores))) - np.eye(3)) / 3
        for i, j in [(0, 0), (0, 1), (2, 1)]:
            delta = np.zeros((3, 3))
            delta[i, j] = 1e-5
            assert np.isclose((clip(scores+delta)-clip(scores-delta))/2e-5, gradient[i,j], atol=1e-9)
            assert np.isclose((sigmoid(scores+delta)-sigmoid(scores-delta))/2e-5, binary_gradient[i,j], atol=1e-9)
        assert np.isclose(clip(scores+10000), clip(scores))
        assert np.isclose(sigmoid(scores), np.logaddexp(0, -labels*scores).mean()*3)
    for B in [2, 3, 8, 64]:
        bias = -np.log(B-1)
        assert np.isclose(1/(1+np.exp(-bias)), 1/B)
        def neutral_loss(b):
            return sigmoid(np.full((B, B), b))
        assert neutral_loss(bias) <= neutral_loss(bias+.1)
        assert neutral_loss(bias) <= neutral_loss(bias-.1)
    prob = spaces[3]["class_probabilities"]
    image, labels = spaces[3]["image"], spaces[3]["labels"]
    assert prob(image, labels, 1).argmax() == prob(image, labels, 100).argmax()
    assert np.isclose(prob(image, labels, 5).sum(), 1)
    paired = spaces[5]["paired_scores"]
    for flat in itertools.product([0., 1., 2.], repeat=4):
        m = np.array(flat).reshape(2,2)
        text, visual, group = paired(m)
        assert group == (text and visual)
        assert paired(m.T) == (visual, text, group)
    # 640x480 -> 448x336, then center crop: 80 original pixels per side.
    assert (448 - 336) / 2 / (336/480) == 80
    source = np.full((480, 640, 3), 255, dtype=np.uint8)
    outside, inside = source.copy(), source.copy()
    outside[400:420, 590:610] = 0
    inside[400:420, 400:420] = 0
    resize = lambda x: Image.fromarray(x).resize((448, 336), Image.Resampling.BICUBIC)
    crop = lambda im: np.asarray(im.crop((56, 0, 392, 336)))
    reference = resize(source)
    assert not np.array_equal(np.asarray(reference), np.asarray(resize(outside)))
    assert np.array_equal(crop(reference), crop(resize(outside)))
    assert not np.array_equal(crop(reference), crop(resize(inside)))
    assert 576**2 == 331776 and 2304**2 == 5308416 == 16 * 576**2
    assert np.isclose(np.exp(1) / (2 * np.exp(1)), .5)
    assert np.exp(1) / (2 * np.exp(1) + np.exp(-5)) < .5

    records = json.loads((FIGURES / "assets.json").read_text())
    assert len(records) == 3
    for row in records:
        for key, h in [("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"), ("license_evidence", "license_evidence_sha256")]:
            assert digest(ROOT / row[key]) == row[h]
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        crop = Image.open(ROOT / row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), crop).getbbox() is None
        assert row["asset"] in chapter
    for row in json.loads((SOURCE / "references.json").read_text())["downloaded"]:
        assert digest(ROOT / row["path"]) == row["sha256"]
    curated = ROOT / "00 Учебник/Assets/Figures/curated"
    originals = ROOT / "05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images"
    vit = curated / "vlm-2026/vit-figure1-patch-sequence.png"
    assert digest(originals / "vit.png") == digest(vit)
    for filename, pdf, page_no, rectangle in [
        ("clip-figure1-training-and-zero-shot.png", "clip.pdf", 2, (265, 324, 2725, 1213)),
        ("siglip-parallelism.png", "siglip.pdf", 3, (235, 342, 2745, 1461)),
    ]:
        target = curated / "vision-objectives-2026-11-09" / filename
        rendered = subprocess.check_output([
            "pdftoppm", "-png", "-r", "360", "-f", str(page_no), "-l", str(page_no),
            "-singlefile", str(SOURCE / pdf)])
        page = Image.open(io.BytesIO(rendered)).convert("RGB")
        assert page.size == (3060, 3960)
        assert ImageChops.difference(page.crop(rectangle), Image.open(target).convert("RGB")).getbbox() is None
        assert filename in chapter and filename in english
    assert len(re.findall(r"!\[\[", chapter)) == len(re.findall(r"!\[\[", english)) == 6
    print(json.dumps({"chapter_sha256": [digest(CHAPTER), digest(EN_CHAPTER)],
        "standalone_and_shared_blocks": 24,
        "patch_permutations": 24, "loss_gradient_samples": 40,
        "paired_metric_cases": 81, "pixel_identical_paper_figures": 5,
        "unchanged_course_figures": 1, "crop_controls": 3,
        "pretrained_model_run": False}, indent=2))

if __name__ == "__main__":
    main()

