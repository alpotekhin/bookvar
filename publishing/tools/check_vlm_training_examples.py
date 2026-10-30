#!/usr/bin/env python3
"""Chapter 64c worked steps and pinned-source checks, not VLM evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageChops
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md"
SOURCE = ROOT / "05 Источники/Originals/vlm-training-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_function(path, name, **extra):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    env = dict(torch=torch, **extra)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S)
    assert len(blocks) == 8
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        code = compile(block, f"{CHAPTER}:block-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)

    # Alignment toy: exact SGD update and nonzero gradient through a frozen head.
    first = spaces[0]
    assert torch.allclose(first["projector"].weight.detach(),
                          torch.tensor([[.1, 0.], [-.1, 0.]], dtype=torch.float64))
    assert first["after"] < first["before"]
    assert first["frozen_head"].weight.grad is None

    # An ignored predictor row contributes no direct logit gradient.
    logits = spaces[1]["logits"].detach().clone()
    labels = spaces[1]["labels"]
    baseline = torch.nn.functional.cross_entropy(logits[:-1], labels[1:])
    logits[:4] = torch.tensor([900., -900., 800., -800.])
    assert torch.equal(torch.nn.functional.cross_entropy(logits[:-1], labels[1:]), baseline)
    # But using same-position targets incorrectly would change the objective.
    assert not torch.isclose(torch.nn.functional.cross_entropy(
        spaces[1]["logits"][1:], labels[1:]), baseline)

    mask_cases = 0
    for left in range(1, 7):
        for right in range(1, 7):
            group = np.repeat([0, 1], [left, right])
            pos = np.arange(left + right)
            allowed = (group[:, None] == group[None, :]) & (pos[:, None] >= pos[None, :])
            assert not allowed[left:, :left].any()
            assert allowed.sum() == left*(left+1)//2 + right*(right+1)//2
            mask_cases += 1
    assert spaces[2]["shared"].sum() == 24

    sizes = spaces[3]["sizes"]
    for alpha in [0, .25, .5, .75, 1]:
        probability = sizes**alpha / (sizes**alpha).sum()
        assert np.isclose(probability.sum(), 1)
        assert np.all(probability > 0)
    assert np.allclose(sizes / sizes.sum(), [100/111, 10/111, 1/111])
    assert spaces[3]["token_share"][1] > 3 * spaces[3]["example_share"][1]

    # Original InternVL helper and packed-example target counting.
    length_weight = source_function(SOURCE/"internvl-train.py", "len2weight")
    packed_weights = source_function(SOURCE/"internvl-dataset-packed.py",
                                     "get_cu_seqlens_and_indexes", IGNORE_TOKEN_ID=-100)
    ids = torch.arange(24)
    group = torch.tensor([0]*6 + [1]*18)
    labels = torch.tensor([-100]*2 + [0]*4 + [-100]*2 + [0]*16)
    model_path = ROOT/"05 Источники/Originals/resolution-2026-10-20/internvl-model.py"
    source_tree = ast.parse(model_path.read_text())
    weighted_branch = next(
        node for node in ast.walk(source_tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
        and ast.unparse(node.test) == "labels is not None and loss_weight is not None"
    )
    # Run only the reviewed weighted-loss branch; no model imports, weights or distributed init.
    module = ast.Module(body=weighted_branch.body, type_ignores=[])
    weighted_code = compile(ast.fix_missing_locations(module), str(model_path)+":weighted-loss", "exec")
    actual_losses = {}
    for mode, expected in [("token", 1.2), ("square", 4/3), ("sample", 1.5)]:
        assert length_weight(0, mode) == 0
        cumulative, positions, weight = packed_weights(
            group, ids, labels, lambda n: length_weight(n, mode)
        )
        assert cumulative == [0, 6, 24]
        assert positions == list(range(6)) + list(range(18))
        weight = torch.where(labels == -100, 0, weight)
        probs = torch.full((1, 24, 2), .5, dtype=torch.float64)
        for i in range(1, len(labels)):
            if labels[i] != -100:
                loss_value = 2. if group[i] == 0 else 1.
                p = torch.exp(torch.tensor(-loss_value, dtype=torch.float64))
                probs[0, i-1, 0], probs[0, i-1, 1] = p, 1-p
        env = dict(
            torch=torch, CrossEntropyLoss=torch.nn.CrossEntropyLoss,
            self=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(vocab_size=2))),
            labels=labels.unsqueeze(0), loss_weight=weight.unsqueeze(0).tolist(),
            logits=probs.log(), loss_reduction_all_gather=False, ignore_flag=False,
        )
        exec(weighted_code, env)
        actual_losses[mode] = env["loss"].item()
        assert np.isclose(actual_losses[mode], expected)

    # Weight normalization does not commute with averaging uneven microbatches.
    assert not np.isclose((2+1)/2, (4*2+16*1)/20)
    assert 30_720 * 50_000 == 1_536_000_000
    assert 560_000 + 350_000 + 800_000 == 1_710_000

    # Original OneVision S1 arithmetic is retained visibly, not silently fixed in the image.
    area = spaces[5]["area"]
    assert not np.isclose(area, 38.01)
    assert not np.isclose(38.01*63, 2386.03)
    assert "арифметические" in chapter and "2661.12" in chapter

    rng = np.random.default_rng(20261020)
    iou = spaces[6]["iou"]
    for _ in range(100):
        lo = rng.uniform(0, 100, 2)
        hi = lo + rng.uniform(.1, 100, 2)
        box = np.r_[lo, hi]
        assert np.isclose(iou(box, box), 1)
        shifted = box + np.tile(rng.normal(size=2), 2)
        assert np.isclose(iou(box, shifted), iou(shifted, box))
        assert 0 <= iou(box, shifted) <= 1
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(5), size=2)
        assert np.sum(p*np.log(p/q)) >= -1e-12

    # Full source-crop reproducibility for repaired versions of the same figures.
    repairs = json.loads((SOURCE/"figure-repairs.json").read_text())
    assert len(repairs) == 2
    for row in repairs:
        for path_key, hash_key in [
            ("asset", "sha256"), ("source_raster", "source_raster_sha256"),
            ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
        ]:
            assert sha(ROOT/row[path_key]) == row[hash_key]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        crop = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), crop).getbbox() is None
    references = json.loads((SOURCE/"references.json").read_text())
    for row in references["downloaded"]:
        assert sha(ROOT/row["path"]) == row["sha256"]
    for row in references["reused_source_hashes"]:
        assert sha(ROOT/row["path"]) == row["sha256"]
    pinned = ROOT/"05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images"
    for row in references["figure_inventory"]:
        assert sha(ROOT/row["asset"]) == row["sha256"]
        if "stanford-cs336-2026" in row["asset"]:
            assert sha(pinned/Path(row["asset"]).name) == row["sha256"]
    assert len(re.findall(r"!\[\[", chapter)) == 8
    print(json.dumps({
        "chapter_sha256": sha(CHAPTER), "standalone_and_shared_code_runs": 16,
        "packing_mask_sizes": mask_cases, "pinned_internvl_losses": actual_losses,
        "source_pack_boundaries": [0, 6, 24], "iou_property_cases": 100,
        "kl_nonnegativity_cases": 100, "repaired_original_figures": 2,
        "unchanged_pinned_course_figures": 6, "pretrained_model_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
