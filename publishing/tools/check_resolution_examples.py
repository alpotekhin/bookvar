#!/usr/bin/env python3
"""Chapter 64b arithmetic, pinned code behavior and image provenance.

No pretrained VLM inference, OCR evaluation or performance claim.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import warnings

import numpy as np
from PIL import Image, ImageChops
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md"
SOURCE = ROOT / "05 Источники/Originals/resolution-2026-10-20"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/resolution-review-2026-10"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_function(path, name, **extra):
    """Compile only an audited pure function, not the source module imports."""
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    env = dict(np=np, torch=torch, math=math, warnings=warnings, Image=Image, **extra)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S)
    assert len(blocks) == 9
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        code = compile(block, f"{CHAPTER}:block-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)

    # Molmo's nested function counts patches before the final pooling division.
    molmo_count = source_function(
        SOURCE / "molmo-preprocess.py", "get_num_patches",
        crop_window_patches=16, crop_patches=24, left_margin=4, right_margin=4,
    )
    for n in range(1, 7):
        assert molmo_count(n, 2) // 2 == spaces[0]["molmo_axis"](n)
    assert spaces[0]["positions"] == 580

    # AnyRes canvas choice and exact row ordering, not only final counts.
    choose = source_function(SOURCE / "llavanext-mm-utils.py", "select_best_resolution")
    canvases = [(384, 384), (768, 384), (768, 768)]
    assert choose((768, 384), canvases) == (768, 384)
    unpad = source_function(SOURCE / "llavanext-arch.py", "unpad_image")
    grid = spaces[1]["grid"]
    source_tiles = torch.arange(2 * 27**2).reshape(1, 2, 27, 27, 1)
    source_grid = source_tiles.permute(4, 0, 2, 1, 3).contiguous()
    source_grid = source_grid.flatten(1, 2).flatten(2, 3)
    actual = unpad(source_grid, (768, 384))[0].numpy()
    assert np.array_equal(actual, grid)
    assert np.array_equal(grid[:, :27], np.arange(729).reshape(27, 27))
    assert np.array_equal(grid[:, 27:], np.arange(729, 1458).reshape(27, 27))
    # A real padding removal case uses the source function's integer convention.
    assert unpad(torch.zeros(1, 54, 54), (768, 384)).shape == (1, 28, 54)
    # anyres_max_1 interpolation of this 27x54 local grid.
    factor = math.sqrt(27 * 54 / 729)
    out_h, out_w = int(27 // factor), int(54 // factor)
    assert (out_h, out_w) == (19, 38)
    assert out_h * (out_w + 1) + 729 == 1470

    # README is Markdown: extract the complete Python helper block for parsing.
    readme = (SOURCE / "internvl-README.md").read_text()
    helpers = readme[readme.index("def find_closest_aspect_ratio("):]
    helpers = helpers[:helpers.index("def load_image(")]
    env = dict(Image=Image)
    exec(compile(helpers, "internvl-README.md:helpers", "exec"), env)
    image_counts = []
    for width, height, expected_grid in [
        (1600, 800, (4, 2)), (800, 400, (2, 1)), (448, 448, (1, 1))
    ]:
        assert spaces[2]["internvl_grid"](width, height) == expected_grid
        tiles = env["dynamic_preprocess"](
            Image.new("RGB", (width, height)), use_thumbnail=True,
        )
        expected_count = math.prod(expected_grid)
        expected_count += expected_count > 1
        assert len(tiles) == expected_count
        assert all(tile.size == (448, 448) for tile in tiles)
        image_counts.append(len(tiles))

    shuffle = source_function(SOURCE / "internvl-model.py", "pixel_shuffle")
    for n, h, w, c in [(1, 4, 4, 1), (2, 6, 8, 3), (1, 32, 32, 4)]:
        x = torch.arange(n*h*w*c).reshape(n, h, w, c)
        y = shuffle(SimpleNamespace(ps_version="v2"), x, .5)
        assert y.shape == (n, h//2, w//2, 4*c)
        assert torch.equal(y.sort().values.flatten().sort().values, x.flatten().sort().values)
        # Explicit neighborhood layout agrees for every output cell.
        expected = x.reshape(n, h//2, 2, w//2, 2, c)
        expected = expected.permute(0, 1, 3, 2, 4, 5).reshape(n, h//2, w//2, 4*c)
        assert torch.equal(y, expected)
    assert np.array_equal(spaces[3]["y"], shuffle(
        SimpleNamespace(ps_version="v2"), torch.tensor(spaces[3]["x"]), .5).numpy())

    # Compare the pedagogical resize implementation with the pinned pure function.
    smart = source_function(SOURCE / "qwen2vl-image.py", "smart_resize")
    resize = spaces[4]["resize_example"]
    rng = np.random.default_rng(20261020)
    resize_cases = 0
    for height, width in rng.integers(28, 10000, (200, 2)):
        if max(height, width) / min(height, width) > 200:
            continue
        got = smart(int(height), int(width), factor=28,
                    min_pixels=56**2, max_pixels=1280*28**2)
        assert got == resize(int(height), int(width))
        assert got[0] % 28 == got[1] % 28 == 0
        resize_cases += 1
    assert smart(1000, 1600) == (784, 1260)
    assert 224//14 == 16 and 16**2//4 + 2 == 66

    # The Pixtral processor's actual expansion statements use these same markers.
    pixtral = (SOURCE / "pixtral-processor.py").read_text()
    begin = pixtral.index("                    replace_tokens = [")
    end = pixtral.index("                    replace_str =", begin)
    import textwrap
    expand = compile(textwrap.dedent(pixtral[begin:end]), "pixtral:row-expansion", "exec")
    markers = SimpleNamespace(image_token="[IMG]", image_break_token="[IMG_BREAK]",
                              image_end_token="[IMG_END]")
    for h in range(1, 5):
        for w in range(1, 6):
            env = dict(self=markers, num_height_tokens=h, num_width_tokens=w)
            exec(expand, env)
            result = env["replace_tokens"]
            assert result == spaces[5]["pixtral_rows"](h, w)
            assert result.count("[IMG]") == h*w
            assert result.count("[IMG_BREAK]") == h-1
            assert result.count("[IMG_END]") == 1
    assert spaces[5]["allowed"].sum() == 12**2 + 4**2

    # Qwen2-VL exact 3D index builder; fake IDs, no tokenizer/weights/downloads.
    rope_index = source_function(SOURCE / "qwen2vl-model.py", "get_rope_index")
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=100, video_token_id=101, vision_start_token_id=99,
    )
    ids = torch.tensor([[10, 11, 99] + [100]*6 + [98, 12]])
    positions, delta = rope_index(
        SimpleNamespace(config=config), ids, image_grid_thw=torch.tensor([[1, 4, 6]])
    )
    assert np.array_equal(positions[:, 0].numpy(), spaces[6]["position_ids"])
    assert delta.tolist() == [[-3]]
    for h in range(1, 5):
        for w in range(1, 6):
            ids = torch.tensor([[10, 11, 99] + [100]*(h*w) + [98, 12]])
            positions, delta = rope_index(
                SimpleNamespace(config=config), ids,
                image_grid_thw=torch.tensor([[1, h*2, w*2]]),
            )
            assert positions[:, 0, -2].tolist() == [3+max(h,w)]*3
            assert int(delta.item()) == max(h,w)-h*w

    rotate = spaces[7]["rotate_three"]
    for _ in range(100):
        q, k = rng.normal(size=(2, 6))
        p, r, offset = rng.integers(-100, 100, (3, 3))
        assert np.isclose(rotate(q, p)@rotate(k, r),
                          rotate(q, p+offset)@rotate(k, r+offset))
    interleave = source_function(
        ROOT/"05 Источники/Originals/connectors-2026-10-20/qwen3vl-model.py",
        "apply_interleaved_mrope",
    )
    axes = torch.arange(3).reshape(3, 1, 1, 1).expand(3, 1, 1, 64).clone()
    assigned = interleave(None, axes, [24, 20, 20]).flatten()
    assert assigned.bincount().tolist() == [24, 20, 20]
    assert assigned[-4:].tolist() == [0, 0, 0, 0]

    inverse = spaces[8]["to_original"]
    for _ in range(100):
        box = np.sort(rng.uniform(0, 1000, (2, 2)), axis=0)
        scale = rng.uniform(.1, 3, 2)
        pad, origin = rng.uniform(0, 100, (2, 2))
        local = box*scale + pad-origin
        assert np.allclose(inverse(local, scale, pad, origin), box.ravel())
    assert 16*64**2 == 65_536 and 1024**2 == 1_048_576

    # Pixel equality, hashes and explicit paper-license evidence for the new crop.
    records = json.loads((FIGURES/"assets.json").read_text())
    assert len(records) == 1
    for row in records:
        for path_key, hash_key in [
            ("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
            ("license_evidence", "license_evidence_sha256"),
        ]:
            assert sha(ROOT/row[path_key]) == row[hash_key]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        crop = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), crop).getbbox() is None
        assert row["asset"] in chapter
        assert row["license"] == "CC BY 4.0"
    references = json.loads((SOURCE/"references.json").read_text())
    for row in references["downloaded"]:
        assert sha(ROOT/row["path"]) == row["sha256"]
    repair = json.loads((SOURCE/"molmo-repair.json").read_text())
    for path_key, hash_key in [("asset", "sha256"), ("source_raster", "source_raster_sha256")]:
        assert sha(ROOT/repair[path_key]) == repair[hash_key]
    before = Image.open(ROOT/repair["source_raster"]).convert("RGB")
    after = Image.open(ROOT/repair["asset"]).convert("RGB")
    assert list(before.size) == repair["source_raster_size_px"]
    assert ImageChops.difference(before.crop(repair["crop_box_px"]), after).getbbox() is None
    assert len(re.findall(r"!\[\[", chapter)) == 6
    print(json.dumps({
        "chapter_sha256": sha(CHAPTER), "standalone_and_shared_code_runs": 18,
        "pinned_resize_cases": resize_cases, "internvl_images_with_thumbnail": image_counts,
        "molmo_axis_cases": 6, "pixtral_row_shapes": 20, "mrope_grid_shapes": 20,
        "rotary_relative_position_cases": 100, "coordinate_roundtrips": 100,
        "interleaved_frequency_pairs": [24, 20, 20], "new_original_figure_crops": 1,
        "existing_figure_whitespace_repairs": 1,
        "pretrained_model_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
