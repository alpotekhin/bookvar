#!/usr/bin/env python3
"""Chapter 64h: arithmetic, pinned pure vocabulary methods and original crops.

No pretrained model, GPU inference, quality evaluation or performance benchmark.
"""
import ast
from functools import cached_property
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md"
ARCHIVE = ROOT / "05 Источники/Originals/chameleon-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned_vocabulary():
    tree = ast.parse((ARCHIVE / "vocab.py").read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"VocabInfo", "VocabTranslation"}]
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + classes, type_ignores=[]))
    namespace = {"cached_property": cached_property}
    exec(compile(module, "pinned-vocab-pure-properties", "exec"), namespace)
    values = {"<s>": 0, "</s>": 1, "<racm3:break>": 2, "<eoss>": 3,
              "<pad>": 4, "IMGIMGAZ": 11, "IMGIMGBZ": 7,
              "IMGIMGCZ": 19, "IMGIMGDZ": 5, "A": 30}
    vocab = namespace["VocabInfo"](values)
    translation = namespace["VocabTranslation"](vocab)
    assert vocab.begin_image == 2 and vocab.end_image == 3
    assert translation.img2bpe == {0: 11, 1: 7, 2: 19, 3: 5}
    assert [translation.bpe2img[i] for i in [7, 19, 5, 11]] == [1, 2, 3, 0]
    assert vocab.text_tokens == [30]


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    shared = {}
    for index, block in enumerate(blocks):
        code = compile(block, f"{CHAPTER}:example-{index+1}", "exec")
        exec(code, {})
        exec(code, shared)
    assert len(blocks) >= 14
    check_pinned_vocabulary()

    rng = np.random.default_rng(20261020)
    for _ in range(100):
        logits = rng.normal(size=16)
        shift = rng.uniform(-100, 100)
        p = np.exp(logits-logits.max())
        p /= p.sum()
        shifted = logits + shift
        p_shift = np.exp(shifted-shifted.max())
        p_shift /= p_shift.sum()
        assert np.allclose(p, p_shift)
        logz = np.logaddexp.reduce(logits)
        assert np.isclose(np.logaddexp.reduce(shifted)-logz, shift)

        step, lam = 1e-5, 1e-5
        analytic = 2*lam*logz*p
        numeric = np.zeros_like(logits)
        for i in range(16):
            delta = np.zeros_like(logits)
            delta[i] = step
            numeric[i] = lam*(np.logaddexp.reduce(logits+delta)**2
                             - np.logaddexp.reduce(logits-delta)**2)/(2*step)
        assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-10)

        book = rng.normal(size=(8, 4))
        latent = rng.normal(size=(5, 4))
        distances = ((latent[:, None]-book[None])**2).sum(-1)
        code = distances.argmin(-1)
        assert np.allclose(distances[np.arange(5), code], distances.min(-1))

    refs = json.loads((ARCHIVE / "references.json").read_text())
    for row in refs["downloaded"] + refs["reused"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert len(refs["figures"]) == 5
    for row in refs["figures"]:
        for field, digest in [("asset", "sha256"),
                              ("source_asset", "source_asset_sha256"),
                              ("source_page_snapshot", "source_page_sha256"),
                              ("license_evidence", "license_evidence_sha256")]:
            assert sha(ROOT / row[field]) == row[digest], (row["asset"], field)
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        crop = page.crop(tuple(row["crop_box_px"]))
        actual = Image.open(ROOT / row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert crop.size == actual.size
        assert ImageChops.difference(crop, actual).getbbox() is None
        assert row["license"] == "CC BY 4.0"
        assert "creativecommons.org/licenses/by/4.0/" in (ROOT / row["license_evidence"]).read_text()

    print(json.dumps({
        "chapter_sha256": sha(CHAPTER), "code_blocks": len(blocks),
        "standalone_and_shared_runs": 2*len(blocks),
        "randomized_identity_cases": 100,
        "pinned_vocabulary_methods_executed": True,
        "source_downloads_verified": len(refs["downloaded"]),
        "original_figure_crops_verified": len(refs["figures"]),
        "pretrained_model_run": False, "gpu_or_performance_benchmark": False
    }, indent=2))


if __name__ == "__main__":
    main()

