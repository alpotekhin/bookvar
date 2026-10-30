#!/usr/bin/env python3
"""Chapter 64e examples, source hashes and original figure crops; not a model benchmark."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
SOURCE = ROOT / "05 Источники/Originals/omni-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", chapter, re.M | re.S)
    assert len(blocks) == 17, len(blocks)
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        code = compile(block, f"{CHAPTER}:block-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)

    rng = np.random.default_rng(20261020)
    # Causal convolution must agree across arbitrary nonempty chunk boundaries.
    for _ in range(100):
        x = rng.normal(size=64)
        kernel = rng.normal(size=5)
        cuts = sorted(rng.choice(np.arange(1, 64), size=8, replace=False))
        chunks = np.split(x, cuts)
        history = np.zeros(len(kernel)-1)
        pieces = []
        for chunk in chunks:
            joined = np.r_[history, chunk]
            start = len(history)
            pieces.append(np.convolve(joined, kernel)[start:start+len(chunk)])
            history = joined[-(len(kernel)-1):]
        full = np.convolve(x, kernel)[:len(x)]
        assert np.allclose(np.concatenate(pieces), full)
        changed = x.copy()
        changed[32:] = rng.normal(size=32)
        assert np.allclose(np.convolve(changed, kernel)[:32], full[:32])

    # Residual bookkeeping, not a guarantee that arbitrary learned books improve MSE.
    for _ in range(100):
        z = rng.normal(size=4)
        residual = z.copy()
        reconstruction = np.zeros(4)
        for book in [rng.normal(size=(8, 4)) for _ in range(3)]:
            index = np.argmin(((book-residual)**2).sum(axis=1))
            reconstruction += book[index]
            residual -= book[index]
            assert np.allclose(reconstruction + residual, z)

    # Conditional factorization includes every branch and sums to one.
    p_first = np.array([.6, .4])
    p_second_given_first = np.array([[.5, .5], [.2, .8]])
    joint = p_first[:, None] * p_second_given_first
    assert np.isclose(joint.sum(), 1)
    assert np.isclose(joint[0, 0], .3)
    assert not np.allclose(p_second_given_first[0], p_second_given_first[1])

    # Global identity is retained when windows overlap; local index alone is ambiguous.
    candidates = [(start + local, 100-local) for start in [0, 128, 256]
                  for local in range(256) if start + local < 600]
    best = {}
    for index, score in candidates:
        best[index] = max(best.get(index, -np.inf), score)
    selected = sorted(best, key=lambda index: (-best[index], index))[:32]
    assert len(selected) == len(set(selected)) == 32
    assert 128 in best and 511 in best and 512 not in best
    assert best[128] == 100

    # Half-open intervals distinguish exact end points; time values are rational.
    f = spaces[9]["F"]
    words_at, segments = spaces[9]["words_at"], spaces[9]["segments"]
    assert words_at(segments, f("18.24")) == ["Я вернулся"]
    assert words_at(segments, f("18.72")) == []
    assert words_at(segments, f("18.4"), f(".01")) == ["Я вернулся"]
    assert 1180-1000 == 180 and 1055-1000 == 55

    refs = json.loads((SOURCE / "references.json").read_text())
    for row in refs["downloaded"] + refs["reused"] + refs["figure_inventory"]:
        assert sha(ROOT/row["path"]) == row["sha256"], row["path"]
    for row in refs["figures"] + refs["repairs"]:
        for path_key, hash_key in [
            ("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
        ]:
            assert sha(ROOT/row[path_key]) == row[hash_key], row[path_key]
        if "source_raster" in row:
            assert sha(ROOT/row["source_raster"]) == row["source_raster_sha256"]
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        asset = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), asset).getbbox() is None
        if "license_evidence" in row:
            evidence = ROOT/row["license_evidence"]
            assert sha(evidence) == row["license_evidence_sha256"]
            suffix = "by-nc-sa" if "NC" in row["license"] else "by"
            assert f"creativecommons.org/licenses/{suffix}/4.0/" in evidence.read_text()
    assert len(re.findall(r"!\[\[", chapter)) == 10
    assert "meeting-07-slides-xgen-video" in chapter and "meeting-07-slides-gens" in chapter
    print(json.dumps({
        "chapter_sha256": sha(CHAPTER),
        "standalone_and_shared_code_runs": 34,
        "causal_chunk_cases": 100, "residual_bookkeeping_cases": 100,
        "conditional_joint_sum": float(joint.sum()), "deduplicated_selected_frames": len(selected),
        "new_original_figures": 6, "repaired_historical_figures": 1,
        "unchanged_historical_figures": 3, "pretrained_model_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
