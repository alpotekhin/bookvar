#!/usr/bin/env python3
"""Chapter 64e examples, source hashes and original figure crops; not a model benchmark."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md"
ENGLISH = ROOT / "en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md"
SOURCE = ROOT / "05 Источники/Originals/omni-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_blocks(path, fence):
    blocks = re.findall(rf"^{fence}python\n(.*?)^{fence}", path.read_text(), re.M | re.S)
    assert len(blocks) == 18, len(blocks)
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
    assert len(spaces) == len(english_spaces) == 18

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
    alignment, = [space for space in spaces if "words_at" in space]
    f = alignment["F"]
    words_at, segments = alignment["words_at"], alignment["segments"]
    assert words_at(segments, f("18.24")) == ["Я вернулся"]
    assert words_at(segments, f("18.72")) == []
    assert words_at(segments, f("18.4"), f(".01")) == ["Я вернулся"]
    assert 1180-1000 == 180 and 1055-1000 == 55
    offset = Fraction(".12")
    bounds = tuple(int((Fraction(t) - offset) * 16_000) for t in ("18.0", "18.8"))
    assert bounds == (286_080, 298_880)
    assert bounds[1] - bounds[0] == 12_800 and 288_000 - bounds[0] == 1920
    assert 552_000 * Fraction(1, 30_000) == Fraction("18.4")
    assert sum([72, 88, 57, 14, 3]) == 234
    assert sum([160, 160, 210, 14, 3]) == 547
    assert sum([94, 468, 145, 16, 5]) == 728
    assert round((1000/75 + 1000/140 + 14 + 3) / 80, 2) == .47

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
    reader_figures = {
        link.split("|", 1)[0].rsplit("/", 1)[-1]
        for link in re.findall(r"!\[\[([^\]\n]+)\]\]", chapter)
    }
    assert reader_figures == {
        "llava-onevision-modalities.png", "gens-frame-selection.png", "mimi-split-rvq.png",
        "qwen25omni-figure3-tmrope.png", "qwen25omni-figure2-thinker-talker.png",
        "qwen3omni-architecture.png", "qwen3omni-figure3-aut.png",
        "moshi-rq-transformer.png", "moshi-duplex-streams.png", "qwen25omni-block-attention.png",
        "panda-act-figure2.jpg",
    }
    assert "https://arxiv.org/pdf/2305.05665v2#page=1" in chapter
    journal = ROOT / "docs/audit/2026-11-13-internvl-imagebind-figures-additions.json"
    application = json.loads(journal.read_text())["figures"][1]
    assert application["asset"] in chapter and application["license"] == "CC BY-NC-ND 4.0"
    assert sha(ROOT / application["asset"]) == application["sha256"]
    assert sha(ROOT / application["license_evidence"]) == application["license_evidence_sha256"]
    image = Image.open(ROOT / application["asset"])
    assert image.format == "JPEG" and list(image.size) == application["dimensions_px"]
    retrieval, = [space for space in spaces if "scores" in space and "audio" in space]
    permutation = np.array([1, 2, 0])
    ranked = retrieval["images"][permutation] @ retrieval["audio"]
    assert int(permutation[ranked.argmax()]) == int(retrieval["scores"].argmax()) == 0
    assert "meeting-07-slides-xgen-video" in chapter and "meeting-07-slides-gens" in chapter
    print(json.dumps({
        "chapter_sha256": [sha(CHAPTER), sha(ENGLISH)],
        "standalone_and_shared_code_runs": 2 * (len(spaces) + len(english_spaces)),
        "offset_audio_slice": bounds, "qwen3_omni_rtf": .47,
        "causal_chunk_cases": 100, "residual_bookkeeping_cases": 100,
        "conditional_joint_sum": float(joint.sum()), "deduplicated_selected_frames": len(selected),
        "new_original_figures": 6, "repaired_historical_figures": 1,
        "unchanged_historical_figures": 3, "pretrained_model_run": False,
        "imagebind_application_original_jpeg_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
