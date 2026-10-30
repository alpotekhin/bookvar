#!/usr/bin/env python3
"""Validate chapter 57a's displayed CPU examples, excerpts and original figures.

No model-quality test, GPU attention kernel or serving benchmark is run.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload.md"
SOURCES = ROOT / "05 Источники/Originals/kv-compression-2026-10-19"
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/kv-compression-review-2026-10"
OLD_H2 = [
    "Сколько байтов стоит один токен", "Три уровня сжатия", "KV-quantization",
    "Sparse attention и удаление состояния", "Offload как задача о критическом пути",
    "Проверка конфигурации", "Практика и первоисточники",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(text.split())


def check_sources(text):
    manifest = json.loads((SOURCES / "excerpts.json").read_text())
    for source in manifest["sources"]:
        assert digest(SOURCES / source["file"]) == source["sha256"], source["file"]
    for item in manifest["excerpts"]:
        original = (SOURCES / item["source"]).read_text()
        selected = original[original.index(item["start"]):original.index(item["end_exclusive"])].strip()
        if item["id"] == "streaming-faq":
            selected = re.sub(r"(?m)^\d+\. ", "", selected)
        displayed = re.search(
            rf"<!-- source-excerpt: {item['id']}:start -->(.*?)"
            rf"<!-- source-excerpt: {item['id']}:end -->", text, re.S
        )
        assert displayed, item["id"]
        quote = "\n".join(re.sub(r"^> ?", "", line) for line in displayed[1].splitlines()).strip()
        assert normalize(selected) == normalize(quote), item["id"]
    return {"source_hashes": len(manifest["sources"]), "exact_excerpts": len(manifest["excerpts"])}


def check_figures(text):
    items = json.loads((ASSETS / "assets.json").read_text())["assets"]
    rerenders = 0
    for item in items:
        output = ROOT / item["asset"]
        assert digest(output) == item["sha256"]
        assert struct.unpack(">II", output.read_bytes()[16:24]) == (item["width"], item["height"])
        if item["derivation"] != "pdf-page-render-crop":
            continue
        pdf = ROOT / item["source_asset"]
        assert digest(pdf) == item["source_asset_sha256"]
        archive = ROOT / item["source_archive"]
        assert digest(archive) == item["source_archive_sha256"]
        with tarfile.open(archive) as package:
            assert package.extractfile(item["source_member"]).read() == pdf.read_bytes()
        if shutil.which("pdftoppm"):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "figure"
                subprocess.run(
                    ["pdftoppm", "-singlefile", "-scale-to", str(item["width"]),
                     "-png", str(pdf), str(target)],
                    check=True, capture_output=True,
                )
                assert target.with_suffix(".png").read_bytes() == output.read_bytes()
            rerenders += 1
    embeds = re.findall(r"!\[\[([^\]]+)\]\]", text)
    assert len(embeds) == 6
    assert "kv-survey-figure4-sparsity.png" not in text
    for embed in embeds:
        assert (ROOT / embed.removeprefix("02 Areas/ML & DL/")).is_file(), embed
    return {"new_figures": len(items), "total_embeds": len(embeds), "full_pdf_rerenders": rerenders}


def main():
    text = CHAPTER.read_text()
    fence = chr(96) * 3
    blocks = re.findall(
        rf"<!-- checked-example: ([\w-]+) -->\s*{fence}python\n(.*?)\n{fence}", text, re.S
    )
    assert len(blocks) == 7
    namespace = {"__name__": "bookvar_kv_compression_example"}
    for ident, code in blocks:
        exec(compile(code, f"{CHAPTER}:{ident}", "exec"), namespace)
    rng = np.random.default_rng(57101919)
    quantize = namespace["minmax_restore"]
    for _ in range(500):
        x = rng.normal(size=(int(rng.integers(2, 25)), int(rng.integers(2, 25))))
        for axis in (0, 1):
            for bits in (2, 3, 4, 8):
                restored = quantize(x, axis, bits)
                step = np.ptp(x, axis=axis, keepdims=True) / (2**bits - 1)
                assert np.all(np.abs(restored - x) <= step / 2 + 1e-12)
    np.testing.assert_array_equal(quantize(np.full((4, 3), 7.0), 0), 7)

    # Independent incremental state machine versus the displayed closed-form counts.
    for residual in range(1, 17):
        k_low, k_high, v_low, v_high = 0, 0, 0, 0
        for length in range(1, 257):
            k_high += 1
            v_high += 1
            if k_high == residual:
                k_low += k_high
                k_high = 0
            if v_high > residual:
                v_low += v_high - residual
                v_high = residual
            assert namespace["residual_counts"](length, residual) == (k_low, k_high, v_low, v_high)
            assert k_low + k_high == v_low + v_high == length

    # Memory ledger uses decimal GB throughout.
    per_request = namespace["kv_bytes"]([17000], layers=94, kv_heads=4)
    assert per_request == 3272704000
    assert int((1128e9 - 470e9 - 20e9) // per_request) == 194
    assert int((640e9 - 470e9 - 20e9) // per_request) == 45

    # Removing a zero value still changes the normalization denominator.
    p = np.array([.6, .1, .1, .2])
    v = np.array([0., 10., 20., 30.])
    np.testing.assert_allclose([p @ v, (p[1:] / p[1:].sum()) @ v[1:]], [9., 22.5])
    for _ in range(400):
        p = namespace["softmax"](rng.normal(size=12))
        values = rng.normal(size=(12, 5))
        keep = rng.random(12) > .5
        if not keep.any() or keep.all():
            keep[0], keep[-1] = True, False
        epsilon = p[~keep].sum()
        kept = p[keep] @ values[keep] / p[keep].sum()
        dropped = p[~keep] @ values[~keep] / epsilon
        full = p @ values
        np.testing.assert_allclose(kept - full, epsilon * (kept - dropped), atol=1e-13)
        assert np.linalg.norm(kept - full) <= 2 * epsilon * np.linalg.norm(values, axis=1).max() + 1e-12

    # H2O: reserve the recent position, then choose two old accumulated scores.
    def choose_heavy(positions, scores, recent):
        old = [i for i in range(len(positions)) if positions[i] != recent]
        selected = sorted(old, key=lambda i: (-scores[i], positions[i]))[:2]
        return sorted([positions[i] for i in selected] + [recent])
    assert choose_heavy([0, 1, 2, 3], [1.4, 1.5, .5, .6], 3) == [0, 1, 3]
    assert choose_heavy([0, 1, 3, 4], [1.5, 1.55, .65, .8], 4) == [0, 1, 4]

    # Independent padded-window average versus the displayed NumPy convolution.
    for _ in range(200):
        score = rng.random(20)
        for width in (3, 5, 7):
            padded = np.pad(score, (width // 2, width // 2))
            expected = np.array([padded[i:i+width].sum() / width for i in range(len(score))])
            np.testing.assert_allclose(np.convolve(score, np.ones(width) / width, "same"), expected)

    # Pipeline event recurrence versus the displayed formula, not a measured timeline.
    for _ in range(200):
        layers = int(rng.integers(1, 100))
        transfer, compute = rng.uniform(1e-5, .05, size=2)
        end_compute = 0.0
        for i in range(layers):
            end_transfer = (i + 1) * transfer
            end_compute = max(end_compute, end_transfer) + compute
        formula = transfer + (layers - 1) * max(transfer, compute) + compute
        np.testing.assert_allclose(end_compute, formula, atol=1e-12)

    for heading in OLD_H2:
        assert f"## {heading}\n" in text
    print(json.dumps({
        "status": "PASS", "device": "CPU",
        "executed_markdown_blocks": [ident for ident, _ in blocks],
        "quantization_group_axis_bit_cases": 4000,
        "residual_state_steps": 4096,
        "eviction_error_identities": 400,
        "pooling_cases": 600, "pipeline_cases": 200,
        "sources": check_sources(text), "figures": check_figures(text),
        "gpu_execution": "NOT RUN", "model_quality": "NOT MEASURED",
        "benchmark": "NOT RUN",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
