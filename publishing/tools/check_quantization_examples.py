#!/usr/bin/env python3
"""Check chapter 57's actual Markdown examples, sources and figure files.

CPU mathematics only. No quantized model, CUDA kernel or benchmark is run.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md"
SOURCES = ROOT / "05 Источники/Originals/quantization-2026-10-19"
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/quantization-review-2026-10"
LEGACY_H2 = ["## Действительное число заменяется кодом и масштабом","## Симметричная и асимметричная сетка","## Что именно квантуется","## PTQ: квантизация уже обученной модели","## QAT: модель видит ошибку во время обучения","## NF4 и QLoRA решают задачу дообучения","## Формат файла не равен формату вычисления","## Почему средняя perplexity недостаточна","## Проверка готового артефакта","## Практика и первоисточники"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_error(function, *args):
    try:
        function(*args)
    except (ValueError, TypeError):
        return
    raise AssertionError(f"Expected input rejection: {args!r}")


def normalize(text):
    return " ".join(text.split())


def displayed_excerpt(text, ident):
    found = re.search(
        rf"<!-- source-excerpt: {ident}:start -->([\s\S]*?)"
        rf"<!-- source-excerpt: {ident}:end -->", text
    )
    assert found, ident
    return "\n".join(re.sub(r"^> ?", "", line) for line in found[1].splitlines()).strip()


def check_sources(text):
    manifest = json.loads((SOURCES / "excerpts.json").read_text())
    for source in manifest["sources"]:
        assert digest(SOURCES / source["file"]) == source["sha256"], source["file"]
    hf = (SOURCES / "hf-concept-guide.md").read_text()
    item = manifest["excerpts"][0]
    selected = hf[hf.index(item["start"]):hf.index(item["end_exclusive"])].strip()
    assert normalize(selected) == normalize(displayed_excerpt(text, item["id"]))

    item = manifest["excerpts"][1]
    archive = ROOT / item["archive"]
    assert digest(archive) == item["archive_sha256"]
    with tarfile.open(archive) as package:
        tex = package.extractfile(item["member"]).read().decode()
        figure = package.extractfile("figures/qlora.pdf").read()
    selected = tex[tex.index(item["start"]):tex.index(item["end_exclusive"])].strip()
    snapshot = (SOURCES / item["source"]).read_text().strip()
    assert normalize(selected) == normalize(snapshot)
    selected = selected[selected.index("We introduce"):]
    for old, new in [
        (r"\emph{\Doublequant}", "*Double Quantization*"),
        (r"\Doublequant", "Double Quantization"),
        (r"~\citep{dettmers2022case}", " [13]"),
        (r"\citet{dettmers2022case}", "Dettmers and Zettlemoyer [13]"),
    ]:
        selected = selected.replace(old, new)
    assert normalize(selected) == normalize(displayed_excerpt(text, item["id"]))
    pdf = ROOT / "05 Источники/Papers/Source PDFs/qlora-2305.14314v1-figure1.pdf"
    assert pdf.read_bytes() == figure
    assert "/licenses/transformers-1d9654e7-Apache-2.0.txt" in text
    assert "/licenses/smoothquant-c61476d7-MIT.txt" in text
    return {"unchanged_sources": len(manifest["sources"]), "exact_excerpts": 2}


def check_assets(text):
    manifest = json.loads((ASSETS / "assets.json").read_text())
    for item in manifest["assets"]:
        path = ROOT / item["asset"]
        assert digest(path) == item["sha256"], path
        if path.suffix == ".png":
            blob = path.read_bytes()
            assert blob[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", blob[16:24]) == (item["width"], item["height"])
    embeds = re.findall(r"!\[\[([^\]]+)\]\]", text)
    assert len(embeds) == 6
    for path in embeds:
        relative = path.removeprefix("02 Areas/ML & DL/")
        assert (ROOT / relative).is_file(), relative
    rendering = "not run: pdftoppm unavailable"
    if shutil.which("pdftoppm"):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "figure"
            subprocess.run([
                "pdftoppm", "-r", "250", "-png", "-singlefile",
                str(ROOT / "05 Источники/Papers/Source PDFs/qlora-2305.14314v1-figure1.pdf"),
                str(output),
            ], check=True, capture_output=True)
            assert output.with_suffix(".png").read_bytes() == (ASSETS / "qlora-training-original.png").read_bytes()
            rendering = "full original figure reproduced byte-for-byte"
    return {"new_figure_hashes": 4, "chapter_embeds": len(embeds), "qlora_render": rendering}


def main():
    text = CHAPTER.read_text()
    fence = chr(96) * 3
    blocks = re.findall(
        rf"<!-- checked-example: ([\w-]+) -->\s*{fence}python\n(.*?)\n{fence}",
        text, re.S
    )
    assert len(blocks) == 6
    namespace = {"__name__": "bookvar_quantization_example"}
    for ident, code in blocks:
        exec(compile(code, f"{CHAPTER}:{ident}", "exec"), namespace)

    quantize = namespace["symmetric_quantize"]
    rng = np.random.default_rng(571019)
    for bits in range(2, 9):
        qmax = 2 ** (bits - 1) - 1
        for _ in range(250):
            values = rng.normal(size=int(rng.integers(1, 65))) * 10 ** rng.uniform(-5, 5)
            codes, scale, restored = quantize(values, bits)
            assert len(codes) == len(values)
            assert all(-qmax <= q <= qmax for q in codes)
            assert all(isinstance(q, int) for q in codes)
            np.testing.assert_allclose(restored, np.array(codes) * scale)
            assert np.max(np.abs(values - restored)) <= scale / 2 + 1e-10
    for values, bits in [([], 4), ([math.nan], 4), ([math.inf], 4), ([1], 1), ([1], True)]:
        require_error(quantize, values, bits)
    pack, unpack = namespace["pack_int4_pair"], namespace["unpack_int4_pair"]
    for a in range(-8, 8):
        for b in range(-8, 8):
            assert unpack(pack(a, b)) == (a, b)
    for a, b in [(-9, 0), (8, 0), (0, 8), (0.5, 1)]:
        require_error(pack, a, b)
    for byte in [-1, 256, 0.5]:
        require_error(unpack, byte)

    # Affine example, including exact representability of zero.
    weights = np.array([-1., 0., 1., 2.])
    scale, zero = 3 / 15, 5
    codes = np.clip(np.rint(weights / scale) + zero, 0, 15)
    np.testing.assert_array_equal(codes, [0, 5, 10, 15])
    np.testing.assert_allclose(scale * (codes - zero), weights)

    # LLM.int8 decomposition example: one exactly representable FP16 channel.
    x8, w8 = np.array([1024., 1.]), np.array([1 / 1024, 1.])
    assert x8 @ w8 == 2
    assert np.array(quantize(x8, 8)[2]) @ np.array(quantize(w8, 8)[2]) == 0
    assert np.float16(x8[0]) * np.float16(w8[0]) == 1
    assert x8[0] * w8[0] + np.array(quantize(x8[1:], 8)[2]) @ np.array(quantize(w8[1:], 8)[2]) == 2

    # Compensation formula versus an independently solved least-squares problem.
    for _ in range(400):
        width = int(rng.integers(2, 9))
        x = rng.normal(size=(width + 6, width))
        w = rng.normal(size=width)
        h_inv = np.linalg.inv(2 * x.T @ x)
        q0 = np.rint(w[0])
        compensated = w - (w[0] - q0) / h_inv[0, 0] * h_inv[:, 0]
        remaining = np.linalg.lstsq(x[:, 1:], x @ w - x[:, 0] * q0, rcond=None)[0]
        np.testing.assert_allclose(compensated, np.r_[q0, remaining], atol=3e-12)
        uncompensated = w.copy()
        uncompensated[0] = q0
        assert np.linalg.norm(x @ compensated - x @ w) <= np.linalg.norm(x @ uncompensated - x @ w) + 1e-12

    # Equivalent scaling and rotations before quantization; no universal accuracy gain.
    improved = worsened = 0
    for _ in range(400):
        x, w = rng.normal(size=(5, 8)), rng.normal(size=(8, 3))
        diagonal = np.exp(rng.uniform(-1, 1, size=8))
        np.testing.assert_allclose((x / diagonal) @ (w * diagonal[:, None]), x @ w, atol=1e-13)
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        np.testing.assert_allclose((x @ rotation) @ (rotation.T @ w), x @ w, atol=1e-13)
        plain = np.array(quantize(w.ravel(), 3)[2]).reshape(w.shape)
        scaled = np.array(quantize((w * diagonal[:, None]).ravel(), 3)[2]).reshape(w.shape)
        old_error = np.linalg.norm(x @ plain - x @ w)
        new_error = np.linalg.norm((x / diagonal) @ scaled - x @ w)
        improved += new_error < old_error
        worsened += new_error > old_error
    assert improved and worsened

    # Fixed-grid STE is deliberately not the derivative of rounding.
    fake = lambda w: np.clip(np.rint(w / 0.1), -7, 7) * 0.1
    latent = 0.26
    derivative = (fake(latent + 1e-6) - fake(latent - 1e-6)) / 2e-6
    assert derivative == 0
    surrogate = 2 * (fake(latent) - 1)
    updated = latent - 0.1 * surrogate
    np.testing.assert_allclose([surrogate, updated], [-1.4, 0.4])
    np.testing.assert_allclose([(fake(latent) - 1) ** 2, (fake(updated) - 1) ** 2], [0.49, 0.36])

    # E2M1 values from exponent/mantissa fields; shared-scale example.
    e2m1 = np.array([m * 0.5 if e == 0 else (1 + m / 2) * 2 ** (e - 1)
                    for e in range(4) for m in range(2)])
    np.testing.assert_array_equal(e2m1, [0, .5, 1, 1.5, 2, 3, 4, 6])
    original = np.r_[[.5, 1, 2, 6], np.zeros(12), [8, 16, 32, 96], np.zeros(12)]
    def reconstruct(values, scale):
        distances = np.abs(values[:, None] / scale - e2m1[None, :])
        return e2m1[np.argmin(distances, axis=1)] * scale
    separate = np.r_[reconstruct(original[:16], 1), reconstruct(original[16:], 16)]
    combined = reconstruct(original, 16)
    np.testing.assert_array_equal(separate, original)
    np.testing.assert_allclose(np.mean((combined - original) ** 2), .2890625)
    assert (1 + 6 / 8) * 2 ** (15 - 7) == 448
    assert (1 + 3 / 4) * 2 ** (30 - 15) == 57344
    assert 4 + 8 / 32 == 4.25 and 4 + 8 / 16 == 4.5
    dq_cost = 4 + 8 / 64 + 32 / (64 * 256)
    np.testing.assert_allclose(dq_cost, 4.126953125)
    assert 4096 ** 2 * 2 / 2**20 == 32
    assert 4096 ** 2 / 2 / 2**20 + 4096 ** 2 / 128 * 2 / 2**20 == 8.25
    for heading in LEGACY_H2:
        assert heading in text, heading
    for anchor in ["worked-execution-weight-only-против-wa",
                   "rotation-methods-где-возникает-и-куда-исчезает-поворот"]:
        assert f'id="{anchor}"' in text
    print(json.dumps({
        "status": "PASS", "device": "CPU", "executed_markdown_blocks": [b[0] for b in blocks],
        "symmetric_random_groups": 1750, "all_int4_pairs": 256, "least_squares_cases": 400,
        "equivalent_transform_cases": 400, "scaling_not_universally_better": {
            "improved": int(improved), "worsened": int(worsened)},
        "ste_true_local_derivative": float(derivative), "dq_bits_per_weight": dq_cost,
        "sources": check_sources(text), "figures": check_assets(text),
        "gpu_execution": "NOT RUN", "model_quality": "NOT MEASURED", "benchmark": "NOT RUN",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
