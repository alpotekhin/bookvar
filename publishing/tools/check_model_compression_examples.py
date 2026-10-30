#!/usr/bin/env python3
"""Check chapter 57b's actual NumPy examples, excerpts and figure provenance.

These are CPU algebra checks, not GPU benchmarks or model-quality evaluations.
"""
from __future__ import annotations

import hashlib
import html
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
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank.md"
SOURCES = ROOT / "05 Источники/Originals/model-compression-2026-10-19"
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/model-compression-review-2026-10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return " ".join(value.split())


def check_sources(chapter):
    manifest = json.loads((SOURCES / "excerpts.json").read_text())
    for source in manifest["sources"]:
        assert digest(SOURCES / source["file"]) == source["sha256"], source["file"]
    assert digest(ROOT / "05 Источники/Papers/Source PDFs/hinton-distillation-1503.02531v1.pdf") == (
        "74e6689115c539db499f9d6153edecb364059926fd45402454f8210442444c5a"
    )
    for excerpt in manifest["excerpts"]:
        original = (SOURCES / excerpt["source"]).read_text()
        if "html_ids" in excerpt:
            fragments = []
            for ident in excerpt["html_ids"]:
                match = re.search(rf'<span id="{re.escape(ident)}"[^>]*>(.*?)</span>', original, re.S)
                assert match, ident
                fragments.append(html.unescape(re.sub(r"<[^>]+>", "", match[1])))
            selected = "\n\n".join(fragments)
        else:
            selected = original[original.index(excerpt["start"]):original.index(excerpt["end_exclusive"])]
        displayed = re.search(
            rf'<!-- source-excerpt: {excerpt["id"]}:start -->(.*?)'
            rf'<!-- source-excerpt: {excerpt["id"]}:end -->', chapter, re.S
        )
        assert displayed, excerpt["id"]
        quote = "\n".join(re.sub(r"^> ?", "", line) for line in displayed[1].splitlines())
        assert normalize(selected) == normalize(quote), excerpt["id"]
    return {"source_hashes": len(manifest["sources"]) + 1, "exact_excerpts": len(manifest["excerpts"])}


def check_figures(chapter):
    items = json.loads((ASSETS / "assets.json").read_text())["assets"]
    rerenders = 0
    for item in items:
        image = ROOT / item["asset"]
        assert digest(image) == item["sha256"]
        assert struct.unpack(">II", image.read_bytes()[16:24]) == (item["width"], item["height"])
        pdf = ROOT / item["source_asset"]
        archive = ROOT / item["source_archive"]
        assert digest(pdf) == item["source_asset_sha256"]
        assert digest(archive) == item["source_archive_sha256"]
        with tarfile.open(archive) as package:
            assert package.extractfile(item["source_member"]).read() == pdf.read_bytes()
        if shutil.which("pdftoppm"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "figure"
                subprocess.run(
                    ["pdftoppm", "-singlefile", "-scale-to", str(item["width"]),
                     "-png", str(pdf), str(output)], check=True, capture_output=True
                )
                assert output.with_suffix(".png").read_bytes() == image.read_bytes()
            rerenders += 1
    unchanged_svg = {
        "5b8ca5d44a1556b84569cdc9156a94c4b1500e92": "6495864f718014d687f96bc7f52e05f505ba26d4b64b89898f426bee9c5060d0",
        "fc4813524b57d9439c3a0b9101f2c1f2a79382c3": "c91056b0c1fa5d1dd563f2c25add405bbbbd234b83450586c337a61517dae534",
        "4000264baf2311c2b61ca0ddd3fcedaa9d047ed3": "98e148cd5a8e2f510824a3f2f6c7e01f6f9e17932b759a14d96c39315b83d2ce",
        "dcbd6535b09a28528559354f928838baaae7644b": "23dd3b216eccd3993963d7686da137e7220c9cd52f82ba56e1ed34ebf1a20e02",
        "708770dcb60d00ea7f382f39d15b3206ba6ccb1f": "e844be4c3e46e1ea2b5c6ac03db47edab3ae062e19930f107a3889cf5853b692",
    }
    for name, expected in unchanged_svg.items():
        assert digest(ROOT / f"Assets/Sources/Harvard ML Systems/vol1/model_compression/{name}.svg") == expected
    embeds = re.findall(r"!\[\[([^\]]+)\]\]", chapter)
    # Tensor factorization and the later pruning/NAS/exit additions are additive.
    assert len(embeds) == 12
    for embed in embeds:
        assert (ROOT / embed.removeprefix("02 Areas/ML & DL/")).is_file(), embed
    return {"total_figures": len(embeds), "unchanged_source_svgs": len(unchanged_svg),
            "new_vector_pdf_rerenders": rerenders}


def main():
    chapter = CHAPTER.read_text()
    fence = chr(96) * 3
    blocks = re.findall(
        rf"<!-- checked-example: ([\w-]+) -->\s*{fence}python\n(.*?)\n{fence}", chapter, re.S
    )
    assert [ident for ident, _ in blocks] == [
        "compression-magnitude", "compression-wanda", "compression-structured",
        "compression-pruning-schedule",
        "compression-svd", "tensor-cp-als", "tensor-tucker-cross",
        "tensor-factorized-conv", "tensor-cp-conv", "tensor-train-entry",
        "compression-distillation", "compression-budget-and-fusion",
        "compression-nas-relaxation", "compression-early-exit",
        "compression-compound-scaling", "compression-bn-folding", "compression-binary-dot"
    ]
    namespace = {"__name__": "bookvar_compression_example"}
    for ident, code in blocks:
        exec(compile(code, f"{CHAPTER}:{ident}", "exec"), namespace)

    rng = np.random.default_rng(571019)
    # Exact counts with tied magnitudes; a percentile comparison need not give k zeros.
    tied = np.ones(8)
    keep = np.ones(8, dtype=bool)
    keep[np.argsort(abs(tied), kind="stable")[:4]] = False
    assert keep.sum() == 4
    assert (abs(tied) > np.percentile(abs(tied), 50)).sum() == 0
    assert 8 * 2 == 16
    assert 4 * 2 + 4 * 4 + (2 + 1) * 4 == 36

    # The per-output Wanda score uses columns in this chapter's X @ W convention.
    for _ in range(300):
        x = rng.normal(size=(9, 5))
        w = rng.normal(size=(5, 4))
        score = abs(w) * np.linalg.norm(x, axis=0)[:, None]
        for j in range(5):
            for k in range(4):
                masked = w.copy()
                masked[j, k] = 0
                error = np.linalg.norm((x @ w - x @ masked)[:, k])
                np.testing.assert_allclose(error, score[j, k], atol=1e-13)
    future = np.array([0., 1000.])
    assert abs(future @ (np.array([.05, 0]) - np.array([.05, 1]))) == 1000
    assert abs(future @ (np.array([0, 1]) - np.array([.05, 1]))) == 0

    # Physical width removal equals a post-activation mask, including both biases.
    for _ in range(400):
        x, a, b = rng.normal(size=(7, 3)), rng.normal(size=(3, 8)), rng.normal(size=(8, 4))
        bias_a, bias_b = rng.normal(size=8), rng.normal(size=4)
        chosen = rng.choice(8, size=4, replace=False)
        mask = np.zeros(8)
        mask[chosen] = 1
        full_hidden = np.maximum(x @ a + bias_a, 0)
        masked = (full_hidden * mask) @ b + bias_b
        compact = np.maximum(x @ a[:, chosen] + bias_a[chosen], 0) @ b[chosen] + bias_b
        np.testing.assert_allclose(masked, compact, atol=1e-13)
    # Zeroing only input weights lets a positive removed bias survive.
    assert np.maximum(np.array([[0., 0.]]) + [2., 3.], 0).sum() == 5

    # Truncated SVD error, shapes, input error, and the exact displayed memory table.
    for _ in range(400):
        m, n = int(rng.integers(2, 15)), int(rng.integers(2, 15))
        w = rng.normal(size=(m, n))
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        rank = int(rng.integers(1, min(m, n)))
        a = u[:, :rank] * np.sqrt(s[:rank])
        b = np.sqrt(s[:rank])[:, None] * vt[:rank]
        assert a.shape == (m, rank) and b.shape == (rank, n)
        np.testing.assert_allclose(np.linalg.norm(w - a @ b)**2, np.sum(s[rank:]**2), atol=1e-12)
    for rank, params, mib, kib in [(512, 4194304, 8, 32), (2048, 16777216, 32, 128),
                                   (3072, 25165824, 48, 192)]:
        assert rank * 8192 == params
        assert params * 2 == mib * 2**20
        assert 32 * rank * 2 == kib * 2**10
        assert f"{params:,}".replace(",", " ") in chapter

    # Analytic tau^2 forward-KL gradient versus central finite differences.
    softmax = namespace["softmax_temperature"]
    for _ in range(150):
        teacher, student = rng.normal(size=7), rng.normal(size=7)
        temperature = rng.uniform(.5, 5)
        p = softmax(teacher, temperature)
        q = softmax(student, temperature)
        def loss(logits):
            return temperature**2 * np.sum(p * (np.log(p) - np.log(softmax(logits, temperature))))
        gradient = temperature * (q - p)
        numeric = np.array([
            (loss(student + np.eye(7)[i] * 1e-5) - loss(student - np.eye(7)[i] * 1e-5)) / 2e-5
            for i in range(7)
        ])
        np.testing.assert_allclose(gradient, numeric, atol=3e-9)
        assert loss(student - .01 * gradient) < loss(student)
        ce = -temperature**2 * (p @ np.log(q))
        entropy = -temperature**2 * (p @ np.log(p))
        np.testing.assert_allclose(ce - loss(student), entropy, atol=1e-13)
        hard_gradient = softmax(student, 1) - np.eye(7)[0]
        hard_numeric = np.array([
            (-np.log(softmax(student + np.eye(7)[i]*1e-5, 1)[0])
             + np.log(softmax(student - np.eye(7)[i]*1e-5, 1)[0])) / 2e-5
            for i in range(7)
        ])
        np.testing.assert_allclose(hard_gradient, hard_numeric, atol=1e-9)

    # Legacy section fragments and the three original course source-unit IDs survive.
    for fragment in [
        "pruning-параметр-равен-нулю--но-исчезла-ли-работа",
        "knowledge-distillation-передать-функцию-а-не-веса",
        "architecture-optimization-и-graph-optimization--разные-уровни",
        "minitron-структурное-pruning-как-измеряемая-кампания"
    ]:
        assert f'id="{fragment}"' in chapter
    for unit in ["lecture-10-model-pruning-distillation", "lecture-10-figure-step-325-rendering-1",
                 "lecture-10-figure-step-331-rendering-1"]:
        assert unit in chapter
    print(json.dumps({"displayed_python_blocks": len(blocks), "wanda_trials": 300,
                      "structured_trials": 400, "svd_trials": 400, "gradient_trials": 150,
                      **check_sources(chapter), **check_figures(chapter)}, indent=2))


if __name__ == "__main__":
    main()

