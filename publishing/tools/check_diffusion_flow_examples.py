#!/usr/bin/env python3
"""Chapter 64g arithmetic and source excerpts. No training or GPU benchmark."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений.md"
ARCHIVE = ROOT / "05 Источники/Originals/diffusion-flow-2026-10-20"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    assert len(blocks) == 20, len(blocks)
    shared, spaces = {}, []
    for i, block in enumerate(blocks, 1):
        code = compile(block, f"{CHAPTER}:example-{i}", "exec")
        local = {}
        exec(code, local)
        exec(code, shared)
        spaces.append(local)

    rng = np.random.default_rng(20261020)
    for _ in range(100):
        betas = rng.uniform(0.0001, .5, 2)
        alphas = 1 - betas
        assert np.isclose(betas[1] + alphas[1] * betas[0], 1 - alphas.prod())
        angle = rng.uniform(.01, np.pi/2-.01)
        a, sigma = np.cos(angle), np.sin(angle)
        clean, noise = rng.normal(size=(2, 16))
        noisy = a*clean + sigma*noise
        v = a*noise - sigma*clean
        assert np.allclose(a*noisy - sigma*v, clean)
        assert np.allclose(sigma*noisy + a*v, noise)

        tau = rng.uniform(.01, .99)
        target = clean-noise
        point = (1-tau)*noise + tau*clean
        assert np.allclose((clean-point)/(1-tau), target)
        step = 1e-5
        derivative = (((1-tau-step)*noise+(tau+step)*clean)-point)/step
        assert np.allclose(derivative, target)

        targets = rng.normal(size=8)
        prediction = float(rng.normal())
        assert np.isclose(np.mean((targets-prediction)**2),
                          np.var(targets)+(targets.mean()-prediction)**2)
        gradient = 2*(prediction-targets.mean())
        finite = (np.mean((targets-prediction-step)**2)
                  - np.mean((targets-prediction+step)**2))/(2*step)
        assert np.isclose(finite, gradient)

    refs = json.loads((ARCHIVE/"references.json").read_text())
    for row in refs["downloaded"] + refs["reused"]:
        assert sha(ROOT/row["path"]) == row["sha256"], row["path"]
    assert len(refs["figures"]) == 7
    for row in refs["figures"]:
        for key, hash_key in [
            ("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"),
            ("license_evidence", "license_evidence_sha256")
        ]:
            assert sha(ROOT/row[key]) == row[hash_key], (row["asset"], key)
        page = Image.open(ROOT/row["source_page_snapshot"]).convert("RGB")
        actual = Image.open(ROOT/row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        crop = page.crop(tuple(row["crop_box_px"]))
        assert crop.size == actual.size
        assert ImageChops.difference(crop, actual).getbbox() is None
        if row["asset"].split("/")[-1].startswith("mit-"):
            assert row["license"] == "CC BY-NC-ND 4.0"
            assert row["source_asset"].endswith("/mit-notes-v3.pdf")
            assert "Unmodified complete figure and original English caption" in row["modifications"]
            evidence = (ROOT/row["license_evidence"]).read_text()
            assert "creativecommons.org/licenses/by-nc-nd/4.0/" in evidence
        elif row["asset"].endswith("/dit-architecture.png"):
            assert row["license"] == "CC BY 4.0"
            assert "creativecommons.org/licenses/by/4.0/" in (ROOT/row["license_evidence"]).read_text()
        else:
            assert row["license"] == "CC BY 4.0"
            assert (ROOT/row["license_evidence"]).read_bytes().startswith(b"%PDF-")

    headings = re.findall(r"^## (.+)$", text, re.M)
    assert len(headings) == 8
    assert "не является конфигурацией SD3" in text
    assert "77\\times2048" in text and "154" in text
    assert "условному среднему" in text
    print(json.dumps({
        "chapter_sha256": sha(CHAPTER),
        "code_blocks": len(blocks),
        "standalone_and_shared_runs": 2*len(blocks),
        "randomized_identity_cases": 100,
        "source_objects": len(refs["downloaded"])+len(refs["reused"]),
        "original_figure_excerpts": len(refs["figures"]),
        "ddpm_example_mean": float(spaces[3]["mean"]),
        "pretrained_model_run": False,
        "network_or_gpu_benchmark": False
    }, indent=2))


if __name__ == "__main__":
    main()
