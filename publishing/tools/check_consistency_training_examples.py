"""Check displayed AT/VAT/UDA programs, independent derivatives and original figures.

CPU fixtures only. No TensorFlow-1 installation, translation checkpoint,
LSTM/BERT training, robustness certificate or benchmark reproduction.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pymupdf
from PIL import Image
import torch
import torch.nn.functional as F
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "05 Источники/Originals/consistency-training-2026-10-29"
ROBUST = ROOT / "00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем.md"
SEMI = ROOT / "00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md"
torch.set_num_threads(1)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    counts, scopes, captured = {}, {}, io.StringIO()
    for chapter, count in [(ROBUST, 3), (SEMI, 10)]:
        checked_text = before_topic_consolidation(chapter.read_text(), chapter,
            stop_before_journal="2026-10-29-consistency-training-additions.json")
        blocks = re.findall(r"^```python\n(.*?)^```\s*$", checked_text, re.M | re.S)
        assert len(blocks) == count
        scopes[chapter] = []
        for i, block in enumerate(blocks):
            scope = {}
            with contextlib.redirect_stdout(captured):
                exec(compile(block, f"{chapter}:python-{i+1}", "exec"), scope)
            scopes[chapter].append(scope)
        counts[chapter.relative_to(ROOT).as_posix()] = len(blocks)

    # General linear-softmax Hessian, independently derived from its Fisher matrix.
    generator = torch.Generator().manual_seed(29)
    hessian_cases = 0
    for classes in (2, 3, 5):
        for dimensions in (2, 4):
            for _ in range(3):
                W = torch.randn(classes, dimensions, generator=generator, dtype=torch.float64)
                x = torch.randn(dimensions, generator=generator, dtype=torch.float64)
                q = (W @ x).softmax(0).detach()

                def kl(r):
                    return (q * (q.log() - (W @ (x + r)).log_softmax(0))).sum()

                zero = torch.zeros_like(x, requires_grad=True)
                gradient, = torch.autograd.grad(kl(zero), zero)
                torch.testing.assert_close(gradient, torch.zeros_like(x), atol=1e-13, rtol=0)
                H = torch.autograd.functional.hessian(kl, zero)
                fisher = torch.diag(q) - torch.outer(q, q)
                torch.testing.assert_close(H, W.T @ fisher @ W, atol=1e-12, rtol=1e-12)
                d = torch.randn(dimensions, generator=generator, dtype=torch.float64)
                d /= d.norm()
                xi = 1e-5
                probe = (xi * d).requires_grad_()
                g, = torch.autograd.grad(kl(probe), probe)
                torch.testing.assert_close(g / xi, H @ d, atol=5e-5, rtol=1e-4)
                hessian_cases += 1

    # Independently check the scalar FreeLB gradient at fixed selected perturbations.
    original = scopes[ROBUST][1]
    assert np.allclose(original["points"], [0, -.15, -.3])
    assert abs(original["average"].item() + .036) < 1e-14
    assert original["accumulated_update"] > .8 > original["last_point_update"]
    fixed_points = np.array(original["points"])
    def average_loss(theta):
        return np.mean(.5 * (theta * (.2 + fixed_points) - 1) ** 2)
    derivative = (average_loss(.8 + 1e-5) - average_loss(.8 - 1e-5)) / 2e-5
    assert abs(derivative - original["average"].item()) < 1e-11

    # SMART grid optimum versus an independent analytic derivative and bisection.
    best = scopes[ROBUST][2]["best"]
    for mu, grid in zip((1, 5), best[1:]):
        def derivative(p):
            return (-1 / p + mu * (
                np.log(p / (1-p)) - np.log(.2 / .8)
                + (p - .2) / (p * (1-p))))
        lo, hi = .2, .99999
        for _ in range(80):
            mid = (lo + hi) / 2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        assert abs(grid - (lo + hi) / 2) <= 1e-4

    # Soft-target UDA: full-batch reduction, both loss values, all masked gradients.
    uda_cases = 0
    for n in (1, 3, 8):
        for classes in (2, 3, 5):
            raw = torch.randn(n, classes, generator=generator, dtype=torch.float64)
            for temperature in (.3, .5, 1.):
                target = (raw / temperature).softmax(-1).detach()
                for threshold in (.0, .8, 1.):
                    logits = torch.randn(n, classes, generator=generator,
                                         dtype=torch.float64, requires_grad=True)
                    mask = raw.softmax(-1).max(-1).values > threshold
                    values = (target * (target.log() - logits.log_softmax(-1))).sum(-1)
                    loss = (values * mask).mean()
                    gradient, = torch.autograd.grad(loss, logits)
                    expected = (logits.detach().softmax(-1) - target) * mask[:, None] / n
                    torch.testing.assert_close(gradient, expected, atol=1e-13, rtol=1e-12)
                    if not mask.any():
                        assert loss.item() == 0 and gradient.count_nonzero() == 0
                    uda_cases += 1

    # Execute the exact author's TSA function with a tiny numeric TF-operation shim.
    # This checks its arithmetic, not TensorFlow graph/session behaviour.
    source = (SRC / "uda-text.py").read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "get_tsa_threshold")
    module = ast.Module(body=[function], type_ignores=[])
    scope = {"tf": SimpleNamespace(to_float=float, exp=np.exp)}
    exec(compile(module, "original-uda-get_tsa_threshold", "exec"), scope)
    tsa_cases = 0
    for classes in (2, 3, 5):
        for u in (0., .25, .5, .75, 1.):
            for name, coefficient in (
                ("linear_schedule", u),
                ("exp_schedule", np.exp(5 * (u - 1))),
                ("log_schedule", 1 - np.exp(-5 * u)),
            ):
                result = scope["get_tsa_threshold"](name, u * 100, 100, 1/classes, 1)
                assert abs(result - (1/classes + (1 - 1/classes) * coefficient)) < 1e-14
                tsa_cases += 1
    assert "tf.greater(\n          correct_label_probs, tsa_threshold)" in source
    assert "tf.maximum(tf.reduce_sum(loss_mask), 1)" in source
    assert "tf.reduce_mean(per_example_kl_loss)" in source
    assert "largest_prob, FLAGS.uda_confidence_thresh" in source
    original_text = (SRC / "tf-adversarial_losses.py").read_text()
    assert "length - 1" in original_text
    assert "Scale over the full sequence, dims (1, 2)" in original_text
    assert "logits = tf.stop_gradient(logits)" in original_text

    manifest = json.loads((SRC / "manifest.json").read_text())
    for entry in manifest["sources"]:
        raw = (SRC / entry["file"]).read_bytes()
        assert digest(raw) == entry["sha256"] and len(raw) == entry["bytes"]
    assets = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/assets.json").read_text())
    for entry in assets:
        path = ROOT / entry["asset"]
        assert digest(path.read_bytes()) == entry["sha256"]
        assert digest((ROOT / entry["source_asset"]).read_bytes()) == entry["source_asset_sha256"]
        assert digest((ROOT / entry["license_evidence"]).read_bytes()) == entry["license_evidence_sha256"]
        with Image.open(path) as picture:
            assert list(picture.size) == entry["dimensions_px"]
            if entry["derivation"] == "source-gif-copy":
                assert picture.n_frames == 14
                assert path.read_bytes() == (ROOT / entry["source_asset"]).read_bytes()
                assert sum(picture.seek(i) is None for i in range(picture.n_frames)) == 14
            else:
                with pymupdf.open(ROOT / entry["source_asset"]) as pdf:
                    page = pdf[entry["source_page"] - 1]
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(400/72, 400/72),
                        clip=pymupdf.Rect(entry["crop_box_points"]), alpha=False)
                    assert np.array_equal(np.asarray(picture.convert("RGB")),
                                          np.frombuffer(pixmap.samples, dtype=np.uint8)
                                          .reshape(pixmap.height, pixmap.width, 3))
    print(captured.getvalue(), end="")
    print(json.dumps({"reader_programs": counts, "hessian_cases": hessian_cases,
                      "uda_gradient_cases": uda_cases, "original_tsa_cases": tsa_cases,
                      "source_files": len(manifest["sources"]), "original_figures": len(assets),
                      "hashes": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes())
                                 for p in (ROBUST, SEMI)}}, indent=2))
    print("LIMITS: CPU fixtures and exact sources only; no released model training, GPU timing or semantic-attack benchmark.")


if __name__ == "__main__":
    main()

