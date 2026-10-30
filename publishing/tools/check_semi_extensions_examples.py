"""Check current reader programs and selected author-code semantics on CPU.

No downloaded model, translation generation, EfficientNet/BERT benchmark,
TensorFlow-1 runtime or historical full training is executed.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import re
from types import SimpleNamespace

import numpy as np
import pymupdf
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "05 Источники/Originals/semi-supervised-extensions-2026-10-29"
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/assets.json"
torch.set_num_threads(1)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 14
    captured = io.StringIO()
    for i, block in enumerate(blocks):
        with contextlib.redirect_stdout(captured):
            exec(compile(block, f"{CHAPTER}:python-{i+1}", "exec"), {})

    # Independent soft-target CE derivative: equality has positive entropy but zero gradient.
    generator = torch.Generator().manual_seed(41)
    soft_cases = 0
    for batch in (1, 3, 7):
        for classes in (2, 3, 5):
            raw = torch.randn(batch, classes, generator=generator, dtype=torch.float64)
            q = raw.softmax(-1).detach()
            logits = raw.clone().requires_grad_()
            ce = -(q * logits.log_softmax(-1)).sum(-1).mean()
            gradient, = torch.autograd.grad(ce, logits)
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)
            assert ce > 0
            torch.testing.assert_close(ce.detach(), -(q * q.log()).sum(-1).mean())
            soft_cases += 1

    # Execute only the read original loss class and ramp-up, not its GPU training loop.
    source = (SRC / "mixtext-train.py").read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ("linear_rampup", "SemiLoss")]
    assert len(nodes) == 2
    scope = {"np": np, "torch": torch, "F": F,
             "args": SimpleNamespace(epochs=50, mix_method=0, margin=.7,
                                     lambda_u=1., lambda_u_hinge=1.)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "author-SemiLoss", "exec"), scope)
    lx = torch.tensor([[.7, .3]], dtype=torch.float64).log()
    lu = torch.tensor([[.6, .4], [.9, .1]], dtype=torch.float64).log()
    target_x = torch.tensor([[.8, .2]], dtype=torch.float64)
    target_u = torch.tensor([[.7, .3], [.5, .5]], dtype=torch.float64)
    Lx, Lu, w, hinge, w2 = scope["SemiLoss"]()(lx, target_x, lu, target_u, lu, 5.)
    torch.testing.assert_close(Lx, -(target_x * lx.log_softmax(-1)).sum(-1).mean())
    expected_kl = (target_u * (target_u.log() - lu.log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(Lu, expected_kl)
    entropy = -(lu.softmax(-1) * lu.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(hinge, (entropy - .7).clamp_min(0).mean())
    assert w == w2 == .1
    loop = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    loss_if = next(n for n in ast.walk(loop) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "mix_ == 1"
                   and any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "loss"
                                                            for t in a.targets) for a in n.body))
    assembled = []
    for mixed in (1, 0):
        local = {"mix_": mixed, "Lx": 1., "w": 3., "Lu": 2., "w2": 5., "Lu2": 4.}
        exec(compile(ast.Module(body=[loss_if], type_ignores=[]), "author-loss-assembly", "exec"), local)
        assembled.append(local["loss"])
    assert assembled == [7., 27.]

    # Read author encoder class, replacing BERT layers with explicit arithmetic stubs.
    # This tests branch traversal and mixing placement, not BERT numerical equivalence.
    model_tree = ast.parse((SRC / "mixtext-model.py").read_text())
    encoder = next(n for n in model_tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "BertEncoder4Mix")
    class StubLayer(nn.Module):
        instances = 0
        calls = []
        def __init__(self, config):
            super().__init__()
            self.index = StubLayer.instances
            StubLayer.instances += 1
        def forward(self, hidden, attention_mask, head_mask):
            StubLayer.calls.append(self.index)
            return (hidden + self.index + 1,)
    model_scope = {"nn": nn, "torch": torch, "BertLayer": StubLayer}
    exec(compile(ast.Module(body=[encoder], type_ignores=[]), "author-encoder", "exec"), model_scope)
    encoder_cases = 0
    for layer in (-1, 0, 1, 2):
        StubLayer.instances, StubLayer.calls = 0, []
        config = SimpleNamespace(num_hidden_layers=3, output_attentions=False, output_hidden_states=False)
        model = model_scope["BertEncoder4Mix"](config)
        a = torch.ones(1, 2, 3, dtype=torch.float64, requires_grad=True)
        b = torch.full_like(a, 3., requires_grad=True)
        result = model(a, b, .75, layer, head_mask=[None]*3)[0]
        torch.testing.assert_close(result, torch.full_like(result, 7.5))
        ga, gb = torch.autograd.grad(result.sum(), (a, b))
        torch.testing.assert_close(ga, torch.full_like(a, .75))
        torch.testing.assert_close(gb, torch.full_like(b, .25))
        assert len(StubLayer.calls) == 3 + (layer + 1)
        encoder_cases += 1

    manifest = json.loads((SRC / "manifest.json").read_text())
    assert len(manifest["sources"]) == 13
    for row in manifest["sources"]:
        data = (SRC / row["file"]).read_bytes()
        assert len(data) == row["bytes"] and digest(data) == row["sha256"]
    rows = json.loads(FIGURES.read_text())
    assert len(rows) == 2
    for row in rows:
        data = (ROOT / row["asset"]).read_bytes()
        assert digest(data) == row["sha256"]
        assert digest((ROOT / row["source_asset"]).read_bytes()) == row["source_asset_sha256"]
        assert digest((ROOT / row["license_evidence"]).read_bytes()) == row["license_evidence_sha256"]
        with pymupdf.open(ROOT / row["source_asset"]) as doc:
            page = doc[row["source_page"]-1]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(240/72, 240/72),
                                 clip=pymupdf.Rect(row["crop_box_points"]), alpha=False)
        with Image.open(ROOT / row["asset"]) as image:
            assert list(image.size) == row["dimensions_px"]
            assert image.convert("RGB").tobytes() == pix.samples
    noisy = (SRC / "noisy-main.py").read_text()
    assert "model_is_training=False" in noisy and "tf.stop_gradient(teacher_logits)" in noisy
    assert "data_loss = lab_loss * real_lab_bsz + unl_loss * real_unl_bsz" in noisy
    assert "data_loss = data_loss / real_lab_bsz" in noisy
    print(captured.getvalue(), end="")
    print(json.dumps({"chapter_sha256": digest(CHAPTER.read_bytes()), "reader_programs": 14,
                      "soft_ce_cases": soft_cases, "author_loss_branches": assembled,
                      "author_encoder_stub_cases": encoder_cases, "source_files": 13,
                      "original_figures": 2}, indent=2))
    print("LIMITS: CPU examples; author encoder uses arithmetic layer stubs; no full model, GPU or benchmark reproduction.")


if __name__ == "__main__":
    main()
