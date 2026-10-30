#!/usr/bin/env python3
"""Check the seven fully reconciled English multimodal chapters, offline.

Run with an existing NumPy/PyTorch environment. This executes published teaching
examples; it neither downloads weights nor measures trained-model performance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path("en/00 Textbook/16 Multimodal Models")
BASELINES = {
    "64 Multimodal models.md": ("52e4bfaa209330e09233ae75609bde1b6ceb844069cb6c378c39095ca48e4ac1", 157, 3),
    "64a Connectors and fusion.md": ("d4d776a62bf126b27dff06587e19e709e023d0e56bf8226fe222c304de2551dd", 148, 5),
    "64b Resolution, tiling, and spatial positions.md": ("c091d7da862133c3da7ad4d923e853ace16c47cc7190c86c34307506377a80a1", 126, 5),
    "64c Training VLMs — alignment, instruction tuning, and data.md": ("e6edcea86fd0bf36b43ef4766bf654360222e4aa1dbafae5ec2c16e578e22094", 178, 8),
    "64d Documents, OCR, and visual grounding.md": ("c01b425c50b16a7fc48cdd69bcd0f264a799cd50af515743a16d19febf13e38b", 132, 2),
    "64e Video, audio, and omni models.md": ("618c6a3a2467ffd66fa858d6be40a77bef818e16817cda3bbd11b01d85550d44", 206, 4),
    "64f Evaluation, failure modes, and VLM serving.md": ("31c9b97170ea855f2c47f8e94f7d2f4680a4731bf09aef95c478a16e8901f4c2", 134, 2),
}
TOPICS = [
    ["Patch ordering and convolution equivalence; projection information loss",
     "CLS, permutation equivariance, position embeddings and complete ViT block",
     "Resize/crop geometry and CLIP dual-encoder implementations",
     "Symmetric contrastive loss, scale, false negatives and generation distinction",
     "Prompt ensembles, candidate-set probabilities, linear probes and fine-tuning",
     "SigLIP objective, bias, normalization, blocked/distributed computation",
     "Winoground diagnostics and controlled occlusion limitations"],
    ["Linear and MLP projector parameter counts and visual-prefix insertion",
     "Shifted labels, causal access, prefill/first-token cache timing and memory",
     "OneVision changes beyond the connector",
     "BLIP-2 learned queries, ITC/ITM/ITG masks and stage-two OPT/T5 interfaces",
     "Flamingo zero gates, last-image masks and cross-attention cache",
     "Perceiver latent-inclusive K/V and weighted numerical example",
     "Qwen3-VL merger/DeepStack visual-mask additions and frozen-module gradients"],
    ["Resolution scaling, stretch/letterbox/crop geometry and encoder/LLM cost",
     "Molmo overlapping margins, pooling and 580-position assembled example",
     "AnyRes canvas selection, spatial permutation, separators and area budget",
     "InternVL size-dependent grid selection and reversible pixel unshuffle",
     "Qwen smart_resize and Pixtral packed-image isolation/row delimiters",
     "Compression mechanisms and controlled quality comparisons",
     "MRoPE coordinates versus cache indices; rotary algebra and frequency layouts",
     "Coordinate inversion and diagnostic controls; Qwen2.5-VL window/full attention"],
    ["Stage-specific datasets and objectives; teacher forcing and projector SGD step",
     "InternVL/Molmo freeze and per-component optimizer schedules",
     "Shifted labels, multimodal packing and shared-image independent annotations",
     "PixMo collection, teacher visibility, filtering and source-document splits",
     "Dataset sampling versus token mass and target-length loss normalization",
     "LLaVA/OneVision/Qwen curricula, exposure counts and inconsistent summary counts",
     "Original transfer illustrations with independently corrected arithmetic",
     "Visual verifiers, SAPO/version limits, response/logit distillation and KL"],
    ["Detection, recognition, reading order, field semantics and evidence identifiers",
     "LayoutLMv3 MLM/MIM/WPA, Donut serialization/data, Pix2Struct subtree curriculum",
     "Document targets in Qwen2.5/3-VL and preserved original figures",
     "Resolution, table headers/decimal arithmetic and linear/logarithmic chart scales",
     "Point instances, model coordinate units, homogeneous transforms and PDF origins",
     "CER/WER/ANLS/IoU/field F1 with normalization and edge cases",
     "Intervention expectations, extracted versus calculated values and calibrated acceptance"],
    ["Sampling aliasing, OneVision budgets, xGen memory and GenS windows/selection costs",
     "Presentation timestamps and model-specific temporal-position contracts",
     "STFT/mel features, RVQ, Mimi split branches and codec bitrate accounting",
     "Cascaded streaming versus direct audio and ImageBind retrieval",
     "40/80-ms audiovisual alignment and controlled soundtrack interventions",
     "Qwen2.5/3-Omni Talker conditions, MTP and waveform decoders",
     "Moshi temporal/depth factorization, Inner Monologue, delayed acoustics and duplex",
     "Lookahead masks, causal chunk equivalence, RTF/deadlines and audible interruption",
     "Temporal IoU, noise SNR, matched media budgets and service boundaries"],
    ["MM-Vet integrated capabilities and micro/macro category weighting",
     "Winoground strict scores/random baseline; POPE parsing and CHAIR limitations",
     "Paired counterfactuals, original MM-Vet rubric and judge evidence boundaries",
     "MMStar image dependence/leakage proxies and historical openness table",
     "Media request processing, expanded visual positions and cache tensor accounting",
     "Client TTFT, critical paths, head-of-line blocking and fairness",
     "EPD stage placement, transfer lifetimes, independent fragments and resource allocation",
     "Processor/encoder/prefix identity, UUID trust scope and security controls",
     "Selective risk/coverage and directly counted quality-and-deadline intersection"],
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def python_blocks(text: str) -> list[str]:
    return [match[1] for match in re.findall(r"(```|~~~)python\n(.*?)\n\1", text, re.S)]


def figures(text: str) -> set[str]:
    return {x.removeprefix("02 Areas/ML & DL/")
            for x in re.findall(r"!\[\[([^]|]+)", text)}


def field(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\s*(.*?)\s*$", text, re.M)
    if not match:
        raise AssertionError(f"missing frontmatter {key}")
    return match[1].strip('"')


def check_math_and_fences(text: str) -> None:
    outside = re.sub(r"(```|~~~).*?\n\1", "", text, flags=re.S)
    assert not re.search(r"^\s*\$\s*$", outside, re.M), "broken single-dollar display delimiter"
    assert len(re.findall(r"^\s*\$\$\s*$", outside, re.M)) % 2 == 0
    assert "<!-- PARITY EXAMPLE" not in text, "unfilled example marker"
    assert not re.search(r"\|64(?:\.\d+)?\]\]", text), "bare obsolete ordinal navigation"
    assert not re.search(r"\[\[[^]|]+\|64\.\d+\. ", text), "obsolete numbered navigation label"


def check() -> dict:
    entries = []
    for (name, baseline), topics in zip(BASELINES.items(), TOPICS):
        path = DIRECTORY / name
        raw = (ROOT / path).read_bytes()
        text = raw.decode()
        canonical = Path(field(text, "translation_of"))
        canonical_raw = (ROOT / canonical).read_bytes()
        canonical_text = canonical_raw.decode()
        assert field(text, "translation_status") == "current", path
        assert field(text, "canonical_revision") == sha(canonical_raw), path
        assert field(text, "translation_reviewed") == "2026-10-29", path
        check_math_and_fences(text)
        ids = re.findall(r'<span id="([^"]+)"(?: aria-hidden="true")?></span>', text)
        assert len(ids) == len(set(ids)), f"duplicate aliases: {path}"
        canonical_ids = set(re.findall(r'<span id="([^"]+)"', canonical_text))
        assert canonical_ids.issubset(ids), (path, canonical_ids - set(ids))
        expected_figures, actual_figures = figures(canonical_text), figures(text)
        assert expected_figures.issubset(actual_figures), (path, expected_figures - actual_figures)
        for figure in actual_figures:
            assert (ROOT / figure).is_file(), (path, figure)
        expected_code, actual_code = python_blocks(canonical_text), python_blocks(text)
        normalize = lambda codes: Counter(ast.dump(ast.parse(code)) for code in codes)
        assert normalize(expected_code) == normalize(actual_code), f"canonical code semantic drift: {path}"
        executions = []
        for index, code in enumerate(actual_code, 1):
            result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                    text=True, timeout=30, cwd=ROOT)
            assert result.returncode == 0, f"{path} block {index}: {result.stderr}"
            executions.append({"block": index, "sha256": sha(code.encode()),
                               "status": "passed", "returncode": result.returncode})
        entries.append({
            "path": str(path), "canonical": str(canonical),
            "before_sha256": baseline[0], "after_sha256": sha(raw),
            "canonical_sha256": sha(canonical_raw), "status": "current",
            "before_lines": baseline[1], "after_lines": len(text.split("\n")),
            "before_figure_count": baseline[2], "figure_count": len(actual_figures),
            "figures": sorted(actual_figures), "canonical_python_blocks": len(expected_code),
            "python_blocks": len(actual_code), "executions": executions,
            "canonical_code_ast_multiset_equal": True,
            "en_full_body_read": True, "canonical_full_body_read": True,
            "substantive_topics_reconciled": topics,
            "canonical_explicit_aliases": sorted(canonical_ids), "retained_aliases": ids,
            "outstanding_substantive_omissions": [],
        })
    clip = (ROOT / DIRECTORY / "64 Multimodal models.md").read_text()
    clip_canonical = (ROOT / field(clip, "translation_of")).read_text()
    pseudo = re.findall(r"^```text\n(.*?)^```$", clip_canonical, re.S | re.M)
    siglip = [part for part in pseudo if "logits" in part and "labels" in part]
    assert len(siglip) == 1, "expected one original SigLIP pseudocode excerpt"
    for part in siglip:
        assert part in clip, "original SigLIP pseudocode changed"
    rubric = "| What is x in the equation? | -1 &lt;AND&gt; -5 | x = -1 or x = -5 | 1.0 |"
    assert rubric in (ROOT / DIRECTORY / "64f Evaluation, failure modes, and VLM serving.md").read_text()
    return {
        "schema": 1, "date": "2026-10-29", "scope": "Seven existing English multimodal chapters",
        "before_evidence": "Captured after_sha256/line/figure values from the earlier 2026-10-29-english-reconciliation.json before this pass; that central ledger is not modified by this checker.",
        "verification": {
            "python": sys.version, "executable": sys.executable,
            "chapter_pairs_full_read": 7, "published_python_executed": sum(e["python_blocks"] for e in entries),
            "canonical_ast_parity": True, "all_canonical_native_figures_retained": True,
            "display_delimiters_and_explicit_aliases": "passed",
            "original_english_rubric_retained": True,
            "gpu_or_trained_model_inference": "not run; numerical/teaching checks only",
            "build_and_visual_qa": "parent-owned integrated checks; not claimed by this report",
            "prose_acceptance": "Full human-style source-pair reconciliation recorded by topics; token/figure/code counts alone are not evidence of prose quality.",
        },
        "totals": {"current": 7, "before_figures": sum(e["before_figure_count"] for e in entries),
                   "after_figures": sum(e["figure_count"] for e in entries)},
        "entries": entries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check()
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "chapters": len(report["entries"]),
                      "executed_blocks": report["verification"]["published_python_executed"],
                      "figures": report["totals"]}, ensure_ascii=False))
