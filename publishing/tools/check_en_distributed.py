#!/usr/bin/env python3
"""Check distributed-serving EN parity offline; execute CPU examples in chapter order."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path("en/00 Textbook/14 Inference and optimization")
FENCE = chr(96) * 3
BASELINES = {
    "58a Distributed inference and disaggregated serving.md":
        ("81091952aa4d93f369b484ac83591b19c9165755d2a147b79667da789b50cdc1", 346, 5, 8),
    "58a2 Disaggregated prefill and decode serving.md":
        ("1824786b27b084d16c5dde66ee37cf540e2f3629ff19d704b3df9f2e0731b812", 440, 4, 10),
}
CANONICAL_CORRECTION = {
    "path": "00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md",
    "before_sha256": "b7b7f12fe4726130c9d5d75697e7cda4d3314216fbb3a8195a474af77cb5efe4",
    "after_sha256": "7973703ca7ec77248f0fdc77321e7c5f9805164d182508234fb5c49c3e86a13d",
    "old": "8192 активных и 4096 зарезервированных новый запрос на 4096 позиций уже не",
    "new": "8192 активных и 4096 зарезервированных новый запрос на 8192 позиций уже не",
    "reason": "16384 - 8192 - 4096 = 4096: incoming 4096 fits exactly, whereas 8192 does not. Existing <= code unchanged.",
}
TOPICS = [
    [
        "Weight placement, group-local rank, full versus partial tensor semantics",
        "AllGather/ReduceScatter/AllReduce/AllToAll values and ring latency calculation",
        "Independent dense DP, prefix routing, and attention DP distinction",
        "TP MLP numerical example, nonlinear placement, single bias/residual addition",
        "SwiGLU alignment, attention-head TP, GQA limits, HBM versus collectives",
        "PP schedule, bubble, unequal stages, autoregressive dependency, layer KV",
        "Token-row SP normalization and AG/RS; CP global masks and KV exchange",
        "Stable partial-attention merge with denominator weighting and masked partitions",
        "MoE top-2 dispatch/return, ordering, router weights, padding and network bytes",
        "Physical paths and hybrid group numbering; EP axes not freely multiplicative",
        "70B TP4/8/16 capacity arithmetic with GQA replication and stated limits",
        "KV request-axis sharding, query AllToAll and restored output equality",
    ],
    [
        "Prompt KV versus sampled uncached token; continuation state and timestamp boundaries",
        "Prefill/decode intensity, MLA formula boundary, matched HBM accounting",
        "Unified scheduling, chunk arithmetic, synchronized MoE interference",
        "Separate and mixed pools with full-resource accounting",
        "KV bytes, conditional suffix reuse, TP2-to-TP1 axis restoration",
        "Push/pull ownership, cancellation and late DMA writes, backend readiness",
        "Splitwise original excerpt/figure; dependency-correct layer-transfer tail",
        "P-first versus D-ready TTFT/TTST and first ITL example",
        "Prefix-chain identity and cache/queue/network routing calculation",
        "Reservation versus forecast admission, future KV growth and overload",
        "Versioned SGLang allocation constraints; queue-aware provisioning and 18 GPUs",
        "M/D/1 TP/PP crossover and assumptions; nine Splitwise percentile constraints",
        "DistServe failure boundary, Splitwise historical hardware costs, Mooncake CPP",
        "Causal two-layer chunk equivalence with stage-local KV",
        "Dynamo v1.5.1 metadata, D prefix deduction limits, failure policy",
        "Joint per-request SLO goodput and complete-pool cost denominator",
    ],
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field(text, key):
    return re.search(rf"^{key}:\s*(.*?)\s*$", text, re.M)[1].strip('"')


def blocks(text):
    return [m[1] for m in re.findall(rf"({FENCE}|~~~)python\n(.*?)\n\1", text, re.S)]


def prose(text):
    return re.sub(rf"({FENCE}|~~~)[^\n]*\n.*?\n\1", "", text, flags=re.S)


def headings(text):
    return re.findall(r"^#{1,3} (.+)$", prose(text), re.M)


def slug(text):
    # publishing/adapter/links.ts wikiHeadingSlug normalization.
    return "".join(c for c in re.sub(r"\s+", "-", text.lower())
                   if c.isalnum() or c in "_-")


def figures(text):
    return {p.removeprefix("02 Areas/ML & DL/")
            for p in re.findall(r"!\[\[([^]|]+)", text)}


def check():
    fix = CANONICAL_CORRECTION
    corrected = (ROOT / fix["path"]).read_bytes()
    assert sha(corrected) == fix["after_sha256"]
    restored = corrected.decode().replace(fix["new"], fix["old"], 1).encode()
    assert sha(restored) == fix["before_sha256"], "canonical scope exceeded one-line correction"
    entries = []
    for (name, baseline), topics in zip(BASELINES.items(), TOPICS):
        path = DIRECTORY / name
        raw = (ROOT / path).read_bytes()
        text = raw.decode()
        canonical = Path(field(text, "translation_of"))
        canonical_raw = (ROOT / canonical).read_bytes()
        canonical_text = canonical_raw.decode()
        assert field(text, "translation_status") == "current"
        assert field(text, "canonical_revision") == sha(canonical_raw)
        assert field(text, "translation_reviewed") == "2026-10-29"
        outside = prose(text)
        assert not re.search(r"^\s*\$\s*$", outside, re.M), "single-dollar display delimiter"
        assert len(re.findall(r"^\s*\$\$\s*$", outside, re.M)) % 2 == 0
        assert "<!-- EXAMPLE:" not in text and "<!-- SOURCE-EXCERPT -->" not in text
        ids = re.findall(r'<(?:a|span) id="([^"]+)"', text)
        assert len(ids) == len(set(ids)), "duplicate fragment alias"
        canonical_ids = set(re.findall(r'<(?:a|span) id="([^"]+)"', canonical_text))
        expected_headings, actual_headings = headings(canonical_text), headings(text)
        assert len(expected_headings) == len(actual_headings)
        expected_aliases = {slug(h) for h in expected_headings} | canonical_ids
        assert expected_aliases.issubset(ids), expected_aliases - set(ids)
        expected_figures, actual_figures = figures(canonical_text), figures(text)
        assert expected_figures.issubset(actual_figures)
        for figure in actual_figures:
            assert (ROOT / figure).is_file(), figure
        expected_code, actual_code = blocks(canonical_text), blocks(text)
        assert len(actual_code) == baseline[3]
        assert [ast.dump(ast.parse(c)) for c in expected_code] == [
            ast.dump(ast.parse(c)) for c in actual_code], "canonical example AST changed"
        runner = (
            "import json\nscope = {'__name__': '__chapter_examples__'}\n"
            f"codes = {actual_code!r}\n"
            "for index, code in enumerate(codes, 1):\n"
            "    exec(compile(code, f'<published block {index}>', 'exec'), scope)\n"
            "    print(json.dumps({'block': index, 'status': 'passed'}))\n"
        )
        result = subprocess.run([sys.executable, "-c", runner], text=True,
                                capture_output=True, timeout=30, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        executed = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(executed) == len(actual_code)
        excerpts = re.findall(
            r"<!-- source-excerpt: ([^ ]+):start -->.*?<!-- source-excerpt: \1:end -->",
            canonical_text, re.S)
        for name_ in excerpts:
            excerpt = re.search(
                rf"<!-- source-excerpt: {re.escape(name_)}:start -->.*?"
                rf"<!-- source-excerpt: {re.escape(name_)}:end -->",
                canonical_text, re.S)[0]
            assert excerpt in text, "approved original English excerpt altered"
        entries.append({
            "path": str(path), "canonical": str(canonical),
            "before_sha256": baseline[0], "after_sha256": sha(raw),
            "canonical_sha256": sha(canonical_raw), "status": "current",
            "before_lines": baseline[1], "after_lines": len(text.splitlines()),
            "before_figure_count": baseline[2], "figure_count": len(actual_figures),
            "canonical_figure_count": len(expected_figures), "figures": sorted(actual_figures),
            "python_blocks": len(actual_code), "canonical_code_ast_sequence_equal": True,
            "executions": [dict(e, sha256=sha(actual_code[e["block"]-1].encode()))
                           for e in executed],
            "execution_mode": "chapter-ordered blocks sharing a namespace in one fresh CPU subprocess",
            "canonical_explicit_aliases": sorted(canonical_ids),
            "heading_map": [{"canonical_heading": r, "english_heading": e,
                             "canonical_fragment": slug(r)}
                            for r, e in zip(expected_headings, actual_headings)],
            "retained_aliases": ids, "original_english_excerpts": excerpts,
            "canonical_full_body_read": True, "en_full_body_read": True,
            "substantive_topics_reconciled": topics,
            "outstanding_substantive_omissions": [],
        })
    return {
        "schema": 1, "date": "2026-10-29",
        "scope": "Two English distributed-inference chapters plus one authorized canonical arithmetic correction",
        "before_evidence": "Four-file hashes and complete text captured before this pass.",
        "canonical_correction": fix,
        "verification": {
            "python": sys.version, "executable": sys.executable,
            "chapter_pairs_full_read": 2,
            "published_python_executed": sum(e["python_blocks"] for e in entries),
            "canonical_ast_parity": True, "all_canonical_native_figures_retained": True,
            "canonical_heading_aliases": sum(len(e["heading_map"]) for e in entries),
            "display_delimiters_and_fragment_aliases": "passed",
            "original_english_excerpt_retained": True,
            "gpu_or_model_inference": "not run; CPU mathematical and scheduling examples only",
            "build_and_visual_qa": "parent-owned integrated checks; not claimed by this report",
            "editorial_review": "Full source-pair reading and mechanism/example/caveat reconciliation; counts are not a prose-quality score.",
        },
        "totals": {"current": 2, "before_figures": 9,
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
    print(json.dumps({"status": "passed", "chapters": 2,
                      "executed_blocks": report["verification"]["published_python_executed"],
                      "figures": report["totals"]}))

