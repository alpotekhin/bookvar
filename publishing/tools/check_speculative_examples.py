#!/usr/bin/env python3
"""Check the actual chapter-58 CPU examples and original-source provenance.

This does not load an LLM, execute EAGLE, or benchmark GPU serving.
"""
from __future__ import annotations
import hashlib
import itertools
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
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md"
SOURCES = ROOT / "05 Источники/Originals/speculative-2026-10-19"
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/speculative-review-2026-10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return " ".join(value.split())


def check_sources(chapter):
    manifest = json.loads((SOURCES / "excerpts.json").read_text())
    for source in manifest["sources"]:
        assert digest(SOURCES / source["file"]) == source["sha256"], source["file"]
    for item in manifest["excerpts"]:
        source = (SOURCES / item["source"]).read_text()
        selected = source[source.index(item["start"]):source.index(
            item["end_exclusive"], source.index(item["start"]))].strip()
        displayed = re.search(
            rf'<!-- source-excerpt: {item["id"]}:start -->(.*?)'
            rf'<!-- source-excerpt: {item["id"]}:end -->', chapter, re.S
        )
        assert displayed, item["id"]
        quote = "\n".join(re.sub("^> ?", "", line) for line in displayed[1].splitlines())
        assert norm(quote) == norm(selected), (item["id"], norm(selected), norm(quote))
    return {"source_hashes": len(manifest["sources"]), "exact_excerpts": len(manifest["excerpts"])}


def check_figures(chapter):
    assets = json.loads((ASSETS / "assets.json").read_text())["assets"]
    rerenders = 0
    for item in assets:
        output = ROOT / item["asset"]
        assert digest(output) == item["sha256"]
        assert struct.unpack(">II", output.read_bytes()[16:24]) == (item["width"], item["height"])
        if "source_archive" in item:
            archive = ROOT / item["source_archive"]
            assert digest(archive) == item["source_archive_sha256"]
            with tarfile.open(archive) as package:
                assert package.extractfile(item["source_member"]).read() == (
                    ROOT / item["source_asset"]).read_bytes()
        if item["derivation"] != "pdf-page-render-crop":
            continue
        pdf = ROOT / item["source_asset"]
        assert digest(pdf) == item["source_asset_sha256"]
        if shutil.which("pdftoppm"):
            page = str(item["source_page"])
            x0, y0, x1, y1 = item["crop_box_px"]
            with tempfile.TemporaryDirectory() as directory:
                prefix = Path(directory) / "image"
                subprocess.run([
                    "pdftoppm", "-f", page, "-l", page, "-singlefile",
                    "-scale-to", str(max(item["render_page_size_px"])),
                    "-x", str(x0), "-y", str(y0), "-W", str(x1-x0), "-H", str(y1-y0),
                    "-png", str(pdf), str(prefix)
                ], check=True, capture_output=True)
                assert output.read_bytes() == prefix.with_suffix(".png").read_bytes(), item["id"]
            rerenders += 1
    embeds = re.findall(r"!\[\[([^\]]+)\]\]", chapter)
    assert len(embeds) == 8
    for embed in embeds:
        assert (ROOT / embed.removeprefix("02 Areas/ML & DL/")).is_file(), embed
    assert "figures/speculative_sampling_results_2.pdf" in json.dumps(assets)
    assert "includegraphics[width=\\columnwidth]{speculative_sampling_results_2}" in (
        SOURCES / "chen-main.tex").read_text()
    return {"figures": len(embeds), "new_assets": len(assets), "pdf_rerenders": rerenders}


def sampling_probabilities(prefix, kind):
    """A tiny prefix-dependent categorical model, not independent token rows."""
    code = sum((i + 3) * (x + 1) for i, x in enumerate(prefix))
    offsets = np.array([.2, .8, 1.5]) if kind == "target" else np.array([1.3, .4, .9])
    scores = offsets + .65 * np.sin(code + np.arange(3) * (1.2 if kind == "target" else .8))
    weight = np.exp(scores - scores.max())
    return weight / weight.sum()


def exact_cycle_test(verify, k, horizon=4):
    """Enumerate all proposals, acceptance branches and correction outcomes.

    Invoke the displayed verifier for every branch with representative uniforms;
    complete the tail with the target model, then compare the full sequence law.
    """
    actual = {}
    def add_tail(prefix, probability):
        if len(prefix) == horizon:
            actual[prefix] = actual.get(prefix, 0.) + probability
            return
        for token, p in enumerate(sampling_probabilities(prefix, "target")):
            add_tail(prefix + (token,), probability * p)
    for candidates in itertools.product(range(3), repeat=k):
        p_rows, q_rows, draft_probability = [], [], 1.
        for i, token in enumerate(candidates):
            p = sampling_probabilities(candidates[:i], "proposal")
            q = sampling_probabilities(candidates[:i], "target")
            p_rows.append(p)
            q_rows.append(q)
            draft_probability *= p[token]
        q_rows.append(sampling_probabilities(candidates, "target"))
        ratios = [min(1., q_rows[i][x] / p_rows[i][x]) for i, x in enumerate(candidates)]
        prefix_mass = 1.
        for accepted in range(k + 1):
            if accepted < k:
                branch_mass = prefix_mass * (1 - ratios[accepted])
                residual = np.maximum(q_rows[accepted] - p_rows[accepted], 0)
                if branch_mass < 1e-16:
                    prefix_mass *= ratios[accepted]
                    continue
                residual /= residual.sum()
                uniforms = [ratios[i]/2 for i in range(accepted)] + [
                    (1+ratios[accepted])/2] + [0.] * (k-accepted-1)
                expected_reason = "reject"
            else:
                branch_mass = prefix_mass
                residual = q_rows[-1]
                uniforms = [a/2 for a in ratios]
                expected_reason = "all-accepted"
            if branch_mass:
                cdf = np.cumsum(residual)
                for token, prob in enumerate(residual):
                    if prob == 0:
                        continue
                    u = cdf[token] - prob / 2
                    out, count, reason = verify(candidates, p_rows, q_rows, uniforms, u)
                    assert count == accepted and reason == expected_reason
                    assert out == list(candidates[:accepted]) + [token]
                    add_tail(tuple(out), draft_probability * branch_mass * prob)
            if accepted < k:
                prefix_mass *= ratios[accepted]
    expected = {}
    for seq in itertools.product(range(3), repeat=horizon):
        probability = 1.
        for i, token in enumerate(seq):
            probability *= sampling_probabilities(seq[:i], "target")[token]
        expected[seq] = probability
    assert set(actual) == set(expected)
    for seq in expected:
        np.testing.assert_allclose(actual[seq], expected[seq], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sum(actual.values()), 1)
    return len(expected)


def main():
    chapter = CHAPTER.read_text()
    fence = chr(96)*3
    blocks = re.findall(
        rf"<!-- checked-example: ([\w-]+) -->\s*{fence}python\n(.*?)\n{fence}", chapter, re.S)
    assert len(blocks) == 7
    ns = {"__name__": "bookvar_speculative_examples"}
    for ident, code in blocks:
        exec(compile(code, f"{CHAPTER}:{ident}", "exec"), ns)
    rng = np.random.default_rng(581019)
    # Categorical accepted mass, including zero proposal support and p=q.
    distributions = [(np.array([1., 0., 0.]), np.array([0., .4, .6])),
                     (np.array([.2, .3, .5]), np.array([.2, .3, .5]))]
    distributions += [(rng.dirichlet(np.ones(11)), rng.dirichlet(np.ones(11))) for _ in range(500)]
    for p, q in distributions:
        accepted = np.minimum(p, q)
        rejected = 1 - accepted.sum()
        residual = np.maximum(q-p, 0)
        np.testing.assert_allclose(rejected, residual.sum(), atol=1e-14)
        np.testing.assert_allclose(accepted + residual, q, atol=1e-14)
        np.testing.assert_allclose(accepted.sum(), 1 - np.abs(p-q).sum()/2)
    # Actual Markdown verifier versus exact, prefix-dependent target sequence law.
    sequence_checks = sum(exact_cycle_test(ns["verify_chain"], k) for k in [1, 2, 3])
    # The fixed-logit greedy proof, with deterministic argmax tie handling.
    for _ in range(500):
        k = int(rng.integers(1, 9))
        candidates = rng.integers(0, 4, size=k)
        rows = rng.random((k+1, 4))
        expected = []
        for i, x in enumerate(candidates):
            chosen = int(np.argmax(rows[i]))
            expected.append(chosen)
            if chosen != x:
                break
        else:
            expected.append(int(np.argmax(rows[-1])))
        assert ns["greedy_verify"](candidates, rows) == expected
    # Token selection versus KV materialization, for every rejection depth.
    for n in range(1, 20):
        for k in range(1, 10):
            confirmed = list(range(n))
            draft = list(range(n, n+k))
            temporary = confirmed[:-1] + [confirmed[-1]] + draft
            for accepted in range(k+1):
                text = confirmed + draft[:accepted] + [-99]
                retained = temporary[:n+accepted]
                assert retained == text[:-1]
    # Expected progress versus the exact capped-geometric probability mass.
    for k in range(1, 15):
        for alpha in np.linspace(0, 1, 31):
            expected = sum((a+1)*(alpha**a)*(1-alpha) for a in range(k)) + (k+1)*alpha**k
            np.testing.assert_allclose(ns["expected_progress"](alpha, k), expected, atol=1e-13)
    np.testing.assert_allclose((2.5+13)/4.2, 3.6904761904761907)
    np.testing.assert_allclose(10/((2.5+13)/4.2), 2.7096774193548385)
    # Tree-masked attention equals independent ancestor-path attention.
    mask = ns["tree_mask"]
    for _ in range(300):
        scores, values = rng.normal(size=(6, 6)), rng.normal(size=(6, 5))
        weights = np.exp(np.where(mask, scores, -np.inf) - np.max(scores, axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        packed = weights @ values
        for row in range(6):
            indices = np.flatnonzero(mask[row])
            weight = np.exp(scores[row, indices] - scores[row, indices].max())
            weight /= weight.sum()
            np.testing.assert_allclose(packed[row], weight @ values[indices], atol=1e-13)
    # Reranking: monotone path products + shallower-first ties preserve ancestors.
    for _ in range(300):
        parents = [-1] + [int(rng.integers(i)) for i in range(1, 20)]
        values, depth = [1.], [0]
        for i in range(1, 20):
            values.append(values[parents[i]] * rng.choice([.2, .5, 1.]))
            depth.append(depth[parents[i]] + 1)
        for budget in range(1, 21):
            selected = set(sorted(range(20), key=lambda i: (-values[i], depth[i], i))[:budget])
            assert all(parents[i] == -1 or parents[i] in selected for i in selected)
    # The second sibling uses a new proposal and the current residual distribution.
    q, p1, p2 = np.array([.2, .5, .3]), np.array([.6, .3, .1]), np.array([0., .75, .25])
    residual1 = np.maximum(q-p1, 0)
    residual1 /= residual1.sum()
    np.testing.assert_allclose(residual1, [0, .5, .5])
    np.testing.assert_allclose(residual1[1]/p2[1], 2/3)
    residual2 = np.maximum(residual1-p2, 0)
    residual2 /= residual2.sum()
    np.testing.assert_allclose(residual2, [0, 0, 1])
    for source_unit in [
        "lecture-10-section-341-use-shortcuts-but-double-check-lossless",
        "lecture-10-figure-step-353-rendering-1", "lecture-10-figure-step-366-rendering-1",
        "lecture-10-figure-step-367-rendering-1"]:
        assert source_unit in chapter
    assert 'id="процент-принятия-связывает-качество-и-скорость"' in chapter
    assert 'id="worked-trade-off"' in chapter
    print(json.dumps({
        "displayed_python_blocks": len(blocks), "exact_sequence_probabilities": sequence_checks,
        "probability_mass_trials": len(distributions), "greedy_trials": 500,
        "tree_attention_trials": 300, "reranking_trees": 300,
        **check_sources(chapter), **check_figures(chapter)
    }, indent=2))


if __name__ == "__main__":
    main()

