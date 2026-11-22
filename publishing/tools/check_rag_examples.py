#!/usr/bin/env python3
"""Chapter 63 executable mechanics and source crops; not an LLM benchmark."""
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md"
SOURCES = ROOT / "05 Источники/Originals/rag-2026-10-20"
FIGURES = ROOT / "00 Учебник/Assets/Figures/curated/rag-review-2026-10"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S)
    assert len(blocks) == 7
    spaces, shared = [], {}
    for i, block in enumerate(blocks, 1):
        ns = {}
        code = compile(block, f"{CHAPTER}:block-{i}", "exec")
        exec(code, ns)
        exec(code, shared)
        spaces.append(ns)

    rag = spaces[0]["rag_probabilities"]
    rng = np.random.default_rng(20261020)
    for _ in range(40):
        weights = rng.dirichlet(np.ones(3)).tolist()
        probabilities = rng.uniform(.01, .99, (3, 3))
        total = np.zeros(2)
        for answer in itertools.product([0, 1], repeat=3):
            rows = [[float(p if a else 1-p) for p, a in zip(row, answer)] for row in probabilities]
            total += rag(weights, rows)
        assert np.allclose(total, [1, 1])
    logits = np.log([.6, .4])
    likelihoods = np.array([.1, .8])
    posterior = np.array([.6, .4]) * likelihoods / .38
    for i in range(2):
        delta = np.eye(2)[i] * 1e-5
        def loss(s):
            weights = np.exp(s) / np.exp(s).sum()
            return -np.log(weights @ likelihoods)
        numeric = (loss(logits + delta) - loss(logits - delta)) / 2e-5
        assert np.isclose(numeric, [.6, .4][i] - posterior[i], atol=1e-9)

    allow, rows = spaces[1]["eligible"], spaces[1]["rows"]
    assert allow(rows[0], "2026-08-31", {"employee"})
    assert not allow(rows[0], "2026-09-01", {"employee"})
    assert allow(rows[1], "2026-09-01", {"employee"})
    assert not allow(rows[2], "2026-10-20", {"employee"})

    graph = spaces[2]
    exact = np.linalg.solve(np.eye(4) - .8 * graph["P"].T, .2 * graph["s"])
    assert np.allclose(exact, graph["r"], atol=1e-12)
    assert np.allclose(graph["passage_scores"], [9/14, 29/54, 25/54])
    pack, candidates = spaces[3]["pack_context"], spaces[3]["candidates"]
    for order in itertools.permutations(candidates):
        for budget in [0, 2600, 2800, 3000, 3100, 5700, 6144]:
            chosen, used, covered = pack(order, budget)
            chosen_rows = [r for r in order if r["id"] in chosen]
            assert used == sum(r["tokens"] for r in chosen_rows) <= budget
            assert covered == set().union(*(set(r["spans"]) for r in chosen_rows))

    score, entails = spaces[4]["citation_scores"], spaces[4]["toy_entails"]
    options = [list(c) for n in range(4) for c in itertools.combinations(["S1", "S2", "noise"], n)]
    for combo in itertools.product(options, repeat=3):
        flags = [bool(c) and entails(i, frozenset(c)) for i, c in enumerate(combo)]
        useful = sum(flags[i] and (entails(i, frozenset([c]))
                     or not entails(i, frozenset(set(ids) - {c})))
                     for i, ids in enumerate(combo) for c in ids)
        n = sum(map(len, combo))
        assert score(combo, entails) == (sum(flags)/3, useful/n if n else 0)
    adaptive = spaces[5]
    assert np.isclose(adaptive["retrieve"], .60)
    assert np.isclose(adaptive["a"], .75) and np.isclose(adaptive["b"], .30)
    assert (.50 + .5 * adaptive["a"]) < (.80 + .5 * adaptive["b"])
    assert (.50 + adaptive["a"]) > (.80 + adaptive["b"])

    coverage = spaces[6]["evidence_coverage"]
    required = {"p4-exception", "p2-cleanup"}
    assert coverage({"p4-exception", "noise"}, required) == (.5, 0)
    by_id = {row["id"]: row for row in candidates}
    source_spans = {"exception": "p4-exception", "backup": "p2-cleanup"}
    def coverage_for(chosen):
        spans = set().union(*(by_id[key]["spans"] for key in chosen))
        return coverage({source_spans[s] for s in spans if s in source_spans}, required)

    plain_budget, plain = 6144, []
    for row in candidates:
        if row["tokens"] <= plain_budget:
            plain.append(row["id"])
            plain_budget -= row["tokens"]
    assert plain_budget == 244
    expected = [coverage_for(plain), coverage_for(pack(candidates, 6144)[0]),
                coverage_for(pack(candidates, 3000)[0])]
    assert expected == [(.5, 0), (1, 1), (.5, 0)]

    assets = json.loads((FIGURES / "assets.json").read_text())
    assert len(assets) == 5
    for row in assets:
        for key, hash_key in [("asset", "sha256"), ("source_asset", "source_asset_sha256"),
            ("source_page_snapshot", "source_page_sha256"), ("license_evidence", "license_evidence_sha256")]:
            assert digest(ROOT / row[key]) == row[hash_key]
        page = Image.open(ROOT / row["source_page_snapshot"]).convert("RGB")
        target = Image.open(ROOT / row["asset"]).convert("RGB")
        assert list(page.size) == row["render_page_size_px"]
        assert ImageChops.difference(page.crop(row["crop_box_px"]), target).getbbox() is None
        assert row["asset"] in chapter
    for old, frozen in [("hf-rag-evaluation-workflow.png", "hf-flow.png"),
                        ("hf-rag-settings-accuracy.png", "hf-accuracy.png")]:
        assert digest(ROOT / "00 Учебник/Assets/Figures/curated/source-first-60-64" / old) == digest(SOURCES / frozen)
    for source in json.loads((SOURCES / "references.json").read_text())["downloaded"]:
        assert digest(ROOT / source["path"]) == source["sha256"]
    for block in re.findall(r"^```json\n(.*?)^```", chapter, re.M | re.S):
        json.loads(block)
    assert len(re.findall(r"!\[\[", chapter)) == 10
    for locale in ("", "en/"):
        page = ROOT / f"site/dist/{locale}textbook/rag/full-pipeline/index.html"
        soup = BeautifulSoup(page.read_text(), "html.parser")
        article = soup.select_one(".sl-markdown-content")
        assert article is not None and len(article.select("img")) == 10
        assert len(article.select('details[id^="rag-code-"]')) == 9
        assert len(article.select(".katex-display")) == 5 and not article.select(".katex-error")
        figures = [img["src"] for img in article.select("img")]
        assert "hf-rag-evaluation-workflow.png" in figures[0]
        assert "rag-sequence-token.png" in figures[1]
        for anchor, term in (("rag-source-policy", "30 дней"), ("rag-source-backup", "7 суток")):
            proof = article.select_one(f"span#{anchor}")
            assert proof is not None and term in proof.find_parent("tr").get_text()
            citations = article.select(f'blockquote a[href$="#{anchor}"]')
            assert len(citations) == 2 and citations[0].get_text() == citations[1].get_text()
        example_table = article.select_one("#rag-example-coverage").find_next("table")
        assert [r.select("td")[-1].get_text(strip=True) for r in example_table.select("tbody tr")] == ["1/2", "2/2", "1/2"]
        assert article.select_one(f'a[href="/{locale}textbook/retrieval/reranking/#rerank-e42-case"]')
        assert article.select_one(f'a[href="/{locale}textbook/multimodal/models/"]')
        assert not locale or soup.select_one('[aria-label="English edition status"]')
    print(json.dumps({"chapter_sha256": digest(CHAPTER),
        "code_blocks_standalone_and_shared": 14, "probability_normalizations": 40,
        "posterior_gradients": 2, "context_order_budget_cases": 42,
        "citation_subset_cases": 512, "figure_pixel_crops": 5,
        "unchanged_hf_images": 2, "adaptive_retrieval_checks": 4,
        "context_selection_controls": expected, "rendered_locales": 2,
        "ppr": graph["r"].tolist(),
        "llm_quality_run": False}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

