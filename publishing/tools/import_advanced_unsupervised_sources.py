#!/usr/bin/env python3
"""Publish the complete advanced-unsupervised source bundle as readable Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

from html.parser import HTMLParser


FILES = {
    "mlcourse-topic7.html": {
        "title": "mlcourse.ai — Topic 7: PCA and Clustering",
        "url": "https://mlcourse.ai/book/topic07/topic7_pca_clustering.html",
        "repository": "https://github.com/Yorko/mlcourse.ai",
        "commit": "f966c2c9dfdc684e3cf4d338b4af08a75ca8ed1b",
        "license": "CC BY-NC-SA 4.0",
        "kind": "html",
    },
    "distill-misread-tsne.html": {
        "title": "How to Use t-SNE Effectively",
        "url": "https://distill.pub/2016/misread-tsne/",
        "repository": "https://github.com/distillpub/post--misread-tsne",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0",
        "kind": "html",
    },
    "umap-parameters.rst": {
        "title": "UMAP Parameters",
        "url": "https://github.com/lmcinnes/umap/blob/e82ed0d457b566b043ef44f4007a7149b0daca74/doc/parameters.rst",
        "repository": "https://github.com/lmcinnes/umap",
        "commit": "e82ed0d457b566b043ef44f4007a7149b0daca74",
        "license": "BSD-3-Clause",
        "kind": "rst",
    },
    "sklearn-pca-iris.py": {
        "title": "PCA example: Iris",
        "url": "https://github.com/scikit-learn/scikit-learn/blob/b9530aa1fa5b00d933a8f258a094befea41aa86c/examples/decomposition/plot_pca_iris.py",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "commit": "b9530aa1fa5b00d933a8f258a094befea41aa86c",
        "license": "BSD-3-Clause",
        "kind": "python",
    },
    "sklearn-tsne-perplexity.py": {
        "title": "t-SNE perplexity example",
        "url": "https://github.com/scikit-learn/scikit-learn/blob/b9530aa1fa5b00d933a8f258a094befea41aa86c/examples/manifold/plot_t_sne_perplexity.py",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "commit": "b9530aa1fa5b00d933a8f258a094befea41aa86c",
        "license": "BSD-3-Clause",
        "kind": "python",
    },
    "sklearn-gmm-covariances.py": {
        "title": "GMM covariance types example",
        "url": "https://github.com/scikit-learn/scikit-learn/blob/b9530aa1fa5b00d933a8f258a094befea41aa86c/examples/mixture/plot_gmm_covariances.py",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "commit": "b9530aa1fa5b00d933a8f258a094befea41aa86c",
        "license": "BSD-3-Clause",
        "kind": "python",
    },
    "sklearn-kernel-approximation.py": {
        "title": "RBF kernel approximation example",
        "url": "https://github.com/scikit-learn/scikit-learn/blob/b9530aa1fa5b00d933a8f258a094befea41aa86c/examples/miscellaneous/plot_kernel_approximation.py",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "commit": "b9530aa1fa5b00d933a8f258a094befea41aa86c",
        "license": "BSD-3-Clause",
        "kind": "python",
    },
}


class ReadableHTML(HTMLParser):
    """Extract the actual article rather than browser chrome from saved HTML."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.out: list[str] = []
        self.in_article = False
        self.article_depth = 0
        self.skip = 0
        self.links: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        is_article = tag == "dt-article" or (
            tag == "article" and "bd-article" in (attr.get("class") or "").split()
        )
        if not self.in_article and is_article:
            self.in_article = True
            self.article_depth = 1
            return
        if not self.in_article:
            return
        self.article_depth += 1
        if tag in {"script", "style", "noscript", "svg", "button"}:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "pre", "blockquote", "hr"}:
            self.out.append("\n")
        if tag in {"h1", "h2", "h3", "h4"}:
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("- ")
        elif tag == "pre":
            self.out.append("```\n")
        elif tag == "code":
            self.out.append("`")
        elif tag in {"strong", "b"}:
            self.out.append("**")
        elif tag in {"em", "i"}:
            self.out.append("*")
        elif tag == "a":
            href = attr.get("href")
            # Sphinx adds a visible "#" to every heading; it is browser chrome.
            if "headerlink" in (attr.get("class") or "").split():
                self.links.append(None)
                self.skip += 1
            else:
                self.links.append(urljoin(self.base_url, href) if href else "")
                self.out.append("[")
        elif tag == "img" and attr.get("src"):
            src = urljoin(self.base_url, attr["src"])
            alt = (attr.get("alt") or "Source figure").strip()
            self.out.append(f"\n![{alt}]({src})\n")

    def handle_endtag(self, tag: str) -> None:
        if not self.in_article:
            return
        if tag in {"script", "style", "noscript", "svg", "button"} and self.skip:
            self.skip -= 1
        elif tag == "a" and self.links:
            href = self.links.pop()
            if href is None and self.skip:
                self.skip -= 1
            elif not self.skip:
                self.out.append(f"]({href})" if href else "]")
        elif not self.skip and tag == "pre":
            self.out.append("\n```\n")
        elif not self.skip and tag == "code":
            self.out.append("`")
        elif not self.skip and tag in {"strong", "b"}:
            self.out.append("**")
        elif not self.skip and tag in {"em", "i"}:
            self.out.append("*")
        elif not self.skip and tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "blockquote"}:
            self.out.append("\n")
        self.article_depth -= 1
        if self.article_depth == 0:
            self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.in_article and not self.skip:
            self.out.append(data)

def html_to_markdown(raw: str, base_url: str) -> str:
    parser = ReadableHTML(base_url)
    parser.feed(raw)
    text = "".join(parser.out).replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Some article link labels already include square brackets (for example
    # "[PDF]"). Avoid turning them into Obsidian wiki links as "[[PDF]](...)".
    text = re.sub(r"\[\[([^\[\]]+)\]\]\(([^)]+)\)", r"[\1](\2)", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    src = root / "raw/courses/advanced-unsupervised-classical"
    dst = root / "05 Источники/Courses/Advanced Unsupervised and Classical ML"
    dst.mkdir(parents=True, exist_ok=True)
    records = []
    links = []
    for name, meta in FILES.items():
        title = meta["title"]
        url = meta["url"]
        license_name = meta["license"]
        kind = meta["kind"]
        commit = meta["commit"]
        path = src / name
        raw = path.read_text(encoding="utf-8")
        body = html_to_markdown(raw, url) if kind == "html" else f"```{kind}\n{raw.rstrip()}\n```"
        output = dst / f"{name}.md"
        output.write_text(f"""---
title: "{title}"
type: external-resource
status: imported-source
language: en
source_kind: {kind}
---

> [!note] Complete original source
> Imported reproducibly from [`{name}`]({url}). License: {license_name}.
> Upstream repository: [{meta["repository"]}]({meta["repository"]}); pinned
> revision: [`{commit}`]({meta["repository"]}/commit/{commit}).
> Bookvar removes browser chrome and executable scripts from saved HTML while
> preserving the article body. Code and RST are byte-for-byte inside the fence.

{body}
""", encoding="utf-8")
        links.append(f"- [[05 Источники/Courses/Advanced Unsupervised and Classical ML/{name}.md|{title}]]")
        records.append({
            "source": name,
            "output": output.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": url,
            "repository": meta["repository"],
            "upstream_commit": commit,
            "license": license_name,
        })
    (dst / "import-manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "05 Источники/Courses/Advanced Unsupervised and Classical ML.md").write_text("""---
title: Advanced Unsupervised and Classical ML — original sources
type: source-note
status: imported-source
last_verified: 2026-07-26
---

# Advanced Unsupervised and Classical ML — original sources

Seven complete English source artifacts behind the PCA, t-SNE, UMAP, GMM/EM,
kernel, SVM, and random Fourier feature chapters. Regenerate with
`python3 publishing/tools/import_advanced_unsupervised_sources.py --root .`.

Every item records the SHA-256 of the retained local artifact and a pinned
upstream repository revision. The four scikit-learn scripts and the UMAP RST
were additionally verified byte-for-byte against their pinned Git revisions.

""" + "\n".join(links) + "\n", encoding="utf-8")
    print(f"Imported {len(records)} complete sources")


if __name__ == "__main__":
    main()
