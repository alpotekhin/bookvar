#!/usr/bin/env python3
"""Import licensed mlsysbook.ai chapters without rewriting their prose.

The public HTML is used instead of raw QMD because it contains evaluated
variables, resolved cross-references, final tables, and the exact figures seen
by readers.  The local repository commit remains the provenance pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import MarkdownConverter


# The public website reported Vol I v0.7.1 (2026-07-05) and Vol II v0.2.1
# (2026-07-03) when this importer was authored.  Its July 5 deployment matches
# this `main` merge commit.  Do not replace it with the newer `dev` SHA unless
# the Quarto output is rebuilt from that same revision.
PINNED_COMMIT = "2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd"
LICENSE = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
REPOSITORY = "https://github.com/harvard-edge/cs249r_book"


@dataclass(frozen=True)
class Chapter:
    volume: str
    slug: str
    title: str

    @property
    def url(self) -> str:
        return f"https://mlsysbook.ai/{self.volume}/{self.slug}/{self.slug}.html"

    @property
    def source_path(self) -> str:
        return f"book/quarto/contents/{self.volume}/{self.slug}/{self.slug}.qmd"


CHAPTERS = [
    Chapter("vol1", "ml_systems", "ML Systems"),
    Chapter("vol1", "ml_workflow", "ML Workflow"),
    Chapter("vol1", "data_engineering", "Data Engineering"),
    Chapter("vol1", "nn_computation", "Neural Computation"),
    Chapter("vol1", "nn_architectures", "Network Architectures"),
    Chapter("vol1", "frameworks", "ML Frameworks"),
    Chapter("vol1", "training", "Model Training"),
    Chapter("vol1", "data_selection", "Data Selection"),
    Chapter("vol1", "model_compression", "Model Compression"),
    Chapter("vol1", "hw_acceleration", "Hardware Acceleration"),
    Chapter("vol1", "benchmarking", "Benchmarking"),
    Chapter("vol1", "model_serving", "Model Serving"),
    Chapter("vol1", "ml_ops", "ML Operations"),
    Chapter("vol1", "responsible_engr", "Responsible Engineering"),
    Chapter("vol2", "compute_infrastructure", "Compute Infrastructure"),
    Chapter("vol2", "network_fabrics", "Network Fabrics"),
    Chapter("vol2", "data_storage", "Data Storage"),
    Chapter("vol2", "distributed_training", "Distributed Training"),
    Chapter("vol2", "collective_communication", "Collective Communication"),
    Chapter("vol2", "fault_tolerance", "Fault Tolerance"),
    Chapter("vol2", "fleet_orchestration", "Fleet Orchestration"),
    Chapter("vol2", "performance_engineering", "Performance Engineering"),
    Chapter("vol2", "inference", "Inference at Scale"),
    Chapter("vol2", "edge_intelligence", "Edge Intelligence"),
    Chapter("vol2", "ops_scale", "ML Operations at Scale"),
    Chapter("vol2", "security_privacy", "Security & Privacy"),
    Chapter("vol2", "robust_ai", "Robust AI"),
    Chapter("vol2", "sustainable_ai", "Sustainable AI"),
    Chapter("vol2", "responsible_ai", "Responsible AI"),
]


class SourceMarkdownConverter(MarkdownConverter):
    """Keep source wording while producing Markdown Starlight can render."""

    def convert_div(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        classes = set(el.get("class", []))
        callout = next((item for item in classes if item.startswith("callout-")), None)
        if callout:
            kind = callout.removeprefix("callout-").upper()
            return f"\n\n> **{kind}**\n>\n" + "\n".join(
                f"> {line}" if line else ">" for line in text.strip().splitlines()
            ) + "\n\n"
        return text

    def convert_img(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        source = str(el.get("src", ""))
        if source.startswith("Assets/Sources/"):
            return f"![[{source}]]"
        alt = str(el.get("alt", "")).replace("]", r"\]")
        return f"![{alt}]({source})"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Bookvar source importer (+https://github.com/alpotekhin/bookvar)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def normalize_math(value: str) -> str:
    """Replace Unicode typography that the Bookvar KaTeX gate rejects."""
    replacements = {
        "\u00a0": " ",
        "×": r"\times ",
        "÷": r"\div ",
        "≈": r"\approx ",
        "≠": r"\ne ",
        "≤": r"\le ",
        "≥": r"\ge ",
        "±": r"\pm ",
        "−": "-",
        "∞": r"\infty ",
        "→": r"\to ",
        "←": r"\leftarrow ",
        "–": "--",
        "—": "---",
        "‘": "`",
        "’": "'",
        "“": "``",
        "”": "''",
        "·": r"\cdot ",
        "\u2003": " ",
        "ü": r'\"u',
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    # Some rendered spans contain nested Markdown-style "$...$" delimiters.
    # KaTeX is already in math mode here, so keep escaped currency (`\$`) and
    # discard only unescaped delimiter characters.
    value = re.sub(r"(?<!\\)\$", "", value)
    value = value.replace(r"\begin{array}{>{\displaystyle}c}", r"\begin{array}{c}")
    display_outer_text = re.fullmatch(
        r"\\\[\s*\\text\{((?:\\text\{|\\underbrace\{).*)\}\s*\\\]",
        value,
        flags=re.DOTALL,
    )
    inline_outer_text = re.fullmatch(
        r"\\\(\s*\\text\{((?:\\text\{|\\underbrace\{).*)\}\s*\\\)",
        value,
        flags=re.DOTALL,
    )
    outer_text = display_outer_text or inline_outer_text
    if outer_text and re.search(
        r"\\(?:frac|underbrace|begin|sum|prod|int|left|right)\b",
        outer_text.group(1),
    ):
        delimiter = (r"\[", r"\]") if display_outer_text else (r"\(", r"\)")
        value = f"{delimiter[0]}{outer_text.group(1)}{delimiter[1]}"
    remaining = sorted({character for character in value if ord(character) > 127})
    if remaining:
        labels = ", ".join(f"U+{ord(character):04X}" for character in remaining)
        raise ValueError(f"Unsupported non-ASCII character(s) in source math: {labels}")
    return value


def escape_prose_dollars(content: Tag) -> None:
    """Prevent currency amounts from becoming accidental remark-math spans."""
    for node in list(content.find_all(string=True)):
        if not isinstance(node, NavigableString) or "$" not in node:
            continue
        parents = list(node.parents)
        if any(
            parent.name in {"code", "pre"}
            or (parent.name == "span" and "math" in set(parent.get("class", [])))
            for parent in parents
            if isinstance(parent, Tag)
        ):
            continue
        node.replace_with(str(node).replace("$", r"\$"))


def protect_math(content: Tag) -> dict[str, dict[str, object]]:
    """Replace rendered math spans with collision-resistant plain tokens."""
    tokens: dict[str, dict[str, object]] = {}
    for index, node in enumerate(list(content.select("span.math"))):
        raw = normalize_math(node.get_text("", strip=False).strip())
        token = f"BOOKVARMATH{index:06d}TOKEN"
        if raw.startswith(r"\(") and raw.endswith(r"\)"):
            tokens[token] = {"value": raw[2:-2], "displayMode": False}
        elif raw.startswith(r"\[") and raw.endswith(r"\]"):
            tokens[token] = {"value": raw[2:-2], "displayMode": True}
        else:
            raise ValueError(f"Unknown rendered math delimiter: {raw[:80]}")
        node.replace_with(NavigableString(token))
    return tokens


def clean_document(
    html: bytes, chapter: Chapter
) -> tuple[BeautifulSoup, Tag, dict[str, dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#quarto-document-content")
    if content is None:
        raise RuntimeError(f"No #quarto-document-content in {chapter.url}")

    for selector in [
        "script",
        "style",
        "nav",
        ".quarto-title-block",
        ".quarto-secondary-nav",
        ".chapter-summary",
        ".socratiq-widget",
        ".lightbox-gallery",
        "[data-socratiq]",
    ]:
        for node in content.select(selector):
            node.decompose()

    for anchor in content.select("a[href]"):
        anchor["href"] = urllib.parse.urljoin(chapter.url, anchor["href"])
    for anchor in content.select("a.lightbox"):
        anchor.unwrap()
    for image in content.select("img[src]"):
        image["src"] = urllib.parse.urljoin(chapter.url, image["src"])
    math_tokens = protect_math(content)
    escape_prose_dollars(content)
    return soup, content, math_tokens


def localize_images(
    content: Tag,
    chapter: Chapter,
    asset_root: Path,
    *,
    download: bool,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    chapter_root = asset_root / chapter.volume / chapter.slug
    chapter_root.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    for index, image in enumerate(content.select("img[src]"), start=1):
        source_url = str(image["src"])
        parsed = urllib.parse.urlparse(source_url)
        raw_name = Path(parsed.path).name or f"figure-{index}.bin"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name)
        if safe_name in used_names:
            safe_name = f"{index:03d}-{safe_name}"
        used_names.add(safe_name)
        destination = chapter_root / safe_name
        if download and not destination.exists():
            destination.write_bytes(fetch(source_url))
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        else:
            digest = ""
        image["src"] = (
            f"Assets/Sources/Harvard ML Systems/"
            f"{chapter.volume}/{chapter.slug}/{safe_name}"
        )
        records.append(
            {
                "file": destination.relative_to(asset_root.parent.parent).as_posix(),
                "source_url": source_url,
                "sha256": digest,
                "alt": image.get("alt", ""),
            }
        )
    return records


def render_math_tokens(
    math_tokens: dict[str, dict[str, object]], root: Path
) -> dict[str, str]:
    helper = root / "publishing" / "tools" / "render-source-math.mjs"
    process = subprocess.run(
        ["node", str(helper)],
        input=json.dumps(list(math_tokens.values())),
        text=True,
        capture_output=True,
        check=True,
        cwd=root,
    )
    rendered = json.loads(process.stdout)
    return {
        token: (
            f"\n\n<div class=\"source-math-display\">{html}</div>\n\n"
            if bool(specification["displayMode"])
            else f"<span class=\"source-math-inline\">{html}</span>"
        )
        for (token, specification), html in zip(math_tokens.items(), rendered, strict=True)
    }


def markdown_for(
    content: Tag,
    chapter: Chapter,
    math_tokens: dict[str, dict[str, object]],
    root: Path,
) -> str:
    body = SourceMarkdownConverter(
        heading_style="ATX",
        bullets="-",
        strip=["button", "svg"],
    ).convert_soup(content)
    # Currency in attributes can bypass the NavigableString pass.  All real
    # source equations are protected above, so a remaining "$" before a digit
    # is prose currency and must not open a remark-math node.
    body = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", body)
    for token, math in render_math_tokens(math_tokens, root).items():
        body = body.replace(token, math)
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip()
    body = re.sub(r"^#\s+.+?(?:\r?\n)+", "", body, count=1)
    source_blob = f"{REPOSITORY}/blob/{PINNED_COMMIT}/{chapter.source_path}"
    return f"""---
title: "{chapter.title}"
type: external-resource
status: imported-source
language: en
source_kind: verbatim-adaptation
source_commit: {PINNED_COMMIT}
---

> [!note] Original course chapter
> This page contains the complete English chapter
> [{chapter.title}]({chapter.url}) from *Machine Learning Systems* by Vijay
> Janapa Reddi and contributors. Source:
> [{chapter.source_path}]({source_blob}) at commit `{PINNED_COMMIT}`.
> License: [{LICENSE}]({LICENSE_URL}). Bookvar changed only the publication
> markup, internal links, and local image paths; the wording and structure were
> retained.

{body}
"""


def import_chapter(
    chapter: Chapter,
    output_root: Path,
    asset_root: Path,
    cache_root: Path,
    *,
    refresh: bool,
    download_images: bool,
) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{chapter.volume}-{chapter.slug}.html"
    if refresh or not cache_path.exists():
        cache_path.write_bytes(fetch(chapter.url))
    html = cache_path.read_bytes()
    _soup, content, math_tokens = clean_document(html, chapter)
    images = localize_images(content, chapter, asset_root, download=download_images)
    output = output_root / chapter.volume / f"{chapter.slug}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_for(content, chapter, math_tokens, output_root.parents[2]), encoding="utf-8")
    return {
        "volume": chapter.volume,
        "slug": chapter.slug,
        "title": chapter.title,
        "url": chapter.url,
        "source_path": chapter.source_path,
        "rendered_snapshot_sha256": hashlib.sha256(html).hexdigest(),
        "output": output.as_posix(),
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chapter", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    selected = CHAPTERS if args.all else [
        chapter for chapter in CHAPTERS if chapter.slug in set(args.chapter)
    ]
    if not selected:
        parser.error("Select --all or at least one --chapter SLUG")

    root = args.root.resolve()
    output_root = root / "05 Источники" / "Courses" / "Harvard ML Systems"
    asset_root = root / "Assets" / "Sources" / "Harvard ML Systems"
    cache_root = root / ".cache" / "mlsysbook"
    records = [
        import_chapter(
            chapter,
            output_root,
            asset_root,
            cache_root,
            refresh=args.refresh,
            download_images=not args.skip_images,
        )
        for chapter in selected
    ]
    manifest = output_root / "import-manifest.json"
    existing = {}
    if manifest.exists():
        existing = {
            (item["volume"], item["slug"]): item
            for item in json.loads(manifest.read_text(encoding="utf-8"))
        }
    for record in records:
        existing[(record["volume"], record["slug"])] = record
    manifest.write_text(
        json.dumps(
            sorted(existing.values(), key=lambda item: (item["volume"], item["slug"])),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Imported {len(records)} chapter(s); manifest: {manifest}")


if __name__ == "__main__":
    main()
