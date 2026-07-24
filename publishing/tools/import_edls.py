#!/usr/bin/env python3
"""Create a source-native mirror of Efficient Deep Learning Systems.

The importer does not summarize or translate.  It preserves notebook Markdown,
code, saved outputs, README files, and page-separated PDF text.  Original PDFs
remain embedded from a commit-pinned GitHub URL so readers keep the slide
layout and figures.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import subprocess
from pathlib import Path, PurePosixPath

import nbformat


COMMIT = "e632aa89ca9e6638d52e1b686095e7442faffbb0"
REPOSITORY = "https://github.com/mryab/efficient-dl-systems"
LICENSE = "MIT"
LICENSE_URL = f"{REPOSITORY}/blob/{COMMIT}/LICENSE"


def protect_wiki_delimiters(text: str) -> str:
    """Keep array/tensor output from being parsed as an Obsidian wiki link."""
    return text.replace("[[", "[\u200b[").replace("]]", "]\u200b]")


def normalize_math_unicode(markdown: str) -> str:
    """Replace Unicode operators only inside TeX delimiters."""

    def normalize(match: re.Match[str]) -> str:
        body = match.group(2).replace("⫫", r"\perp")
        return f"{match.group(1)}{body}{match.group(1)}"

    return re.sub(r"(\${1,2})(.+?)\1", normalize, markdown, flags=re.S)


def frontmatter(title: str, source_path: str, kind: str) -> str:
    source_url = f"{REPOSITORY}/blob/{COMMIT}/{source_path}"
    return f"""---
title: "{title.replace('"', '\\"')}"
type: external-resource
status: imported-source
language: original
source_kind: {kind}
source_commit: {COMMIT}
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`{source_path}`]({source_url}) in *Efficient Deep Learning Systems* at commit
> `{COMMIT}`. License: [{LICENSE}]({LICENSE_URL}). Bookvar changed only the
> publication markup and link paths.

"""


def pinned_url(source_path: str, target: str, *, raw: bool) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I) or target.startswith("#"):
        return target
    source_dir = PurePosixPath(source_path).parent
    resolved = (source_dir / target).as_posix()
    while resolved.startswith("../"):
        resolved = resolved[3:]
    endpoint = "raw" if raw else "blob"
    return f"https://github.com/mryab/efficient-dl-systems/{endpoint}/{COMMIT}/{resolved}"


def rewrite_relative_links(markdown: str, source_path: str) -> str:
    def image(match: re.Match[str]) -> str:
        return f"![{match.group(1)}]({pinned_url(source_path, match.group(2), raw=True)})"

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        return f"[{label}]({pinned_url(source_path, target, raw=False)})"

    markdown = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", image, markdown)
    markdown = re.sub(
        r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])',
        lambda match: (
            match.group(1)
            + pinned_url(source_path, match.group(2), raw=True)
            + match.group(3)
        ),
        markdown,
        flags=re.I,
    )
    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", link, markdown)


def notebook_title(path: Path, notebook: nbformat.NotebookNode) -> str:
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        match = re.search(r"^#\s+(.+)$", cell.source, flags=re.M)
        if match:
            return match.group(1).strip()
    return path.stem.replace("_", " ").title()


def output_markdown(
    output: nbformat.NotebookNode,
    asset_root: Path,
    source_path: str,
    cell_index: int,
    output_index: int,
) -> str:
    output_type = output.get("output_type")
    if output_type == "stream":
        text = protect_wiki_delimiters(str(output.get("text", "")).rstrip())
        return f"\n```text\n{text}\n```\n" if text else ""
    if output_type == "error":
        traceback = protect_wiki_delimiters("\n".join(output.get("traceback", [])))
        return f"\n```text\n{traceback}\n```\n" if traceback else ""
    data = output.get("data", {})
    blocks: list[str] = []
    for mime in ["text/markdown", "text/plain"]:
        if mime in data:
            value = data[mime]
            text = "".join(value) if isinstance(value, list) else str(value)
            if mime == "text/markdown":
                blocks.append(rewrite_relative_links(text, source_path))
            else:
                blocks.append(
                    f"```text\n{protect_wiki_delimiters(text.rstrip())}\n```"
                )
            break
    for mime, extension in [
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("image/svg+xml", ".svg"),
    ]:
        if mime not in data:
            continue
        filename = f"cell-{cell_index:03d}-output-{output_index:02d}{extension}"
        destination = asset_root / filename
        value = data[mime]
        if mime == "image/svg+xml":
            destination.write_text(
                "".join(value) if isinstance(value, list) else str(value),
                encoding="utf-8",
            )
        else:
            encoded = "".join(value) if isinstance(value, list) else str(value)
            destination.write_bytes(base64.b64decode(encoded))
        assets_at = destination.parts.index("Assets")
        public_path = Path(*destination.parts[assets_at:]).as_posix()
        blocks.append(f"![[{public_path}]]")
    return "\n\n".join(blocks)


def import_notebook(source_root: Path, root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    notebook = nbformat.read(path, as_version=4)
    output = root / "05 Источники" / "Courses" / "Efficient DL Systems" / f"{relative}.md"
    asset_root = (
        root
        / "Assets"
        / "Sources"
        / "Efficient DL Systems"
        / relative.removesuffix(".ipynb")
    )
    asset_root.mkdir(parents=True, exist_ok=True)
    title = notebook_title(path, notebook)
    chunks = [frontmatter(title, relative, "notebook")]
    removed_title_heading = False
    for cell_index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "markdown":
            text = normalize_math_unicode(
                rewrite_relative_links(str(cell.source), relative)
            ).strip()
            if not removed_title_heading:
                heading = re.match(r"^#\s+(.+?)(?:\r?\n|$)", text)
                if heading and heading.group(1).strip() == title:
                    text = text[heading.end() :].lstrip()
                    removed_title_heading = True
            if text:
                chunks.append(text)
        elif cell.cell_type == "code":
            language = notebook.metadata.get("kernelspec", {}).get("language", "python")
            source = protect_wiki_delimiters(str(cell.source).rstrip())
            chunks.append(f"```{language}\n{source}\n```")
            for output_index, saved in enumerate(cell.get("outputs", []), start=1):
                rendered = output_markdown(
                    saved, asset_root, relative, cell_index, output_index
                )
                if rendered:
                    chunks.append(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return {
        "source_path": relative,
        "kind": "notebook",
        "output": output.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cells": len(notebook.cells),
        "saved_outputs": sum(len(cell.get("outputs", [])) for cell in notebook.cells),
    }


def import_readme(source_root: Path, root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else relative
    body = rewrite_relative_links(raw, relative)
    body = re.sub(r"^#\s+.+?(?:\r?\n)+", "", body, count=1)
    output = root / "05 Источники" / "Courses" / "Efficient DL Systems" / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        frontmatter(title, relative, "readme") + body.strip() + "\n",
        encoding="utf-8",
    )
    return {
        "source_path": relative,
        "kind": "readme",
        "output": output.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.M)
    if not match:
        raise RuntimeError(f"Cannot determine page count: {path}")
    return int(match.group(1))


def import_pdf(source_root: Path, root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    pages = pdf_page_count(path)
    title = f"{path.parent.name.replace('_', ' ').title()} — lecture slides"
    source_url = f"{REPOSITORY}/blob/{COMMIT}/{relative}?raw=1"
    chunks = [
        frontmatter(title, relative, "slides"),
        "The embedded PDF is the primary visual version. The page-separated text "
        "below is included for search and quotation; it was extracted mechanically "
        "and has not been rewritten.",
        f'<iframe class="source-pdf" src="{source_url}" title="{title}" '
        'loading="lazy"></iframe>',
        "## Extracted slide text",
    ]
    for page in range(1, pages + 1):
        result = subprocess.run(
            [
                "pdftotext",
                "-layout",
                "-f",
                str(page),
                "-l",
                str(page),
                str(path),
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout.replace("\f", "").rstrip()
        chunks.append(f"### Page {page}\n\n```text\n{text}\n```")
    output = (
        root
        / "05 Источники"
        / "Courses"
        / "Efficient DL Systems"
        / f"{relative}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return {
        "source_path": relative,
        "kind": "slides",
        "output": output.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    records: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*.ipynb")):
        records.append(import_notebook(source_root, root, path))
    for path in sorted(source_root.rglob("README.md")):
        records.append(import_readme(source_root, root, path))
    for path in sorted(source_root.rglob("*.pdf")):
        records.append(import_pdf(source_root, root, path))
    output_root = root / "05 Источники" / "Courses" / "Efficient DL Systems"
    manifest = output_root / "import-manifest.json"
    manifest.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Imported {len(records)} source files: "
        f"{sum(item['kind'] == 'notebook' for item in records)} notebooks, "
        f"{sum(item['kind'] == 'readme' for item in records)} READMEs, "
        f"{sum(item['kind'] == 'slides' for item in records)} slide decks"
    )


if __name__ == "__main__":
    main()
