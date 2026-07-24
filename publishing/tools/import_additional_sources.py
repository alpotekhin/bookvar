#!/usr/bin/env python3
"""Import source-native course collections without translating or summarizing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import nbformat


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    repository: str
    commit: str
    license_name: str
    license_path: str
    output_name: str

    @property
    def license_url(self) -> str:
        return f"{self.repository}/blob/{self.commit}/{self.license_path}"


SOURCES = {
    "cs230": Source(
        key="cs230",
        title="Stanford CS230 Deep Learning Cheatsheets",
        repository="https://github.com/afshinea/stanford-cs-230-deep-learning",
        commit="4653bc01297b269edb19e844b01127ba13de59df",
        license_name="MIT",
        license_path="LICENSE",
        output_name="Stanford CS230 Cheatsheets",
    ),
    "ml-visualized": Source(
        key="ml-visualized",
        title="Machine Learning Visualized",
        repository="https://github.com/gavinkhung/machine-learning-visualized",
        commit="950f8ec4671c0790cdd867514cc114fea01f5ab0",
        license_name="MIT",
        license_path="LICENSE",
        output_name="Machine Learning Visualized",
    ),
    "genai-agents": Source(
        key="genai-agents",
        title="GenAI Agents",
        repository="https://github.com/NirDiamant/GenAI_Agents",
        commit="bd681451b254ac1a790e947b581d3997ab35013d",
        license_name="Custom non-commercial license",
        license_path="LICENSE",
        output_name="GenAI Agents",
    ),
}


def protect_wiki_delimiters(text: str) -> str:
    return text.replace("[[", "[\u200b[").replace("]]", "]\u200b]")


def normalize_math_unicode(markdown: str) -> str:
    replacements = {
        "⫫": r"\perp",
        "≤": r"\le",
        "≥": r"\ge",
        "∑": r"\sum",
        "∏": r"\prod",
        "∞": r"\infty",
        "∈": r"\in",
        "∉": r"\notin",
        "→": r"\to",
    }

    def normalize(match: re.Match[str]) -> str:
        body = match.group(2)
        for old, new in replacements.items():
            body = body.replace(old, new)
        return f"{match.group(1)}{body}{match.group(1)}"

    return re.sub(r"(\${1,2})(.+?)\1", normalize, markdown, flags=re.S)


def pinned_url(source: Source, source_path: str, target: str, *, raw: bool) -> str:
    if (
        re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I)
        or target.startswith("#")
        or target.startswith("/")
    ):
        return target
    resolved = (PurePosixPath(source_path).parent / target).as_posix()
    while resolved.startswith("../"):
        resolved = resolved[3:]
    endpoint = "raw" if raw else "blob"
    return f"{source.repository}/{endpoint}/{source.commit}/{resolved}"


def rewrite_links(markdown: str, source: Source, source_path: str) -> str:
    def image(match: re.Match[str]) -> str:
        target = pinned_url(source, source_path, match.group(2), raw=True)
        return f"![{match.group(1)}]({target})"

    def link(match: re.Match[str]) -> str:
        target = pinned_url(source, source_path, match.group(2), raw=False)
        return f"[{match.group(1)}]({target})"

    markdown = re.sub(
        r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", image, markdown
    )
    markdown = re.sub(
        r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])',
        lambda match: (
            match.group(1)
            + pinned_url(source, source_path, match.group(2), raw=True)
            + match.group(3)
        ),
        markdown,
        flags=re.I,
    )
    return re.sub(
        r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", link, markdown
    )


def frontmatter(source: Source, title: str, source_path: str, kind: str) -> str:
    source_url = f"{source.repository}/blob/{source.commit}/{source_path}"
    safe_title = title.replace('"', '\\"')
    return f"""---
title: "{safe_title}"
type: external-resource
status: imported-source
language: original
source_kind: {kind}
source_commit: {source.commit}
---

> [!note] Original source material
> This page preserves [`{source_path}`]({source_url}) from
> *{source.title}* at commit `{source.commit}`. License:
> [{source.license_name}]({source.license_url}). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.

"""


def output_path(root: Path, source: Source, relative: str) -> Path:
    return root / "05 Источники" / "Courses" / source.output_name / relative


def asset_path(root: Path, source: Source, relative: str) -> Path:
    return root / "Assets" / "Sources" / source.output_name / relative


def notebook_title(path: Path, notebook: nbformat.NotebookNode) -> str:
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        match = re.search(r"^#\s+(.+)$", str(cell.source), flags=re.M)
        if match:
            return re.sub(r"<[^>]+>", "", match.group(1)).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def render_output(
    saved: nbformat.NotebookNode,
    assets: Path,
    root: Path,
    cell_index: int,
    output_index: int,
) -> str:
    output_type = saved.get("output_type")
    if output_type == "stream":
        text = protect_wiki_delimiters(str(saved.get("text", "")).rstrip())
        return f"```text\n{text}\n```" if text else ""
    if output_type == "error":
        text = protect_wiki_delimiters("\n".join(saved.get("traceback", [])))
        return f"```text\n{text}\n```" if text else ""
    data = saved.get("data", {})
    blocks: list[str] = []
    for mime in ("text/markdown", "text/plain"):
        if mime not in data:
            continue
        value = data[mime]
        text = "".join(value) if isinstance(value, list) else str(value)
        if mime == "text/markdown":
            blocks.append(normalize_math_unicode(text))
        else:
            blocks.append(f"```text\n{protect_wiki_delimiters(text.rstrip())}\n```")
        break
    for mime, suffix in (
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("image/svg+xml", ".svg"),
    ):
        if mime not in data:
            continue
        destination = assets / f"cell-{cell_index:03d}-output-{output_index:02d}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = data[mime]
        encoded = "".join(value) if isinstance(value, list) else str(value)
        if mime == "image/svg+xml":
            destination.write_text(encoded, encoding="utf-8")
        else:
            destination.write_bytes(base64.b64decode(encoded))
        blocks.append(f"![[{destination.relative_to(root).as_posix()}]]")
    return "\n\n".join(blocks)


def import_notebook(
    root: Path, source_root: Path, source: Source, path: Path
) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    notebook = nbformat.read(path, as_version=4)
    title = notebook_title(path, notebook)
    chunks = [frontmatter(source, title, relative, "notebook")]
    assets = asset_path(root, source, relative.removesuffix(".ipynb"))
    removed_title = False
    for cell_index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "markdown":
            text = normalize_math_unicode(
                rewrite_links(str(cell.source), source, relative)
            ).strip()
            if not removed_title:
                heading = re.match(r"^#\s+(.+?)(?:\r?\n|$)", text)
                if heading and re.sub(r"<[^>]+>", "", heading.group(1)).strip() == title:
                    text = text[heading.end() :].lstrip()
                    removed_title = True
            if text:
                chunks.append(text)
        elif cell.cell_type == "code":
            language = str(
                notebook.metadata.get("kernelspec", {}).get("language", "python")
            )
            if not re.fullmatch(r"[A-Za-z0-9_+-]{1,24}", language):
                language = "python"
            code = protect_wiki_delimiters(str(cell.source).rstrip())
            chunks.append(f"```{language}\n{code}\n```")
            for output_index, saved in enumerate(cell.get("outputs", []), start=1):
                rendered = render_output(
                    saved, assets, root, cell_index, output_index
                )
                if rendered:
                    chunks.append(rendered)
    destination = output_path(root, source, f"{relative}.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return {
        "source_path": relative,
        "kind": "notebook",
        "output": destination.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cells": len(notebook.cells),
        "saved_outputs": sum(len(cell.get("outputs", [])) for cell in notebook.cells),
    }


def import_markdown(
    root: Path, source_root: Path, source: Source, path: Path
) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    title = (
        re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
        if title_match
        else path.stem.replace("_", " ").title()
    )
    body = rewrite_links(raw, source, relative)
    body = re.sub(r"^#\s+.+?(?:\r?\n)+", "", body, count=1)
    destination = output_path(root, source, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        frontmatter(source, title, relative, "markdown")
        + normalize_math_unicode(body).strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_path": relative,
        "kind": "markdown",
        "output": destination.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.M)
    if not match:
        raise RuntimeError(f"Cannot determine page count: {path}")
    return int(match.group(1))


def import_pdf(
    root: Path,
    source_root: Path,
    source: Source,
    path: Path,
    title: str,
) -> dict[str, object]:
    relative = path.relative_to(source_root).as_posix()
    pages = pdf_pages(path)
    source_url = f"{source.repository}/blob/{source.commit}/{relative}?raw=1"
    chunks = [
        frontmatter(source, title, relative, "slides"),
        "The embedded PDF is the primary visual document. The page-separated "
        "text below was extracted mechanically for search and has not been rewritten.",
        f'<iframe class="source-pdf" src="{source_url}" title="{title}" '
        'loading="lazy"></iframe>',
        "## Extracted document text",
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
        text = protect_wiki_delimiters(result.stdout.replace("\f", "").rstrip())
        chunks.append(f"### Page {page}\n\n```text\n{text}\n```")
    destination = output_path(root, source, f"{relative}.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return {
        "source_path": relative,
        "kind": "slides",
        "output": destination.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pages": pages,
    }


def import_collection(root: Path, source_root: Path, source: Source) -> None:
    records: list[dict[str, object]] = []
    if source.key == "cs230":
        records.append(import_markdown(root, source_root, source, source_root / "README.md"))
        titles = {
            "cheatsheet-convolutional-neural-networks.pdf": "Convolutional Neural Networks",
            "cheatsheet-recurrent-neural-networks.pdf": "Recurrent Neural Networks",
            "cheatsheet-deep-learning-tips-tricks.pdf": "Deep Learning Tips and Tricks",
            "super-cheatsheet-deep-learning.pdf": "Deep Learning Cheatsheet — Complete Compilation",
        }
        for name, title in titles.items():
            records.append(
                import_pdf(root, source_root, source, source_root / "en" / name, title)
            )
    elif source.key == "ml-visualized":
        records.append(import_markdown(root, source_root, source, source_root / "README.md"))
        for relative in (
            "chapter1/interactive_linear_regression.md",
            "chapter3/interactive_perceptron.md",
            "chapter3/interactive_logistic_regression.md",
        ):
            records.append(
                import_markdown(root, source_root, source, source_root / relative)
            )
        records.append(
            import_pdf(
                root,
                source_root,
                source,
                source_root / "book" / "main.pdf",
                "Machine Learning Visualized — Complete Book",
            )
        )
    elif source.key == "genai-agents":
        records.append(import_markdown(root, source_root, source, source_root / "README.md"))
        for path in sorted((source_root / "all_agents_tutorials").glob("*.ipynb")):
            records.append(import_notebook(root, source_root, source, path))
    manifest = output_path(root, source, "import-manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"{source.key}: imported {len(records)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collection", choices=sorted(SOURCES), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    import_collection(
        args.root.resolve(),
        args.source_root.resolve(),
        SOURCES[args.collection],
    )


if __name__ == "__main__":
    main()
