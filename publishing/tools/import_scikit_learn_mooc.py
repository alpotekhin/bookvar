#!/usr/bin/env python3
"""Import the complete scikit-learn MOOC as source-native Bookvar pages.

The upstream course is CC BY 4.0. This importer keeps the original English
prose and code, localizes the course figures, pins every provenance URL to a
commit, and produces a manifest plus a navigation snippet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath


REPOSITORY = "https://github.com/INRIA/scikit-learn-mooc"
LICENSE_NAME = "CC BY 4.0"
OUTPUT_NAME = "Scikit-learn MOOC"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_title(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[*_`#🎥🚧]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" :-")


def first_markdown_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return clean_title(match.group(1)) if match else fallback


def frontmatter(title: str, commit: str, source_path: str, kind: str) -> str:
    safe_title = title.replace('"', '\\"')
    source_url = f"{REPOSITORY}/blob/{commit}/{source_path}"
    license_url = f"{REPOSITORY}/blob/{commit}/LICENSE"
    return f"""---
title: "{safe_title}"
type: external-resource
status: imported-source
language: en
source_kind: {kind}
source_commit: {commit}
---

> [!note] Original source material
> This page preserves [`{source_path}`]({source_url}) from the
> [scikit-learn MOOC]({REPOSITORY}/tree/{commit}) at commit `{commit}`.
> Course material is licensed under [{LICENSE_NAME}]({license_url}).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

"""


def normalize_route(path: str) -> str:
    route = path.removesuffix(".md").removesuffix(".py")
    route = route.lower().replace("_", "-").replace(" ", "-")
    route = re.sub(r"[^a-z0-9/.-]+", "-", route)
    route = re.sub(r"-+", "-", route)
    route = route.strip("/-")
    return route.removesuffix("/index")


def resolve_source_target(source_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    joined = PurePosixPath(source_path).parent / target
    parts: list[str] = []
    for part in joined.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def rewrite_links(text: str, source_path: str, commit: str) -> str:
    """Localize figures and pin other relative links to upstream."""

    def image(match: re.Match[str]) -> str:
        alt, target = match.group(1), match.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I):
            return match.group(0)
        resolved = resolve_source_target(source_path, target)
        if resolved.startswith("figures/"):
            return f"![[Assets/Sources/{OUTPUT_NAME}/{resolved}|{alt or 'Source figure'}]]"
        return (
            f"![{alt}]({REPOSITORY}/raw/{commit}/{resolved})"
        )

    def html_image(match: re.Match[str]) -> str:
        target = match.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I):
            return match.group(0)
        resolved = resolve_source_target(source_path, target)
        if resolved.startswith("figures/"):
            return f"![[Assets/Sources/{OUTPUT_NAME}/{resolved}|Source figure]]"
        return f"![Source figure]({REPOSITORY}/raw/{commit}/{resolved})"

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith("#") or re.match(
            r"^[a-z][a-z0-9+.-]*:", target, flags=re.I
        ):
            return match.group(0)
        resolved = resolve_source_target(source_path, target)
        return f"[{label}]({REPOSITORY}/blob/{commit}/{resolved})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", image, text)
    text = re.sub(
        r'<img\b[^>]*\bsrc=["\']([^"\']+)["\'][^>]*>',
        html_image,
        text,
        flags=re.I,
    )
    text = re.sub(
        r'(<iframe\b[^>]*\bsrc=["\'])\.\./slides/index\.html',
        r'\1https://inria.github.io/scikit-learn-mooc/slides/index.html',
        text,
        flags=re.I,
    )
    return re.sub(
        r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", link, text
    )


def normalize_myst(text: str) -> str:
    """Convert common MyST-only constructs to portable Markdown."""

    text = re.sub(
        r"^```\{tableofcontents\}\s*\n```\s*$",
        "> [!info] Course navigation\n> Use the Bookvar sidebar or the links on the course index.",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^```\{nb-exec-table\}\s*\n```\s*$",
        "> [!info] Notebook timings\n> See the pinned upstream page for the generated execution-time table.",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"```\{(note|tip|warning|caution|important|admonition)\}\s*(.*?)\n(.*?)```",
        lambda m: (
            f"> [!{('warning' if m.group(1) == 'caution' else ('note' if m.group(1) == 'admonition' else m.group(1)))}]"
            f"{(' ' + m.group(2).strip()) if m.group(2).strip() else ''}\n"
            + "\n".join(f"> {line}" if line else ">" for line in m.group(3).splitlines())
        ),
        text,
        flags=re.S,
    )
    text = re.sub(r"\{[a-z]+ref\}`([^`]+)`", r"`\1`", text)
    text = re.sub(r"\{term\}`([^`]+)`", r"\1", text)
    return text


def strip_duplicate_title(text: str, title: str) -> str:
    match = re.match(r"^\s*#\s+(.+?)(?:\r?\n|$)", text)
    if match and clean_title(match.group(1)) == title:
        return text[match.end() :].lstrip()
    return text


def import_markdown(
    root: Path, source_root: Path, path: Path, commit: str
) -> dict[str, object]:
    source_path = path.relative_to(source_root).as_posix()
    raw = path.read_text(encoding="utf-8")
    title = first_markdown_title(raw, path.stem.replace("_", " ").title())
    body = strip_duplicate_title(raw, title)
    body = normalize_myst(rewrite_links(body, source_path, commit))
    destination = (
        root / "05 Источники" / "Courses" / OUTPUT_NAME / f"{source_path}.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        frontmatter(title, commit, source_path, "markdown")
        + re.sub(r"[ \t]+$", "", body, flags=re.M).rstrip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_path": source_path,
        "kind": "markdown",
        "output": destination.relative_to(root).as_posix(),
        "route": f"sources/courses/scikit-learn-mooc/{normalize_route(source_path)}",
        "sha256": sha256(path),
    }


def parse_jupytext(path: Path) -> list[tuple[str, str]]:
    """Parse the percent-format cells used by the upstream course."""

    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    current: list[str] = []
    in_header = False
    for line in lines:
        if line == "# ---" and not cells and not current:
            in_header = not in_header
            continue
        if in_header:
            continue
        marker = re.match(r"^# %%(?: \[markdown\])?\s*$", line)
        if marker:
            if current:
                cells.append((kind, current))
            kind = "markdown" if "[markdown]" in line else "code"
            current = []
            continue
        current.append(line)
    if current:
        cells.append((kind, current))

    rendered: list[tuple[str, str]] = []
    for cell_kind, cell_lines in cells:
        if cell_kind == "markdown":
            cleaned = [
                line[2:] if line.startswith("# ") else ("" if line == "#" else line)
                for line in cell_lines
            ]
            rendered.append(("markdown", "\n".join(cleaned).strip()))
        else:
            rendered.append(("code", "\n".join(cell_lines).rstrip()))
    return [(kind, text) for kind, text in rendered if text]


def import_jupytext(
    root: Path, source_root: Path, path: Path, commit: str
) -> dict[str, object]:
    source_path = path.relative_to(source_root).as_posix()
    cells = parse_jupytext(path)
    markdown_text = "\n".join(text for kind, text in cells if kind == "markdown")
    title = first_markdown_title(markdown_text, path.stem.replace("_", " ").title())
    chunks: list[str] = []
    removed_title = False
    for kind, text in cells:
        if kind == "markdown":
            if not removed_title:
                text = strip_duplicate_title(text, title)
                removed_title = True
            text = normalize_myst(rewrite_links(text, source_path, commit))
            if text.strip():
                chunks.append(text.strip())
        else:
            safe_code = text.replace("[[", "[\u200b[").replace("]]", "]\u200b]")
            chunks.append(f"```python\n{safe_code}\n```")
    destination = (
        root / "05 Источники" / "Courses" / OUTPUT_NAME / f"{source_path}.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        frontmatter(title, commit, source_path, "jupytext-notebook")
        + "\n\n".join(chunks).rstrip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_path": source_path,
        "kind": "jupytext-notebook",
        "output": destination.relative_to(root).as_posix(),
        "route": f"sources/courses/scikit-learn-mooc/{normalize_route(source_path)}",
        "sha256": sha256(path),
        "cells": len(cells),
    }


def write_index(root: Path, commit: str, records: list[dict[str, object]]) -> None:
    destination = root / "05 Источники" / "Courses" / f"{OUTPUT_NAME}.md"
    by_group: dict[str, list[dict[str, object]]] = {}
    for record in records:
        path = str(record["source_path"])
        group = path.split("/", 1)[0]
        by_group.setdefault(group, []).append(record)
    chunks = [
        frontmatter("Scikit-learn MOOC — complete source course", commit, "README.md", "course-index"),
        "# Scikit-learn MOOC — complete source course",
        "",
        "A complete, source-native snapshot of the official course: original English "
        "lectures, exercises, solutions, slide sources, and reusable figures. "
        "Nothing in this collection is a Bookvar summary.",
        "",
        f"- Upstream: [{REPOSITORY}]({REPOSITORY}/tree/{commit})",
        f"- Commit: `{commit}`",
        f"- License: [{LICENSE_NAME}]({REPOSITORY}/blob/{commit}/LICENSE)",
        f"- Imported pages: **{len(records)}**",
    ]
    for group, items in sorted(by_group.items()):
        chunks.extend(["", f"## {group.replace('_', ' ').title()}", ""])
        for item in items:
            label = Path(str(item["source_path"])).stem.replace("_", " ")
            chunks.append(f"- [{label}](/" + str(item["route"]) + "/)")
    destination.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def update_navigation(root: Path, records: list[dict[str, object]]) -> None:
    """Replace the generated MOOC route block in navigation.yml."""

    navigation = root / "publishing" / "navigation.yml"
    text = navigation.read_text(encoding="utf-8")
    block_pattern = re.compile(
        r"      - source: 05 Источники/Courses/Scikit-learn MOOC(?:\.md|/[^\n]+)\n"
        r"        route: sources/courses/scikit-learn-mooc/[^\n]+\n"
    )
    text = block_pattern.sub("", text)
    entries = [
        "      - source: 05 Источники/Courses/Scikit-learn MOOC.md\n"
        "        route: sources/courses/scikit-learn-mooc/index\n"
    ]
    for record in records:
        entries.append(
            f"      - source: {record['output']}\n"
            f"        route: {record['route']}\n"
        )
    insertion = "".join(entries)
    marker = "      - source: 05 Источники/Source maps/"
    position = text.find(marker)
    if position < 0:
        raise RuntimeError("Cannot find the source-map insertion point in navigation.yml")
    navigation.write_text(text[:position] + insertion + text[position:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--update-navigation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    commit = (
        __import__("subprocess")
        .run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )

    assets = root / "Assets" / "Sources" / OUTPUT_NAME / "figures"
    if assets.exists():
        shutil.rmtree(assets)
    shutil.copytree(source_root / "figures", assets)

    candidates = sorted((source_root / "jupyter-book").rglob("*.md"))
    candidates += sorted((source_root / "python_scripts").rglob("*.py"))
    candidates += sorted((source_root / "slides").glob("*.md"))
    records: list[dict[str, object]] = []
    for path in candidates:
        if path.is_symlink():
            continue
        if path.suffix == ".py":
            records.append(import_jupytext(root, source_root, path, commit))
        else:
            records.append(import_markdown(root, source_root, path, commit))

    output_dir = root / "05 Источники" / "Courses" / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "import-manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_index(root, commit, records)
    if args.update_navigation:
        update_navigation(root, records)
    print(
        f"Imported {len(records)} pages and "
        f"{sum(1 for _ in assets.rglob('*') if _.is_file())} figures "
        f"from scikit-learn MOOC commit {commit}."
    )


if __name__ == "__main__":
    main()
