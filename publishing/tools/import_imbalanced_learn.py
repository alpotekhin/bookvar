#!/usr/bin/env python3
"""Publish the pinned imbalanced-learn corpus as readable source pages.

The byte-for-byte upstream files live under
``05 Источники/imbalanced-learn/0.14.2/original``.  This importer verifies
their SHA-256 values against the pinned manifest, converts RST and
Sphinx-Gallery Python into readable Markdown, and registers deterministic
publication routes.  It never edits the original files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


VERSION = "0.14.2"
COMMIT = "8504e95f0160f61d1b617ca66f779646d2ee609e"
REPOSITORY = "https://github.com/scikit-learn-contrib/imbalanced-learn"
LICENSE_NAME = "MIT"
ROUTE_ROOT = "sources/imbalanced-learn/0-14-2"
SOURCE_ROOT_RELATIVE = Path("05 Источники/imbalanced-learn/0.14.2")
PUBLISHED_DIR = "published"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontmatter(title: str, upstream_path: str, digest: str, kind: str) -> str:
    escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
    upstream_url = f"{REPOSITORY}/blob/{COMMIT}/{upstream_path}"
    return f"""---
title: "{escaped_title}"
type: external-resource
status: imported-source
language: en
source_kind: {kind}
source_version: {VERSION}
source_commit: {COMMIT}
source_sha256: {digest}
license: {LICENSE_NAME}
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`{upstream_path}`]({upstream_url}) at commit
> [`{COMMIT}`]({REPOSITORY}/commit/{COMMIT}). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `{digest}`. The project is distributed under the
> [MIT license]({REPOSITORY}/blob/{COMMIT}/LICENSE).

"""


def inline_rst(text: str) -> str:
    """Translate the small, known subset of inline RST used by this corpus."""

    text = re.sub(
        r"`([^`<]+?)\s*<([^>]+)>`_",
        lambda match: (
            f"[{' '.join(match.group(1).split()).strip('[]')}]"
            f"({match.group(2).strip()})"
        ),
        text,
        flags=re.S,
    )
    text = re.sub(
        r":(?:class|func|mod|meth|attr|data|obj):`([^`]+)`",
        lambda match: f"`{match.group(1).lstrip('~').split('.')[-1]}`",
        text,
    )
    text = re.sub(
        r":math:`([^`]+)`",
        lambda match: f"${match.group(1)}$",
        text,
    )
    text = re.sub(
        r":cite:`([^`]+)`",
        lambda match: " ".join(
            f"[{key.strip()}](/{ROUTE_ROOT}/user-guide/bibliography/#{key.strip()})"
            for key in match.group(1).split(",")
        ),
        text,
    )
    text = re.sub(
        r":ref:`([^`]+)`",
        lambda match: f"[`{match.group(1)}`](https://imbalanced-learn.org/stable/)",
        text,
    )
    text = re.sub(r"``([^`\n]+)``", r"`\1`", text)
    text = text.replace(r"\ ", " ")
    return text


def _indented_block(lines: list[str], start: int) -> tuple[list[str], int]:
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return [], index
    indentation = len(lines[index]) - len(lines[index].lstrip())
    if indentation == 0:
        return [], start
    block: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip():
            current = len(line) - len(line.lstrip())
            if current < indentation:
                break
            block.append(line[indentation:])
        else:
            block.append("")
        index += 1
    while block and not block[-1].strip():
        block.pop()
    return block, index


def rst_to_markdown(raw: str) -> str:
    """Convert imbalanced-learn's narrative RST to portable Markdown."""

    lines = raw.expandtabs(4).splitlines()
    output: list[str] = []
    index = 0
    seen_title = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if (
            re.fullmatch(r"[=\-~^`:+*#.]{3,}", stripped)
            and index + 2 < len(lines)
            and lines[index + 1].strip()
            and re.fullmatch(r"[=\-~^`:+*#.]{3,}", lines[index + 2].strip())
        ):
            # RST permits an overline and underline around the document title.
            # The underline branch below will emit the actual Markdown heading.
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and stripped
            and re.fullmatch(r"[=\-~^`:+*#.]{3,}", lines[index + 1].strip())
            and len(lines[index + 1].strip()) >= len(stripped)
        ):
            marker = lines[index + 1].strip()[0]
            if not seen_title:
                level = 1
                seen_title = True
            else:
                level = {"=": 2, "-": 3, "~": 4, "^": 5, ".": 4}.get(marker, 3)
            output.append(f"{'#' * level} {inline_rst(stripped)}")
            output.append("")
            index += 2
            continue

        if re.fullmatch(r"\.\. _[^:]+:", stripped):
            index += 1
            continue

        if stripped.startswith(".. currentmodule::"):
            index += 1
            continue

        directive = re.match(r"^\.\. (math|image|topic|note|warning|bibliography)::\s*(.*)$", stripped)
        if directive:
            kind, argument = directive.groups()
            block, next_index = _indented_block(lines, index + 1)
            if kind == "math":
                formula = "\n".join(block).strip()
                output.extend(["$$", formula, "$$", ""])
            elif kind == "image":
                image_name = Path(argument).name
                image_url = f"https://imbalanced-learn.org/stable/_images/{image_name}"
                target = next(
                    (
                        item.split(":target:", 1)[1].strip()
                        for item in block
                        if item.strip().startswith(":target:")
                    ),
                    "https://imbalanced-learn.org/stable/auto_examples/index.html",
                )
                if target.startswith("./"):
                    target = "https://imbalanced-learn.org/stable/" + target[2:]
                output.extend(
                    [
                        f"[![Official imbalanced-learn figure: {image_name}]({image_url})]({target})",
                        "",
                    ]
                )
            elif kind == "bibliography":
                output.extend(
                    [
                        f"See the [complete bibliography](/"
                        f"{ROUTE_ROOT}/user-guide/bibliography/).",
                        "",
                    ]
                )
            else:
                label = argument or kind.replace("_", " ").title()
                callout = "warning" if kind == "warning" else "note"
                converted = rst_to_markdown("\n".join(block)).strip()
                output.append(f"> [!{callout}] {inline_rst(label)}")
                if converted:
                    output.extend(
                        f"> {item}" if item else ">" for item in converted.splitlines()
                    )
                output.append("")
            index = next_index if next_index > index + 1 else index + 1
            continue

        definition = re.match(r"^:([^:]+):\s*$", stripped)
        if definition:
            output.extend([f"**{definition.group(1)}**", ""])
            index += 1
            continue

        if stripped.endswith("::"):
            output.append(inline_rst(line.rstrip()[:-1]))
            block, next_index = _indented_block(lines, index + 1)
            if block:
                language = "python" if any(
                    re.match(r"\s*(>>>|\.\.\.)\s", item) for item in block
                ) else "text"
                output.extend([f"```{language}", *block, "```", ""])
                index = next_index
            else:
                index += 1
            continue

        output.append(line)
        index += 1

    result = "\n".join(output)
    result = inline_rst(result)
    result = re.sub(r"(?m)^(\s*)\* ", r"\1- ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"


def python_title_and_body(raw: str, fallback: str) -> tuple[str, str, int]:
    tree = ast.parse(raw)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        node = tree.body[0]
        docstring = node.value.value
        title_match = re.search(
            r"(?m)^([^\n]+)\n[=\-]{3,}\n", docstring.strip()
        )
        title = title_match.group(1).strip() if title_match else fallback
        body = rst_to_markdown(docstring)
        body = re.sub(r"^#\s+[^\n]+\n+", "", body, count=1)
        return title, body, int(node.end_lineno or 0)
    return fallback, "", 0


def markdown_comment_cell(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        if line.startswith("# "):
            cleaned.append(line[2:])
        elif line == "#":
            cleaned.append("")
        elif line.startswith("#"):
            cleaned.append(line[1:].lstrip())
        else:
            cleaned.append(line)
    return rst_to_markdown("\n".join(cleaned)).strip()


def demote_headings(markdown: str) -> str:
    return re.sub(r"(?m)^(#{1,5})\s+", lambda match: "#" + match.group(0), markdown)


def python_gallery_to_markdown(raw: str, fallback: str) -> tuple[str, str]:
    """Render Sphinx-Gallery percent cells as narrative interleaved with code."""

    title, introduction, docstring_end = python_title_and_body(raw, fallback)
    remaining = raw.splitlines()[docstring_end:]
    chunks: list[str] = [f"# {title}"]
    if introduction:
        chunks.append(introduction.strip())

    current_kind = "code"
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        while current and not current[0].strip():
            current.pop(0)
        while current and not current[-1].strip():
            current.pop()
        if not current:
            return
        if current_kind == "markdown":
            rendered = demote_headings(markdown_comment_cell(current))
            if rendered:
                chunks.append(rendered)
        else:
            chunks.append("```python\n" + "\n".join(current) + "\n```")
        current = []

    for line in remaining:
        marker = re.match(r"^# %%(?: \[markdown\])?\s*$", line)
        if marker:
            flush()
            current_kind = "markdown" if "[markdown]" in line else "code"
            continue
        if current_kind == "markdown" and line and not line.startswith("#"):
            flush()
            current_kind = "code"
        current.append(line)
    flush()
    return title, "\n\n".join(chunks).strip() + "\n"


def parse_bibtex(raw: str) -> list[dict[str, Any]]:
    """Parse this pinned BibTeX file without adding a runtime dependency."""

    entries: list[dict[str, Any]] = []
    index = 0
    while True:
        match = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", raw[index:])
        if not match:
            break
        start = index + match.start()
        cursor = index + match.end()
        depth = 1
        while cursor < len(raw) and depth:
            if raw[cursor] == "{":
                depth += 1
            elif raw[cursor] == "}":
                depth -= 1
            cursor += 1
        entry_raw = raw[start:cursor]
        body = entry_raw[entry_raw.find(",") + 1 : -1]
        fields: dict[str, str] = {}
        field_index = 0
        while field_index < len(body):
            field = re.search(r"(\w+)\s*=\s*\{", body[field_index:])
            if not field:
                break
            name = field.group(1).lower()
            value_start = field_index + field.end()
            value_cursor = value_start
            value_depth = 1
            while value_cursor < len(body) and value_depth:
                if body[value_cursor] == "{":
                    value_depth += 1
                elif body[value_cursor] == "}":
                    value_depth -= 1
                value_cursor += 1
            fields[name] = " ".join(body[value_start : value_cursor - 1].split())
            field_index = value_cursor
        entries.append(
            {
                "type": match.group(1),
                "key": match.group(2),
                "fields": fields,
                "raw": entry_raw.strip(),
            }
        )
        index = cursor
    return entries


def bibliography_to_markdown(raw: str) -> tuple[str, str]:
    entries = parse_bibtex(raw)
    chunks = [
        "# Bibliography",
        "",
        f"The upstream bibliography contains **{len(entries)} complete BibTeX entries**.",
        "Each record below keeps its citation key and every supplied field.",
        "",
    ]
    preferred_fields = (
        "author",
        "title",
        "journal",
        "booktitle",
        "year",
        "volume",
        "number",
        "pages",
        "publisher",
        "organization",
    )
    for entry in entries:
        fields = entry["fields"]
        chunks.extend([f"## {entry['key']}", ""])
        for field in preferred_fields:
            if field in fields:
                chunks.append(f"- **{field.title()}:** {fields[field]}")
        for field in sorted(set(fields) - set(preferred_fields)):
            chunks.append(f"- **{field.title()}:** {fields[field]}")
        chunks.extend(
            [
                "",
                "<details>",
                "<summary>Original BibTeX record</summary>",
                "",
                "```bibtex",
                entry["raw"],
                "```",
                "</details>",
                "",
            ]
        )
    return "Bibliography", "\n".join(chunks).strip() + "\n"


def license_to_markdown(raw: str) -> tuple[str, str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", raw) if paragraph.strip()]
    body = ["# The MIT License", ""]
    for paragraph in paragraphs[1:]:
        body.extend([" ".join(paragraph.splitlines()), ""])
    return "The MIT License", "\n".join(body).strip() + "\n"


def output_spec(upstream_path: str) -> tuple[Path, str, str]:
    if upstream_path == "LICENSE":
        return Path(PUBLISHED_DIR) / "license.md", f"{ROUTE_ROOT}/license", "license"
    if upstream_path == "doc/bibtex/refs.bib":
        return (
            Path(PUBLISHED_DIR) / "user-guide/bibliography.md",
            f"{ROUTE_ROOT}/user-guide/bibliography",
            "bibliography",
        )
    if upstream_path.startswith("doc/"):
        stem = Path(upstream_path).stem.replace("_", "-")
        if stem == "zzz-references":
            stem = "references"
        return (
            Path(PUBLISHED_DIR) / "user-guide" / f"{stem}.md",
            f"{ROUTE_ROOT}/user-guide/{stem}",
            "user-guide",
        )
    if upstream_path.startswith("examples/"):
        stem = Path(upstream_path).stem.replace("_", "-")
        return (
            Path(PUBLISHED_DIR) / "gallery" / f"{stem}.md",
            f"{ROUTE_ROOT}/gallery/{stem}",
            "gallery-example",
        )
    raise ValueError(f"Unsupported upstream path: {upstream_path}")


def title_from_rst(raw: str, fallback: str) -> str:
    lines = raw.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() and re.fullmatch(r"[=\-~^]{3,}", lines[index + 1].strip()):
            return line.strip()
    return fallback


def write_index(source_root: Path, records: list[dict[str, str]]) -> None:
    groups = (
        ("User Guide", [item for item in records if item["kind"] == "user-guide"]),
        ("Gallery examples", [item for item in records if item["kind"] == "gallery-example"]),
        (
            "License and bibliography",
            [item for item in records if item["kind"] in {"license", "bibliography"}],
        ),
    )
    chunks = [
        "---",
        'title: "imbalanced-learn 0.14.2 — readable original sources"',
        "type: source-note",
        "status: pinned",
        "language: en",
        "source_kind: corpus-index",
        f"source_commit: {COMMIT}",
        f"license: {LICENSE_NAME}",
        "---",
        "",
        "# imbalanced-learn 0.14.2 — readable original sources",
        "",
        "This publication layer contains every file in Bookvar's pinned",
        "imbalanced-learn corpus: the complete English User Guide selection,",
        "complete Sphinx-Gallery examples with narrative interleaved with code,",
        "the full bibliography, and the license. These are source pages, not",
        "summaries or translations.",
        "",
        f"- Upstream version: **{VERSION}**",
        f"- Pinned commit: [`{COMMIT}`]({REPOSITORY}/commit/{COMMIT})",
        f"- License: [MIT]({REPOSITORY}/blob/{COMMIT}/LICENSE)",
        f"- Verified original files: **{len(records)}**",
        f"- Publication routes including this index: **{len(records) + 1}**",
        "",
        "Regenerate and re-verify the corpus with:",
        "",
        "```bash",
        "python publishing/tools/import_imbalanced_learn.py --root . --update-navigation",
        "```",
        "",
    ]
    for heading, items in groups:
        chunks.extend([f"## {heading}", ""])
        for item in items:
            chunks.append(f"- [{item['title']}](/" + item["route"] + "/)")
        chunks.append("")
    index_path = source_root / PUBLISHED_DIR / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def update_navigation(root: Path, records: list[dict[str, str]]) -> None:
    navigation = root / "publishing/navigation.yml"
    text = navigation.read_text(encoding="utf-8")
    generated = re.compile(
        r"      - source: 05 Источники/imbalanced-learn/0\.14\.2/published/[^\n]+\.md\n"
        r"        route: sources/imbalanced-learn/0-14-2(?:/[^\n]+)?\n"
    )
    text = generated.sub("", text)
    entries = [
        "      - source: 05 Источники/imbalanced-learn/0.14.2/published/index.md\n"
        f"        route: {ROUTE_ROOT}\n"
    ]
    for record in records:
        entries.append(
            f"      - source: {record['output']}\n"
            f"        route: {record['route']}\n"
        )
    marker = "      - source: 05 Источники/Source maps/"
    position = text.find(marker)
    if position < 0:
        raise RuntimeError("Cannot find source-map insertion point in navigation.yml")
    navigation.write_text(
        text[:position] + "".join(entries) + text[position:],
        encoding="utf-8",
    )


def import_corpus(root: Path, update_routes: bool) -> list[dict[str, str]]:
    source_root = root / SOURCE_ROOT_RELATIVE
    manifest = yaml.safe_load((source_root / "manifest.yml").read_text(encoding="utf-8"))
    source = manifest["source"]
    if source["version"] != VERSION or source["commit"] != COMMIT or source["license"] != LICENSE_NAME:
        raise RuntimeError("Pinned source metadata no longer matches the importer")

    records: list[dict[str, str]] = []
    for item in manifest["files"]:
        original = source_root / item["path"]
        digest = sha256(original)
        if digest != item["sha256"]:
            raise RuntimeError(
                f"Source drift for {item['path']}: expected {item['sha256']}, got {digest}"
            )
        destination_relative, route, kind = output_spec(item["upstream_path"])
        destination = source_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = original.read_text(encoding="utf-8")

        if original.suffix == ".rst":
            title = title_from_rst(raw, original.stem.replace("_", " ").title())
            body = rst_to_markdown(raw)
        elif original.suffix == ".py":
            title, body = python_gallery_to_markdown(
                raw, original.stem.replace("_", " ").title()
            )
        elif original.suffix == ".bib":
            title, body = bibliography_to_markdown(raw)
        elif original.name == "LICENSE":
            title, body = license_to_markdown(raw)
        else:
            raise ValueError(f"Unsupported source file: {original}")

        destination.write_text(
            frontmatter(title, item["upstream_path"], digest, kind)
            + body.rstrip()
            + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "title": title,
                "kind": kind,
                "original": item["path"],
                "upstream_path": item["upstream_path"],
                "source_sha256": digest,
                "output": destination.relative_to(root).as_posix(),
                "route": route,
            }
        )

    if len(records) != 20:
        raise RuntimeError(f"Expected all 20 pinned originals, imported {len(records)}")

    write_index(source_root, records)
    publication_manifest = {
        "schema_version": 1,
        "source": source,
        "page_count": len(records),
        "route_count": len(records) + 1,
        "index_route": ROUTE_ROOT,
        "pages": records,
    }
    (source_root / PUBLISHED_DIR / "import-manifest.json").write_text(
        json.dumps(publication_manifest, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    if update_routes:
        update_navigation(root, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--update-navigation", action="store_true")
    arguments = parser.parse_args()
    records = import_corpus(arguments.root.resolve(), arguments.update_navigation)
    print(
        f"Published {len(records)} verified source pages plus one index "
        f"at {ROUTE_ROOT}/ ({len(records) + 1} routes)."
    )


if __name__ == "__main__":
    main()
