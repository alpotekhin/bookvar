#!/usr/bin/env python3
"""Publish Harvard CS249r Marimo labs and slide sources as readable Markdown.

The generated page keeps the complete original Python/TeX source and adds a
readable layer extracted from Marimo markdown cells or Beamer frames.  This is
an archival source layer, not a translation or a replacement for canonical
Bookvar chapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


COMMIT = "45ecc8d82fcae70c149cdce550d3b3d3411df913"
REPOSITORY = "https://github.com/harvard-edge/cs249r_book"
LICENSE = "CC BY-NC-SA 4.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_from_path(path: Path) -> str:
    stem = re.sub(r"^\d+_", "", path.stem)
    stem = re.sub(r"^lab_\d+_", "", stem)
    return stem.replace("_", " ").strip().title()


def extract_marimo_markdown(source: str) -> list[str]:
    """Extract the prose authored for visible ``mo.md`` cells."""
    pattern = re.compile(
        r"""mo\.md\(\s*(?:r|f|rf|fr)?(?P<quote>'''|\"\"\")(?P<body>.*?)(?P=quote)\s*\)""",
        re.DOTALL,
    )
    sections = []
    for match in pattern.finditer(source):
        body = match.group("body").strip()
        body = re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\}", "`…`", body)
        if body:
            sections.append(body)
    return sections


def clean_tex(text: str) -> str:
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\times": "×",
        r"\rightarrow": "→",
        "~": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\alert\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\(?:small|footnotesize|scriptsize|tiny|normalsize)\b", "", text)
    text = re.sub(r"\\(?:pause|centering|vfill|medskip|smallskip|bigskip)\b", "", text)
    text = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"[\2](\1)", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"<\1>", text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]*)\}", r"*Figure: `\1`*", text)
    text = re.sub(r"\\begin\{(?:itemize|enumerate|columns?|block|exampleblock|alertblock)\}(?:\{[^{}]*\})?", "", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate|columns?|block|exampleblock|alertblock)\}", "", text)
    text = re.sub(r"^\s*\\item(?:<[^>]*>)?\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_beamer_frames(source: str) -> list[tuple[str, str]]:
    frames = []
    frame_pattern = re.compile(
        r"\\begin\{frame\}(?:\[[^\]]*\])?(?:\{(?P<title>[^{}]*)\})?"
        r"(?P<body>.*?)\\end\{frame\}",
        re.DOTALL,
    )
    for index, match in enumerate(frame_pattern.finditer(source), start=1):
        title = clean_tex(match.group("title") or f"Frame {index}")
        body = clean_tex(match.group("body"))
        if body:
            frames.append((title, body))
    return frames


def frontmatter(title: str, source_path: str, kind: str) -> str:
    source_url = f"{REPOSITORY}/blob/{COMMIT}/{source_path}"
    return f"""---
title: "{title.replace('"', '\\"')}"
type: external-resource
status: imported-source
language: en
source_kind: {kind}
source_commit: {COMMIT}
---

> [!note] Complete original Harvard source
> Source: [`{source_path}`]({source_url}) at commit `{COMMIT}`.
> License: [{LICENSE}]({REPOSITORY}/blob/{COMMIT}/LICENSE).
> The readable layer below is mechanically extracted from the original;
> the complete unmodified source follows on the same page.

"""


def import_lab(root: Path, source_root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(source_root).as_posix()
    volume = path.parent.name
    title = f"{volume.upper()} Lab — {title_from_path(path)}"
    source = path.read_text(encoding="utf-8")
    prose = extract_marimo_markdown(source)
    readable = "\n\n---\n\n".join(prose) if prose else (
        "This lab is primarily executable code; use the complete source below."
    )
    output = (
        root
        / "05 Источники/Courses/Harvard ML Systems"
        / "labs"
        / volume
        / f"{path.stem}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        frontmatter(title, relative, "marimo-lab")
        + f"# {title}\n\n## Readable lab narrative\n\n{readable}\n\n"
        + "## Complete original Marimo source\n\n"
        + f"```python\n{source.rstrip()}\n```\n",
        encoding="utf-8",
    )
    route = f"sources/courses/harvard-ml-systems/labs/{volume}/{path.stem.replace('_', '-')}"
    return {
        "kind": "lab",
        "source": relative,
        "output": output.relative_to(root).as_posix(),
        "route": route,
        "sha256": sha256(path),
        "title": title,
    }


def import_slides(root: Path, source_root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(source_root).as_posix()
    volume = path.parents[1].name
    lecture = path.parent.name
    title = f"{volume.upper()} Slides — {title_from_path(path.parent)}"
    source = path.read_text(encoding="utf-8")
    frames = extract_beamer_frames(source)
    readable = "\n\n".join(f"## {heading}\n\n{body}" for heading, body in frames)
    if not readable:
        readable = "No Beamer frames were detected; consult the complete source below."
    output = (
        root
        / "05 Источники/Courses/Harvard ML Systems"
        / "slides"
        / volume
        / f"{lecture}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        frontmatter(title, relative, "beamer-slides")
        + f"# {title}\n\n## Readable slide sequence\n\n{readable}\n\n"
        + "## Complete original Beamer source\n\n"
        + f"```tex\n{source.rstrip()}\n```\n",
        encoding="utf-8",
    )
    route = f"sources/courses/harvard-ml-systems/slides/{volume}/{lecture.replace('_', '-')}"
    return {
        "kind": "slides",
        "source": relative,
        "output": output.relative_to(root).as_posix(),
        "route": route,
        "sha256": sha256(path),
        "title": title,
    }


def update_navigation(root: Path, records: list[dict[str, str]]) -> None:
    navigation = root / "publishing/navigation.yml"
    text = navigation.read_text(encoding="utf-8")
    begin = "      # BEGIN GENERATED HARVARD LABS AND SLIDES\n"
    end = "      # END GENERATED HARVARD LABS AND SLIDES\n"
    block = begin + "".join(
        f"      - source: {record['output']}\n        route: {record['route']}\n"
        for record in records
    ) + end
    if begin in text:
        text = text[: text.index(begin)] + block + text[text.index(end) + len(end) :]
    else:
        marker = "      - source: 05 Источники/Source maps/"
        position = text.find(marker)
        if position < 0:
            raise RuntimeError("Cannot find source-map insertion point")
        text = text[:position] + block + text[position:]
    navigation.write_text(text, encoding="utf-8")


def write_index(root: Path, records: list[dict[str, str]]) -> None:
    index = root / "05 Источники/Courses/Harvard ML Systems/Labs and slides.md"
    lines = [
        "---",
        'title: "Harvard ML Systems — labs and slides"',
        "type: source-note",
        "status: imported-source",
        "language: en",
        f"source_commit: {COMMIT}",
        "---",
        "",
        "# Harvard ML Systems — labs and slides",
        "",
        f"Complete source-native practice and lecture layer from commit `{COMMIT}`.",
        "Each page contains a readable extraction and the full original source.",
        "",
    ]
    for kind in ("lab", "slides"):
        lines.extend([f"## {'Marimo labs' if kind == 'lab' else 'Beamer slide decks'}", ""])
        for record in (item for item in records if item["kind"] == kind):
            lines.append(f"- [[{record['output']}|{record['title']}]]")
        lines.append("")
    index.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--update-navigation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()

    lab_paths = sorted((source_root / "labs/vol1").glob("lab_*.py"))
    lab_paths += sorted((source_root / "labs/vol2").glob("lab_*.py"))
    slide_paths = sorted((source_root / "slides/vol1").glob("*/*.tex"))
    slide_paths += sorted((source_root / "slides/vol2").glob("*/*.tex"))
    records = [import_lab(root, source_root, path) for path in lab_paths]
    records += [import_slides(root, source_root, path) for path in slide_paths]

    destination = root / "05 Источники/Courses/Harvard ML Systems"
    (destination / "labs-slides-manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_index(root, records)
    if args.update_navigation:
        update_navigation(root, records)
    print(
        f"Imported {len(lab_paths)} Marimo labs and "
        f"{len(slide_paths)} Beamer slide decks from Harvard commit {COMMIT}."
    )


if __name__ == "__main__":
    main()
