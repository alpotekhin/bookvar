#!/usr/bin/env python3
"""Import the latest HSE ML course seasons as source-native Bookvar pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
from pathlib import Path


COMMIT = "4b21051531fb72dc9eef58632332ad971c92d006"
REPOSITORY = "https://github.com/esokolov/ml-course-hse"
SEASONS = ("ml1-2026-spring", "ml2-2026-spring")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    value = value.lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9а-яё./+-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def provenance(source_path: str, kind: str) -> str:
    url = f"{REPOSITORY}/blob/{COMMIT}/{source_path}"
    return f"""---
title: "{title(Path(source_path)).replace('"', '')}"
type: external-resource
status: imported-source
source_kind: {kind}
source_commit: {COMMIT}
language: ru
---

> [!note] Полный оригинальный материал HSE
> Источник: [`{source_path}`]({url}), commit `{COMMIT}`.
> В репозитории не найдено общей лицензии; материал перенесён без перевода
> по прямому разрешению владельца Bookvar для некоммерческого учебного архива.
> Ссылка на оригинал и provenance сохранены.

"""


def output_text(output: dict) -> str:
    data = output.get("data", {})
    if "text/plain" in data:
        value = data["text/plain"]
        return "".join(value) if isinstance(value, list) else str(value)
    text = output.get("text")
    if isinstance(text, list):
        return "".join(text)
    return str(text or "")


def import_notebook(
    root: Path, source_root: Path, path: Path, asset_root: Path
) -> tuple[str, list[str]]:
    relative = path.relative_to(source_root).as_posix()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = [provenance(relative, "jupyter-notebook"), f"# {title(path)}\n"]
    localized_assets: list[str] = []
    image_index = 0
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        source = "".join(cell.get("source", [])).rstrip()
        if cell.get("cell_type") == "markdown":
            if source:
                def localize_embedded_image(match: re.Match[str]) -> str:
                    nonlocal image_index
                    image_index += 1
                    media_type = match.group(2).lower()
                    extension = "jpg" if media_type in {"jpeg", "jpg"} else media_type
                    target = (
                        asset_root
                        / path.relative_to(source_root).with_suffix("")
                        / f"cell-{cell_index}-markdown-{image_index}.{extension}"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base64.b64decode(match.group(3)))
                    asset_link = target.relative_to(root).as_posix()
                    localized_assets.append(asset_link)
                    alt = match.group(1).strip() or "Notebook illustration"
                    return f"![[{asset_link}|{alt}]]"

                source = re.sub(
                    r"!\[([^\]]*)\]\(data:image/(png|jpe?g|webp);base64,([A-Za-z0-9+/=]+)\)",
                    localize_embedded_image,
                    source,
                )
                parts.append(source + "\n")
            continue
        if cell.get("cell_type") != "code":
            continue
        parts.append(f"```python\n{source}\n```\n")
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            image = data.get("image/png")
            if image:
                image_index += 1
                encoded = "".join(image) if isinstance(image, list) else image
                target = (
                    asset_root
                    / path.relative_to(source_root).with_suffix("")
                    / f"cell-{cell_index}-output-{image_index}.png"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(encoded))
                asset_link = target.relative_to(root).as_posix()
                parts.append(f"![[{asset_link}|Notebook output]]\n")
                localized_assets.append(asset_link)
            text = output_text(output).rstrip()
            if text:
                parts.append(f"```text\n{text}\n```\n")
    return "\n".join(parts).rstrip() + "\n", localized_assets


def update_navigation(root: Path, records: list[dict]) -> None:
    navigation = root / "publishing/navigation.yml"
    text = navigation.read_text(encoding="utf-8")
    begin = "      # BEGIN GENERATED HSE ML COURSE\n"
    end = "      # END GENERATED HSE ML COURSE\n"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--update-navigation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    destination = root / "05 Источники/Courses/HSE ML course"
    asset_root = root / "Assets/Sources/HSE ML course"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for season in SEASONS:
        season_root = source_root / season
        paths = sorted(
            path
            for path in season_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".ipynb", ".py"}
        )
        for path in paths:
            relative = path.relative_to(source_root).as_posix()
            output = destination / Path(relative).with_suffix(path.suffix + ".md")
            output.parent.mkdir(parents=True, exist_ok=True)
            assets: list[str] = []
            if path.suffix.lower() == ".ipynb":
                body, assets = import_notebook(root, source_root, path, asset_root)
                output.write_text(body, encoding="utf-8")
                kind = "jupyter-notebook"
            elif path.suffix.lower() == ".py":
                source = path.read_text(encoding="utf-8")
                output.write_text(
                    provenance(relative, "python-source")
                    + f"# {title(path)}\n\n```python\n{source.rstrip()}\n```\n",
                    encoding="utf-8",
                )
                kind = "python-source"
            else:
                # Keep PDFs beside their published source pages. The publisher
                # resolves wiki links to files inside the source tree, whereas
                # non-image files under Assets are not registered as pages.
                pdf_target = destination / relative
                pdf_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, pdf_target)
                pdf_link = pdf_target.relative_to(root).as_posix()
                output.write_text(
                    provenance(relative, "pdf")
                    + f"# {title(path)}\n\n"
                    + f"[[{pdf_link}|Открыть полный оригинальный PDF]]\n",
                    encoding="utf-8",
                )
                assets = [pdf_link]
                kind = "pdf"
            route = f"sources/courses/hse-ml-course/{slug(relative.replace('.', '-'))}"
            records.append(
                {
                    "season": season,
                    "kind": kind,
                    "source": relative,
                    "output": output.relative_to(root).as_posix(),
                    "route": route,
                    "sha256": digest(path),
                    "assets": assets,
                }
            )

    (destination / "import-manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index = root / "05 Источники/Courses/HSE ML course.md"
    lines = [
        "---",
        'title: "HSE ML course — 2026 source-native archive"',
        "type: source-note",
        "status: imported-source",
        f"source_commit: {COMMIT}",
        "---",
        "",
        "# HSE ML course — 2026 source-native archive",
        "",
        "Последние два трека курса сохранены в исходном русском языке: полные",
        "lecture PDFs, seminar/homework notebooks и сопровождающий Python code.",
        "",
    ]
    for season in SEASONS:
        lines.extend([f"## {season}", ""])
        for record in (item for item in records if item["season"] == season):
            lines.append(f"- [[{record['output']}|{title(Path(record['source']))}]]")
        lines.append("")
    index.write_text("\n".join(lines), encoding="utf-8")
    index_record = {
        "output": index.relative_to(root).as_posix(),
        "route": "sources/courses/hse-ml-course/index",
    }
    if args.update_navigation:
        update_navigation(root, [index_record, *records])
    print(f"Imported {len(records)} HSE course artifacts from commit {COMMIT}.")


if __name__ == "__main__":
    main()
