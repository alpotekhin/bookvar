#!/usr/bin/env python3
"""Import the complete source-native LLMsPracticalGuide repository corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


COMMIT = "c4a39847f5455b8383dffee55c4fe9e5e16966a4"
REPOSITORY = "https://github.com/Mooler0410/LLMsPracticalGuide"
COURSE_NAME = "LLMs Practical Guide"

MARKDOWN_SOURCES = {
    "README.md": {
        "title": "The Practical Guides for Large Language Models",
        "route": "sources/courses/llms-practical-guide/guide",
        "kind": "curated-survey-index",
    },
    "awesome_examples/social_game_werewolf.md": {
        "title": "Lesson — Social Game: One Night Ultimate Werewolf",
        "route": "sources/courses/llms-practical-guide/examples/social-game-werewolf",
        "kind": "case-study",
    },
    "awesome_examples/tableQA.md": {
        "title": "Lesson — Table-based Question Answering",
        "route": "sources/courses/llms-practical-guide/examples/table-qa",
        "kind": "case-study",
    },
    "source/README.md": {
        "title": "Evolutionary-tree figure change log",
        "route": "sources/courses/llms-practical-guide/source/change-log",
        "kind": "source-change-log",
    },
}

MEDIA_SOURCES = (
    "imgs/decision.png",
    "imgs/models-colorgrey.jpg",
    "imgs/qr_version.jpg",
    "imgs/survey-gif-test.gif",
    "imgs/tree.jpg",
    "imgs/tree.png",
    "awesome_examples/imgs/example_1_chatgpt_evidence.png",
    "awesome_examples/imgs/example_1_gpt4_evidence.png",
    "awesome_examples/imgs/example_2_chatgpt_evidence.png",
    "awesome_examples/imgs/example_2_gpt4_evidence.png",
)

EDITABLE_SOURCES = (
    "source/figure_gif.pptx",
    "source/figure_still.pptx",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source_root: Path) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"{source_root} is not a readable Git checkout") from error
    if actual != COMMIT:
        raise RuntimeError(
            f"Expected LLMsPracticalGuide commit {COMMIT}, found {actual}"
        )
    expected = set(MARKDOWN_SOURCES) | set(MEDIA_SOURCES) | set(EDITABLE_SOURCES)
    missing = sorted(path for path in expected if not (source_root / path).is_file())
    if missing:
        raise RuntimeError(f"Missing source files: {', '.join(missing)}")


def source_url(relative: str) -> str:
    return f"{REPOSITORY}/blob/{COMMIT}/{relative}"


def frontmatter(relative: str, metadata: dict[str, str], digest: str) -> str:
    title = metadata["title"].replace('"', '\\"')
    return f"""---
title: "{title}"
type: external-resource
status: imported-source
language: en
source_kind: {metadata["kind"]}
source_commit: {COMMIT}
source_sha256: {digest}
license_status: no-repository-license-found
---

> [!note] Complete original source
> This page preserves the complete English source
> [`{relative}`]({source_url(relative)}) from commit `{COMMIT}` without
> abridgement or translation. Relative media and editable figure links were
> localized for Bookvar.
>
> The repository contains no `LICENSE`, `COPYING`, or `NOTICE` file at this
> commit. Its README asks readers who use the resources to cite the associated
> survey, but that sentence is not treated here as a formal content license.
> The material is retained under the Bookvar owner's confirmed permission for
> this noncommercial educational archive, with authorship and the pinned
> original preserved.

"""


def localized_asset_path(relative: str) -> str:
    return f"Assets/Sources/{COURSE_NAME}/{relative}"


def localized_editable_path(relative: str) -> str:
    return f"05 Источники/Courses/{COURSE_NAME}/{relative}"


def image_embed(relative: str, width: str | None = None) -> str:
    width_suffix = f"|{width}" if width else ""
    return (
        f"![[{localized_asset_path(relative)}{width_suffix}]]\n\n"
        f"[Original image at pinned commit]({source_url(relative)})"
    )


def adapt_markdown(relative: str, body: str) -> str:
    if relative == "README.md":
        # GitHub and Astro use slightly different heading slug rules. Preserve
        # the source wording while making the original table of contents work
        # in the published archive.
        body = body.replace(
            '<h1 align="center">The Practical Guides for Large Language Models </h1>',
            '<h1 id="the-practical-guides-for-large-language-models" '
            'align="center">The Practical Guides for Large Language Models </h1>',
        )
        body = body.replace(
            "(#the-practical-guides-for-large-language-models-)",
            "(#the-practical-guides-for-large-language-models)",
        )
        body = body.replace(
            "(#Usage-and-Restrictions)",
            "(#usage-and-restrictions)",
        )
        body = re.sub(
            r'<p align="center">\s*<img src="https://camo\.githubusercontent\.com/[^"]+"'
            r'[^>]*data-canonical-src="https://awesome\.re/badge\.svg"[^>]*>\s*</p>',
            "[Awesome list badge](https://awesome.re/)",
            body,
        )
        body = re.sub(
            r"\[!\[Star History Chart\]\(https://api\.star-history\.com/[^)]+\)\]"
            r"\((https://star-history\.com/[^)]+)\)",
            r"[Open the live Star History chart](\1)",
            body,
        )
        body = re.sub(
            r'<p align="center">\s*<img width="600" src="\./imgs/tree\.jpg"\s*/>\s*</p>',
            image_embed("imgs/tree.jpg", "600"),
            body,
        )
        body = re.sub(
            r'<p align="center">\s*<img width="500" src="\./imgs/decision\.png"\s*/>\s*</p>',
            image_embed("imgs/decision.png", "500"),
            body,
        )
        for editable in EDITABLE_SOURCES:
            body = body.replace(
                f"[pptx](./{editable})",
                f"[pptx — pinned original]({source_url(editable)})",
            )
    elif relative == "awesome_examples/tableQA.md":
        for image in MEDIA_SOURCES:
            if not image.startswith("awesome_examples/imgs/"):
                continue
            local_reference = image.removeprefix("awesome_examples/")
            body = body.replace(
                f"![alt text]({local_reference})",
                image_embed(image),
            )
    return body.rstrip() + "\n"


def copy_binary(source_root: Path, root: Path, relative: str, destination: str) -> dict:
    source = source_root / relative
    target = root / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source_path": relative,
        "source_url": source_url(relative),
        "output": destination,
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
    }


def update_navigation(root: Path, records: list[dict]) -> None:
    navigation = root / "publishing/navigation.yml"
    text = navigation.read_text(encoding="utf-8")
    begin = "      # BEGIN GENERATED LLMS PRACTICAL GUIDE\n"
    end = "      # END GENERATED LLMS PRACTICAL GUIDE\n"
    block = begin + "".join(
        f"      - source: {record['output']}\n        route: {record['route']}\n"
        for record in records
    ) + end
    if begin in text:
        text = text[: text.index(begin)] + block + text[text.index(end) + len(end) :]
    else:
        marker = "      - source: 05 Источники/Source maps/LLMs Practical Guide — link map.md\n"
        position = text.find(marker)
        if position < 0:
            raise RuntimeError("Cannot find LLMs Practical Guide source-map route")
        text = text[:position] + block + text[position:]
    navigation.write_text(text, encoding="utf-8")


def write_index(root: Path, page_records: list[dict], media_records: list[dict]) -> dict:
    index = root / "05 Источники/Courses" / f"{COURSE_NAME}.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    page_links = "\n".join(
        f"- [[{record['output']}|{record['title']}]] — `{record['source_path']}`"
        for record in page_records
    )
    media_gallery = "\n\n".join(
        f"### `{record['source_path']}`\n\n"
        f"![[{record['output']}|700]]\n\n"
        f"[Pinned original]({record['source_url']}) · "
        f"`SHA-256 {record['sha256']}`"
        for record in media_records
    )
    editable_links = "\n".join(
        f"- [{Path(relative).name}]({source_url(relative)}) — pinned upstream; "
        f"an exact local copy is checksummed in `import-manifest.json`"
        for relative in EDITABLE_SOURCES
    )
    index.write_text(
        f"""---
title: "LLMs Practical Guide — complete source-native archive"
type: source-note
status: imported-source
language: en
source_commit: {COMMIT}
license_status: no-repository-license-found
last_verified: 2026-07-26
---

# LLMs Practical Guide — complete source-native archive

This archive preserves all substantive text and teaching media from
[`Mooler0410/LLMsPracticalGuide`]({REPOSITORY}/tree/{COMMIT}) at commit
`{COMMIT}`. The original-language pages are not summaries: the full curated
survey index, both worked model-behavior case studies, and the figure change
log are reproduced below.

No repository-wide license file exists at the pinned commit. The upstream
README asks users to cite the survey, but Bookvar does not reinterpret that
request as an explicit license. Every page and asset therefore carries a
pinned source URL and checksum; the copy is retained under the Bookvar owner's
confirmed permission for a noncommercial educational archive.

## Complete source pages

{page_links}

## Editable originals for the evolutionary-tree figures

{editable_links}

## Original figure gallery

{media_gallery}
""",
        encoding="utf-8",
    )
    return {
        "title": "Archive index and figure gallery",
        "source_path": "(Bookvar index)",
        "output": index.relative_to(root).as_posix(),
        "route": "sources/courses/llms-practical-guide",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--update-navigation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    verify_source(source_root)

    destination = root / "05 Источники/Courses" / COURSE_NAME
    destination.mkdir(parents=True, exist_ok=True)
    page_records: list[dict] = []
    for relative, metadata in MARKDOWN_SOURCES.items():
        source = source_root / relative
        digest = sha256(source)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        body = adapt_markdown(relative, source.read_text(encoding="utf-8"))
        output.write_text(
            frontmatter(relative, metadata, digest) + body,
            encoding="utf-8",
        )
        page_records.append(
            {
                "title": metadata["title"],
                "kind": metadata["kind"],
                "source_path": relative,
                "source_url": source_url(relative),
                "output": output.relative_to(root).as_posix(),
                "route": metadata["route"],
                "source_sha256": digest,
                "source_bytes": source.stat().st_size,
                "source_lines": len(source.read_text(encoding="utf-8").splitlines()),
                "source_words": len(re.findall(r"\b[\w'-]+\b", body, flags=re.UNICODE)),
            }
        )

    media_records = [
        copy_binary(
            source_root,
            root,
            relative,
            localized_asset_path(relative),
        )
        for relative in MEDIA_SOURCES
    ]
    editable_records = [
        copy_binary(
            source_root,
            root,
            relative,
            localized_editable_path(relative),
        )
        for relative in EDITABLE_SOURCES
    ]
    index_record = write_index(root, page_records, media_records)
    manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "license_status": "no-repository-license-found",
        "page_count": len(page_records),
        "route_count": len(page_records) + 1,
        "media_count": len(media_records),
        "editable_source_count": len(editable_records),
        "pages": page_records,
        "media": media_records,
        "editable_sources": editable_records,
    }
    (destination / "import-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.update_navigation:
        update_navigation(root, [index_record, *page_records])
    print(
        f"Imported {len(page_records)} complete pages, {len(media_records)} media "
        f"assets, and {len(editable_records)} editable originals from {COMMIT}."
    )


if __name__ == "__main__":
    main()
