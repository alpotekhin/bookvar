#!/usr/bin/env python3
"""Import the complete D2L recommender chapter and selected LF Recommenders labs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from import_additional_sources import Source, import_notebook


D2L_COMMIT = "23d7a5aecceee57d1292c56e90cce307f183bb0a"
LF_COMMIT = "6232b154548c955315650d58dca6bf1411c56020"

D2L = Source(
    key="d2l-recommenders",
    title="Dive into Deep Learning — Recommender Systems",
    repository="https://github.com/d2l-ai/d2l-en",
    commit=D2L_COMMIT,
    license_name="CC BY-SA 4.0; sample code under modified MIT",
    license_path="LICENSE-SUMMARY",
    output_name="Dive into Deep Learning — Recommender Systems",
)

LF = Source(
    key="lf-recommenders",
    title="Linux Foundation Recommenders",
    repository="https://github.com/recommenders-team/recommenders",
    commit=LF_COMMIT,
    license_name="MIT",
    license_path="LICENSE",
    output_name="Linux Foundation Recommenders",
)

D2L_FILES = (
    "index.md",
    "recsys-intro.md",
    "movielens.md",
    "mf.md",
    "autorec.md",
    "ranking.md",
    "neumf.md",
    "seqrec.md",
    "ctr.md",
    "fm.md",
    "deepfm.md",
)

LF_NOTEBOOKS = (
    "examples/01_prepare_data/data_split.ipynb",
    "examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb",
    "examples/02_model_collaborative_filtering/sar_deep_dive.ipynb",
    "examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb",
    "examples/00_quick_start/sasrec_amazon.ipynb",
    "examples/03_evaluate/evaluation.ipynb",
    "examples/03_evaluate/als_movielens_diversity_metrics.ipynb",
    "examples/05_operationalize/als_movie_o16n.ipynb",
    "examples/06_benchmarks/movielens.ipynb",
)


def d2l_frontmatter(title: str, source_path: str) -> str:
    source_url = f"{D2L.repository}/blob/{D2L.commit}/{source_path}"
    return f"""---
title: "{title.replace('"', '\\"')}"
type: external-resource
status: imported-source
language: en
source_kind: book-chapter
source_commit: {D2L.commit}
---

> [!note] Complete original chapter
> This page preserves the complete English source
> [`{source_path}`]({source_url}) from *Dive into Deep Learning* at commit
> `{D2L.commit}`. Documentation is licensed under
> [CC BY-SA 4.0]({D2L.license_url}); sample code uses the repository's
> [modified MIT license]({D2L.repository}/blob/{D2L.commit}/LICENSE-SAMPLECODE).
> Bookvar changed only publication directives and localized the original figures.

"""


def adapt_d2l_markup(body: str) -> str:
    body = re.sub(r"^:label:`[^`]+`\s*$", "", body, flags=re.M)
    body = re.sub(r"^:(?:begin|end)_tab:`?[^`\n]*`?\s*$", "", body, flags=re.M)
    body = re.sub(r"^#@tab\s+.*$", "", body, flags=re.M)
    body = re.sub(r":numref:`([^`]+)`", r"`\1`", body)
    body = re.sub(
        r":cite:`([^`]+)`",
        lambda m: f"[{m.group(1)}]({D2L.repository}/blob/{D2L.commit}/d2l.bib)",
        body,
    )
    body = re.sub(r"```\{\.([A-Za-z0-9_+-]+)[^}]*\}", r"```\1", body)
    return re.sub(r"\n{4,}", "\n\n\n", body)


def import_d2l(root: Path, source_root: Path) -> list[dict[str, object]]:
    chapter = source_root / "chapter_recommender-systems"
    destination_root = root / "05 Источники/Courses" / D2L.output_name
    asset_root = root / "Assets/Sources" / D2L.output_name
    destination_root.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name in D2L_FILES:
        path = chapter / name
        raw = path.read_text(encoding="utf-8")
        heading = re.search(r"^#\s+(.+)$", raw, flags=re.M)
        title = heading.group(1).strip() if heading else path.stem
        body = re.sub(r"^#\s+.+?(?:\r?\n)+", "", raw, count=1)
        for image in sorted(set(re.findall(r"\.\./img/([^ )]+)", body))):
            src = source_root / "img" / image
            dst = asset_root / image
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            body = body.replace(
                f"../img/{image}",
                f"Assets/Sources/{D2L.output_name}/{image}",
            )
            body = re.sub(
                rf"!\[([^\]]*)\]\(Assets/Sources/{re.escape(D2L.output_name)}/{re.escape(image)}\)",
                f"![[Assets/Sources/{D2L.output_name}/{image}]]",
                body,
            )
        body = adapt_d2l_markup(body)
        output = destination_root / name
        output.write_text(d2l_frontmatter(title, f"chapter_recommender-systems/{name}") + body.strip() + "\n", encoding="utf-8")
        records.append(
            {
                "source_path": f"chapter_recommender-systems/{name}",
                "output": output.relative_to(root).as_posix(),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


def landing_pages(root: Path) -> None:
    d2l_root = root / "05 Источники/Courses" / D2L.output_name
    links = "\n".join(
        f"- [[05 Источники/Courses/{D2L.output_name}/{Path(name).stem}|{Path(name).stem}]]"
        for name in D2L_FILES
    )
    (root / "05 Источники/Courses" / f"{D2L.output_name}.md").write_text(
        f"""---
title: Dive into Deep Learning — complete recommender-systems chapter
type: source-note
status: imported-source
last_verified: 2026-07-24
---

# Dive into Deep Learning — complete recommender-systems chapter

All eleven English source chapters are preserved at commit
[`{D2L.commit}`]({D2L.repository}/commit/{D2L.commit}). They retain the
original exposition, formulas, runnable MXNet code, exercises, and seven
source figures. Documentation is CC BY-SA 4.0; sample code uses D2L's modified
MIT license.

{links}
""",
        encoding="utf-8",
    )
    lf_links = "\n".join(
        f"- [[05 Источники/Courses/{LF.output_name}/{path}.md|{Path(path).stem}]]"
        for path in LF_NOTEBOOKS
    )
    (root / "05 Источники/Courses" / f"{LF.output_name}.md").write_text(
        f"""---
title: Linux Foundation Recommenders — practical labs
type: source-note
status: imported-source
last_verified: 2026-07-24
---

# Linux Foundation Recommenders — practical labs

These are nine complete notebooks from the Linux Foundation AI & Data
Recommenders project at commit
[`{LF.commit}`]({LF.repository}/commit/{LF.commit}). Markdown cells, code,
saved outputs, and embedded figures are preserved. The repository is MIT
licensed.

The selection forms a practical route rather than a model zoo: split data,
establish a baseline, train collaborative and neural recommenders, evaluate
accuracy and beyond-accuracy metrics, benchmark alternatives, then
operationalize a model.

{lf_links}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d2l-root", type=Path, required=True)
    parser.add_argument("--lf-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    d2l_records = import_d2l(root, args.d2l_root.resolve())
    lf_records = [
        import_notebook(root, args.lf_root.resolve(), LF, args.lf_root.resolve() / relative)
        for relative in LF_NOTEBOOKS
    ]
    for source, records in ((D2L, d2l_records), (LF, lf_records)):
        manifest = root / "05 Источники/Courses" / source.output_name / "import-manifest.json"
        manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    landing_pages(root)
    print(f"D2L: {len(d2l_records)} chapters; LF Recommenders: {len(lf_records)} notebooks")


if __name__ == "__main__":
    main()
