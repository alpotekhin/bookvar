#!/usr/bin/env python3
"""Preserve TinyTorch's 20 progressive modules as source-native Bookvar pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


PINNED_COMMIT = "2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd"
REPOSITORY = "https://github.com/harvard-edge/cs249r_book"
CODE_LICENSE = "MIT"
CONTENT_LICENSE = "CC BY-NC-SA 4.0"


def fenced(value: str, language: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    delimiter = "`" * max(4, longest + 1)
    return f"{delimiter}{language}\n{value.rstrip()}\n{delimiter}"


def module_page(source_root: Path, module_dir: Path) -> tuple[str, dict[str, object]]:
    module_id = module_dir.name
    source_file = next(module_dir.glob("*.py"))
    metadata_file = module_dir / "module.yaml"
    source = source_file.read_text(encoding="utf-8")
    metadata_text = metadata_file.read_text(encoding="utf-8")
    metadata = yaml.safe_load(metadata_text) or {}
    title = str(metadata.get("title") or module_id.replace("_", " ").title())
    source_path = source_file.relative_to(source_root).as_posix()
    metadata_path = metadata_file.relative_to(source_root).as_posix()
    source_url = f"{REPOSITORY}/blob/{PINNED_COMMIT}/{source_path}"
    metadata_url = f"{REPOSITORY}/blob/{PINNED_COMMIT}/{metadata_path}"

    body = f"""---
title: "TinyTorch {module_id}: {title}"
type: external-resource
status: imported-source
language: en
source_kind: verbatim-source
source_commit: {PINNED_COMMIT}
---

> [!note] Complete original module
> This page preserves the complete TinyTorch module
> [`{source_path}`]({source_url}) and its
> [`module.yaml`]({metadata_url}) from commit `{PINNED_COMMIT}`.
> Software is licensed under [{CODE_LICENSE}]({REPOSITORY}/blob/{PINNED_COMMIT}/tinytorch/LICENSE);
> educational prose embedded in the module is licensed under
> [{CONTENT_LICENSE}]({REPOSITORY}/blob/{PINNED_COMMIT}/LICENSE.md).
> Bookvar added only this provenance wrapper and Markdown code fences.

## Module metadata

{fenced(metadata_text, "yaml")}

## Complete Python source

{fenced(source, "python")}
"""
    record = {
        "module": module_id,
        "title": title,
        "source_path": source_path,
        "metadata_path": metadata_path,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
        "source_commit": PINNED_COMMIT,
    }
    return body, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    source_root = args.source_root.resolve()
    tinytorch_root = source_root / "tinytorch"
    module_root = tinytorch_root / "src"
    output_root = (
        root / "05 Источники" / "Courses" / "Harvard ML Systems" / "tinytorch"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    module_dirs = sorted(
        (path for path in module_root.iterdir() if path.is_dir()),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    records = []
    links = []
    for module_dir in module_dirs:
        body, record = module_page(source_root, module_dir)
        output = output_root / f"{module_dir.name}.md"
        output.write_text(body, encoding="utf-8")
        records.append(record | {"output": output.relative_to(root).as_posix()})
        links.append(
            f"- [[05 Источники/Courses/Harvard ML Systems/tinytorch/"
            f"{module_dir.name}|{module_dir.name} — {record['title']}]]"
        )

    overview = f"""---
title: "TinyTorch — complete progressive course"
type: external-resource
status: imported-source
language: en
source_kind: verbatim-source
source_commit: {PINNED_COMMIT}
---

# TinyTorch — complete progressive course

TinyTorch builds a small machine-learning framework across twenty executable
modules: tensors, autograd and optimizers first; tokenization, attention and
Transformers next; then profiling, quantization, compression, acceleration,
memoization and benchmarking. The pages below preserve each complete Python
module and its original metadata rather than summarizing it.

Source: [Harvard TinyTorch]({REPOSITORY}/tree/{PINNED_COMMIT}/tinytorch) at
commit `{PINNED_COMMIT}`. Software license: [{CODE_LICENSE}]({REPOSITORY}/blob/{PINNED_COMMIT}/tinytorch/LICENSE).
Embedded educational prose: [{CONTENT_LICENSE}]({REPOSITORY}/blob/{PINNED_COMMIT}/LICENSE.md).

{chr(10).join(links)}
"""
    (output_root / "README.md").write_text(overview, encoding="utf-8")
    (output_root / "import-manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Imported {len(records)} TinyTorch modules into {output_root}")


if __name__ == "__main__":
    main()
