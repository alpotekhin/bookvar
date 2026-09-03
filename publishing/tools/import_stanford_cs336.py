#!/usr/bin/env python3
"""Freeze and audit the official Stanford CS336 Spring 2026 source layer.

The refresh path downloads complete GitHub repository archives at resolved
40-character commits. The check path is deliberately offline: it verifies
every archived file against the recorded SHA-256 inventory and validates the
derived manifest, source-unit, coverage, and visual ledgers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COURSE_ROOT = REPOSITORY_ROOT / "05 Источники/Courses/Stanford CS336 Spring 2026"
LECTURES_ROOT = COURSE_ROOT / "Lectures/repository"
ASSIGNMENTS_ROOT = COURSE_ROOT / "Assignments"
METADATA_ROOT = COURSE_ROOT / "Metadata"
LOCK_PATH = COURSE_ROOT / "snapshot-lock.json"
INVENTORY_PATH = COURSE_ROOT / "artifact-inventory.json"
HUB_PATH = COURSE_ROOT / "_index.md"
RETRIEVED_AT = "2026-09-04"
COURSE_URL = "https://cs336.stanford.edu/"
GITHUB_ORG = "stanford-cs336"
USER_PERMISSION = (
    "User confirmed open educational reuse on 2026-09-04 for local educational "
    "preservation and later attributed textbook reuse; this is a permission "
    "record, not a named license. Explicit repository license files are preserved."
)
SOURCE_ONLY_REASON = (
    "Baseline source audit only: this unit is preserved and indexed, but no "
    "specific canonical textbook heading with a reciprocal source_unit_id has "
    "yet been verified."
)


@dataclass(frozen=True)
class Repository:
    name: str
    destination: Path

    @property
    def url(self) -> str:
        return f"https://github.com/{GITHUB_ORG}/{self.name}.git"


REPOSITORIES = (
    Repository("lectures", LECTURES_ROOT),
    Repository("assignment1-basics", ASSIGNMENTS_ROOT / "assignment1-basics"),
    Repository("assignment2-systems", ASSIGNMENTS_ROOT / "assignment2-systems"),
    Repository("assignment3-scaling", ASSIGNMENTS_ROOT / "assignment3-scaling"),
    Repository("assignment4-data", ASSIGNMENTS_ROOT / "assignment4-data"),
    Repository("assignment5-alignment", ASSIGNMENTS_ROOT / "assignment5-alignment"),
    Repository("stanford-cs336.github.io", METADATA_ROOT / "site-repository"),
)

SCHEDULE = (
    (1, "2026-03-30", "Overview, tokenization", "Percy Liang", "lecture_01.py"),
    (
        2,
        "2026-04-01",
        "PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)",
        "Percy Liang",
        "lecture_02.py",
    ),
    (3, "2026-04-06", "Architectures, hyperparameters", "Tatsunori Hashimoto", "lecture_03.pdf"),
    (
        4,
        "2026-04-08",
        "Attention alternatives and mixture of experts",
        "Tatsunori Hashimoto",
        "lecture_04.pdf",
    ),
    (5, "2026-04-13", "GPUs, TPUs", "Tatsunori Hashimoto", "lecture_05.pdf"),
    (6, "2026-04-15", "Kernels, Triton", "Percy Liang", "lecture_06.py"),
    (7, "2026-04-20", "Parallelism", "Percy Liang", "lecture_07.py"),
    (8, "2026-04-22", "Parallelism", "Tatsunori Hashimoto", "lecture_08.pdf"),
    (9, "2026-04-27", "Scaling laws", "Tatsunori Hashimoto", "lecture_09.pdf"),
    (10, "2026-04-29", "Inference", "Percy Liang", "lecture_10.py"),
    (11, "2026-05-04", "Scaling laws", "Tatsunori Hashimoto", "lecture_11.pdf"),
    (12, "2026-05-06", "Evaluation", "Percy Liang", "lecture_12.py"),
    (13, "2026-05-11", "Data (sources, datasets)", "Percy Liang", "lecture_13.py"),
    (
        14,
        "2026-05-13",
        "Data (filtering, deduplication, mixing, synthetic data)",
        "Percy Liang",
        "lecture_14.py",
    ),
    (
        15,
        "2026-05-18",
        "Mid/post-training (SFT/RLHF)",
        "Tatsunori Hashimoto",
        "lecture_15.pdf",
    ),
    (16, "2026-05-20", "Post-training - RLVR", "Tatsunori Hashimoto", "lecture_16.pdf"),
    (17, "2026-05-27", "Alignment - multimodality", "Percy Liang", "lecture_17.py"),
    (18, "2026-06-01", "Guest lecture: Daniel Selsam", "Daniel Selsam", None),
    (19, "2026-06-03", "Guest lecture: Dan Fu", "Dan Fu", None),
)

ASSIGNMENT_META = (
    (1, "assignment1-basics", "Basics", "2026-03-30"),
    (2, "assignment2-systems", "Systems", "2026-04-15"),
    (3, "assignment3-scaling", "Scaling", "2026-04-29"),
    (4, "assignment4-data", "Data", "2026-05-06"),
    (5, "assignment5-alignment", "Alignment and Reasoning RL", "2026-05-20"),
)

LECTURE_DEPENDENCIES = (
    "README.md",
    "references.py",
    "facts.py",
    "lecture_util.py",
    "pyproject.toml",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "images",
    "assets",
    "var/traces",
)

GENERATED_LEDGER_FILES = (
    "source-manifest.yml",
    "source-units.yml",
    "coverage.yml",
    "visuals.yml",
)


def direct_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        env.pop(key, None)
    return env


def direct_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(COURSE_ROOT).as_posix()


def repository_sha(repository: Repository) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repository.url, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=direct_environment(),
    )
    sha = result.stdout.split()[0]
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError(f"{repository.name}: invalid upstream SHA {sha!r}")
    return sha


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Bookvar-CS336-source-audit/1.0"},
    )
    with direct_opener().open(request, timeout=120) as response:
        return response.read()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def replace_repository(repository: Repository, sha: str) -> dict[str, Any]:
    archive_url = f"https://codeload.github.com/{GITHUB_ORG}/{repository.name}/tar.gz/{sha}"
    payload = download(archive_url)
    with tempfile.TemporaryDirectory(prefix=f"cs336-{repository.name}-") as temporary:
        temporary_root = Path(temporary)
        archive_path = temporary_root / "archive.tar.gz"
        archive_path.write_bytes(payload)
        unpacked = temporary_root / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, unpacked)
        roots = [path for path in unpacked.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"{repository.name}: expected one archive root")
        if repository.destination.exists():
            shutil.rmtree(repository.destination)
        repository.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0]), repository.destination)
    return {
        "repository": repository.name,
        "canonical_url": f"https://github.com/{GITHUB_ORG}/{repository.name}/tree/{sha}",
        "revision": sha,
        "archive_url": archive_url,
        "archive_sha256": sha256_bytes(payload),
    }


def pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"cannot determine page count: {path}")
    return int(match.group(1))


def pdf_pages(path: Path) -> list[str]:
    count = pdf_page_count(path)
    pages: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cs336-pdf-") as temporary:
        for page in range(1, count + 1):
            output = Path(temporary) / f"{page}.txt"
            subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-layout",
                    str(path),
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            pages.append(output.read_text("utf-8", errors="replace"))
    return pages


def clean_title(value: str, fallback: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" #\t\r\n:-")
    return value[:180] if value else fallback


def literal_string(call: ast.Call, index: int = 0) -> str | None:
    if len(call.args) <= index:
        return None
    value = call.args[index]
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def python_lecture_events(path: Path, sha: str) -> list[dict[str, Any]]:
    source = path.read_text("utf-8")
    tree = ast.parse(source)
    events: list[dict[str, Any]] = []
    rendered_calls = {"text", "image", "link", "code", "table", "plot", "bar", "line"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            events.append(
                {
                    "line": node.lineno,
                    "column": node.col_offset,
                    "kind": "section-boundary",
                    "title": clean_title(node.name.replace("_", " "), "Section"),
                    "heading_level": 2,
                }
            )
        elif isinstance(node, ast.Call) and call_name(node) in rendered_calls:
            name = call_name(node)
            literal = literal_string(node)
            source_text = ast.get_source_segment(source, node) or f"{name}()"
            if name == "image":
                asset = literal or f"unresolved-event-{node.lineno}"
                asset_url = (
                    f"https://raw.githubusercontent.com/{GITHUB_ORG}/lectures/"
                    f"{sha}/{asset.lstrip('/')}"
                    if literal
                    else f"https://github.com/{GITHUB_ORG}/lectures/blob/{sha}/{path.name}#L{node.lineno}"
                )
                events.append(
                    {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "kind": "rendered-image",
                        "title": clean_title(Path(asset).name, f"Image event at line {node.lineno}"),
                        "asset_url": asset_url,
                    }
                )
            elif name == "link":
                target = literal
                if target is None:
                    for keyword in node.keywords:
                        if keyword.arg == "url" and isinstance(keyword.value, ast.Constant):
                            target = keyword.value.value
                if not isinstance(target, str) or not re.match(r"https?://", target):
                    target = f"https://github.com/{GITHUB_ORG}/lectures/blob/{sha}/{path.name}#L{node.lineno}"
                events.append(
                    {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "kind": "rendered-link",
                        "title": clean_title(literal or source_text, f"Link at line {node.lineno}"),
                        "target_url": target,
                    }
                )
            else:
                title = clean_title(literal or source_text, f"Rendered event at line {node.lineno}")
                events.append(
                    {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "kind": "rendered-text",
                        "title": title,
                        "content_sha256": sha256_bytes(source_text.encode("utf-8")),
                        "visual_event": name in {"code", "table", "plot", "bar", "line"},
                    }
                )
    return sorted(events, key=lambda event: (event["line"], event["column"], event["kind"]))


def trace_path(filename: str) -> Path:
    return LECTURES_ROOT / "var/traces" / f"{Path(filename).stem}.json"


def trace_document(filename: str) -> dict[str, Any]:
    path = trace_path(filename)
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise RuntimeError(f"invalid executable trace: {path}")
    return value


def trace_renderings(filename: str, sha: str) -> list[dict[str, Any]]:
    trace = trace_document(filename)
    events: list[dict[str, Any]] = []
    for step_number, step in enumerate(trace["steps"], start=1):
        if not isinstance(step, dict):
            continue
        stack = step.get("stack", [])
        source_frame = stack[-1] if isinstance(stack, list) and stack else {}
        source_line = source_frame.get("line_number") if isinstance(source_frame, dict) else None
        renderings = step.get("renderings", [])
        if not isinstance(renderings, list):
            continue
        for rendering_number, rendering in enumerate(renderings, start=1):
            if not isinstance(rendering, dict):
                continue
            rendering_type = rendering.get("type")
            data = rendering.get("data")
            location = (
                f"{relative(trace_path(filename))}#step={step_number}:"
                f"rendering={rendering_number}"
            )
            if rendering_type == "markdown":
                text_value = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
                heading = re.match(r"^(#{1,6})\s+(.+)", text_value)
                if heading:
                    events.append(
                        {
                            "step": step_number,
                            "kind": "section-boundary",
                            "title": clean_title(heading.group(2), f"Trace heading {step_number}"),
                            "heading_level": len(heading.group(1)),
                            "source_location": location,
                        }
                    )
                else:
                    events.append(
                        {
                            "step": step_number,
                            "kind": "rendered-text",
                            "title": clean_title(text_value, f"Trace text {step_number}"),
                            "content_sha256": sha256_bytes(text_value.encode("utf-8")),
                            "source_location": location,
                        }
                    )
            elif rendering_type == "image":
                asset = data if isinstance(data, str) else f"unresolved-step-{step_number}"
                asset_url = (
                    f"https://raw.githubusercontent.com/{GITHUB_ORG}/lectures/"
                    f"{sha}/{asset.lstrip('/')}"
                    if isinstance(data, str)
                    else f"https://github.com/{GITHUB_ORG}/lectures/blob/{sha}/{filename}"
                )
                events.append(
                    {
                        "step": step_number,
                        "kind": "rendered-image",
                        "title": clean_title(Path(asset).name, f"Trace image {step_number}"),
                        "asset_url": asset_url,
                        "source_location": location,
                    }
                )
            elif rendering_type == "link":
                external = rendering.get("external_link")
                target = external.get("url") if isinstance(external, dict) else None
                if not isinstance(target, str) or not re.match(r"https?://", target):
                    target = (
                        f"https://github.com/{GITHUB_ORG}/lectures/blob/{sha}/"
                        f"{filename}#L{source_line or 1}"
                    )
                events.append(
                    {
                        "step": step_number,
                        "kind": "rendered-link",
                        "title": clean_title(
                            data if isinstance(data, str) else target,
                            f"Trace link {step_number}",
                        ),
                        "target_url": target,
                        "source_location": location,
                    }
                )
    return events


def pdf_embedded_images(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["pdfimages", "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    images: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        if fields[2] != "image":
            continue
        images.append(
            {
                "page": int(fields[0]),
                "number": int(fields[1]),
                "width": int(fields[3]),
                "height": int(fields[4]),
            }
        )
    return images


def first_page_title(page_text: str, fallback: str) -> str:
    for line in page_text.splitlines():
        candidate = clean_title(line, "")
        if candidate:
            return candidate
    return fallback


def repository_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def pinned_blob(repository: str, sha: str, path: str) -> str:
    return f"https://github.com/{GITHUB_ORG}/{repository}/blob/{sha}/{path}"


def source_manifest(lock: dict[str, Any]) -> dict[str, Any]:
    revisions = {item["repository"]: item["revision"] for item in lock["repositories"]}
    objects: list[dict[str, Any]] = []
    for number, meeting_date, title, lecturer, filename in SCHEDULE:
        if filename is None:
            continue
        path = LECTURES_ROOT / filename
        kind = "executable-lecture" if path.suffix == ".py" else "pdf"
        page_count = len(trace_document(filename)["steps"]) if kind == "executable-lecture" else pdf_page_count(path)
        objects.append(
            {
                "id": f"lecture-{number:02d}",
                "kind": kind,
                "source_authority": "official-course",
                "verification_status": "verified",
                "lecturer_or_author": lecturer,
                "meeting_date": meeting_date,
                "title": title,
                "canonical_url": pinned_blob("lectures", revisions["lectures"], filename),
                "revision_or_checksum": revisions["lectures"],
                "local_path": relative(path),
                "mime": "text/x-python" if kind == "executable-lecture" else "application/pdf",
                "bytes": path.stat().st_size,
                "page_count": max(1, page_count),
                "language": "en",
                "rights_status": "permission-recorded",
                "rights_evidence": USER_PERMISSION,
                "retrieved_at": RETRIEVED_AT,
            }
        )
    for number, repository, title, released in ASSIGNMENT_META:
        sha = revisions[repository]
        path = ASSIGNMENTS_ROOT / repository
        handouts = sorted(path.glob("*.pdf"))
        page_count = sum(pdf_page_count(pdf) for pdf in handouts) or 1
        objects.append(
            {
                "id": f"assignment-{number:02d}",
                "kind": "assignment",
                "source_authority": "official-course",
                "verification_status": "verified",
                "lecturer_or_author": "Stanford CS336 staff",
                "meeting_date": released,
                "title": f"Assignment {number}: {title}",
                "canonical_url": f"https://github.com/{GITHUB_ORG}/{repository}/tree/{sha}",
                "revision_or_checksum": sha,
                "local_path": relative(path),
                "mime": "application/vnd.git-repository",
                "bytes": repository_size(path),
                "page_count": max(1, page_count),
                "language": "en",
                "rights_status": "permission-recorded",
                "rights_evidence": USER_PERMISSION,
                "retrieved_at": RETRIEVED_AT,
            }
        )
    supplement = ASSIGNMENTS_ROOT / "assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf"
    assignment5_sha = revisions["assignment5-alignment"]
    objects.append(
        {
            "id": "assignment-05-safety-supplement",
            "kind": "assignment",
            "source_authority": "official-course",
            "verification_status": "verified",
            "lecturer_or_author": "Stanford CS336 staff",
            "meeting_date": "2026-05-20",
            "title": "Assignment 5 optional supplement: safety, instruction tuning, and RLHF",
            "canonical_url": pinned_blob(
                "assignment5-alignment",
                assignment5_sha,
                supplement.name,
            ),
            "revision_or_checksum": assignment5_sha,
            "local_path": relative(supplement),
            "mime": "application/pdf",
            "bytes": supplement.stat().st_size,
            "page_count": pdf_page_count(supplement),
            "language": "en",
            "rights_status": "permission-recorded",
            "rights_evidence": USER_PERMISSION,
            "retrieved_at": RETRIEVED_AT,
        }
    )
    integration_base = subprocess.run(
        ["git", "rev-parse", "0a1fbb8"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "course": "Stanford CS336: Language Modeling from Scratch",
        "offering": "Spring 2026",
        "integration_base_commit": integration_base,
        "retrieved_at": RETRIEVED_AT,
        "objects": objects,
    }


def make_unit(
    object_id: str,
    order: int,
    kind: str,
    source_location: str,
    title: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": f"{object_id}-unit-{order:04d}",
        "source_object": object_id,
        "order": order,
        "source_location": source_location,
        "kind": kind,
        "title": clean_title(title, f"{object_id} unit {order}"),
        **fields,
    }


def executable_units(
    number: int,
    filename: str,
    sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_id = f"lecture-{number:02d}"
    path = trace_path(filename)
    events = trace_renderings(filename, sha)
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for order, event in enumerate(events, start=1):
        location = event["source_location"]
        fields = {
            key: event[key]
            for key in ("heading_level", "content_sha256", "asset_url", "target_url")
            if key in event
        }
        units.append(make_unit(object_id, order, event["kind"], location, event["title"], **fields))
        if event["kind"] == "rendered-image" or event.get("visual_event"):
            visual_kind = "image" if event["kind"] == "rendered-image" else "rendered code/table trace"
            visuals.append(
                source_only_visual(
                    f"{object_id}-visual-{len(visuals) + 1:04d}",
                    object_id,
                    location,
                    [event["step"]],
                    f"What does this {visual_kind} contribute to {object_id}?",
                    [location],
                    object_id,
                    sha256_file(path),
                    "edtrace JSON event stream, rendering audit v1",
                )
            )
    return units, visuals


def pdf_units_and_visuals(
    object_id: str,
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    pages = pdf_pages(path)
    checksum = sha256_file(path)
    images_by_page: dict[int, list[dict[str, Any]]] = {}
    for image in pdf_embedded_images(path):
        images_by_page.setdefault(image["page"], []).append(image)
    for page_number, page_text in enumerate(pages, start=1):
        title = first_page_title(page_text, f"Page {page_number}")
        location = f"{relative(path)}#page={page_number}"
        units.append(
            make_unit(
                object_id,
                len(units) + 1,
                "page",
                location,
                title,
                page=page_number,
            )
        )
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-page-{page_number:04d}",
                object_id,
                location,
                [page_number],
                f"What visual argument, figure, table, derivation, or example appears on page {page_number}?",
                [location],
                object_id,
                checksum,
                "Poppler pdftotext/pdfinfo, page audit v1",
            )
        )
        for block_index, block in enumerate(re.split(r"\n\s*\n", page_text), start=1):
            line = clean_title(block.splitlines()[0] if block.splitlines() else "", "")
            if not line or len(line) > 140:
                continue
            if re.search(
                r"(?i)^(?:\d+(?:\.\d+)*\s+|example|experiment|failure|derivation|"
                r"evaluation|results?|summary|architecture|method|attention|training|data)",
                line,
            ):
                units.append(
                    make_unit(
                        object_id,
                        len(units) + 1,
                        "heading",
                        f"{location}:block={block_index}",
                        line,
                        page=page_number,
                        heading_level=2,
                    )
                )
        for image in images_by_page.get(page_number, []):
            image_location = f"{location}:image={image['number']}"
            units.append(
                make_unit(
                    object_id,
                    len(units) + 1,
                    "figure",
                    image_location,
                    (
                        f"Embedded image {image['number']} "
                        f"({image['width']}x{image['height']})"
                    ),
                    page=page_number,
                    figure_label=f"pdf-image-{image['number']}",
                )
            )
            visuals.append(
                source_only_visual(
                    f"{object_id}-visual-image-{image['number']:04d}",
                    object_id,
                    image_location,
                    [page_number],
                    f"What argument does embedded image {image['number']} support?",
                    [image_location],
                    object_id,
                    checksum,
                    "Poppler pdfimages, embedded-image audit v1",
                )
            )
    for order, unit in enumerate(units, start=1):
        unit["order"] = order
        unit["id"] = f"{object_id}-unit-{order:04d}"
    return units, visuals


def assignment_handout_paths(repository: str) -> list[Path]:
    root = ASSIGNMENTS_ROOT / repository
    if repository == "assignment5-alignment":
        return [
            path
            for path in sorted(root.glob("*.pdf"))
            if "supplement_safety_rlhf" not in path.name
        ]
    return sorted(root.glob("*.pdf"))


def assignment_units(
    number: int,
    repository: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_id = f"assignment-{number:02d}"
    root = ASSIGNMENTS_ROOT / repository
    records: list[tuple[str, str, str, dict[str, Any]]] = []
    visuals: list[dict[str, Any]] = []
    for handout in assignment_handout_paths(repository):
        checksum = sha256_file(handout)
        images_by_page: dict[int, list[dict[str, Any]]] = {}
        for image in pdf_embedded_images(handout):
            images_by_page.setdefault(image["page"], []).append(image)
        for page_number, page_text in enumerate(pdf_pages(handout), start=1):
            location = f"{relative(handout)}#page={page_number}"
            title = first_page_title(page_text, f"Handout page {page_number}")
            records.append(
                (
                    "task",
                    location,
                    title,
                    {"task_id": f"handout-page-{page_number}"},
                )
            )
            visuals.append(
                source_only_visual(
                    f"{object_id}-visual-{handout.stem}-page-{page_number:04d}",
                    object_id,
                    location,
                    [page_number],
                    f"What task, result, table, or experiment is specified on handout page {page_number}?",
                    [location],
                    object_id,
                    checksum,
                    "Poppler pdftotext/pdfinfo, page audit v1",
                )
            )
            for image in images_by_page.get(page_number, []):
                image_location = f"{location}:image={image['number']}"
                visuals.append(
                    source_only_visual(
                        (
                            f"{object_id}-visual-{handout.stem}-"
                            f"image-{image['number']:04d}"
                        ),
                        object_id,
                        image_location,
                        [page_number],
                        f"What assignment mechanism does embedded image {image['number']} specify?",
                        [image_location],
                        object_id,
                        checksum,
                        "Poppler pdfimages, embedded-image audit v1",
                    )
                )
            for block_index, block in enumerate(re.split(r"\n\s*\n", page_text), start=1):
                first = clean_title(block.splitlines()[0] if block.splitlines() else "", "")
                if not first or len(first) > 150:
                    continue
                if not re.search(
                    r"(?i)(problem|task|experiment|writeup|deliverable|submit|report|"
                    r"evaluation|leaderboard|reproduc|points?|benchmark|test)",
                    block,
                ):
                    continue
                kind = "task"
                field = {"task_id": f"handout-p{page_number}-b{block_index}"}
                if re.search(r"(?i)(writeup|deliverable|submit|report)", block):
                    kind = "deliverable"
                    field = {"deliverable_id": f"handout-p{page_number}-b{block_index}"}
                elif re.search(r"(?i)(evaluation|leaderboard|reproduc|points?|benchmark)", block):
                    kind = "evaluation-requirement"
                    field = {"criterion_id": f"handout-p{page_number}-b{block_index}"}
                records.append(
                    (
                        kind,
                        f"{location}:block={block_index}",
                        first,
                        field,
                    )
                )
    for adapter in sorted(root.rglob("adapters.py")):
        tree = ast.parse(adapter.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                records.append(
                    (
                        "task",
                        f"{relative(adapter)}#L{node.lineno}",
                        f"Implement {node.name}",
                        {"task_id": node.name},
                    )
                )
    for test_path in sorted(root.rglob("test*.py")):
        tree = ast.parse(test_path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                records.append(
                    (
                        "test-interface",
                        f"{relative(test_path)}#L{node.lineno}",
                        node.name.replace("_", " "),
                        {"interface_name": node.name},
                    )
                )
    readme = root / "README.md"
    if readme.exists():
        for line_number, line in enumerate(readme.read_text("utf-8").splitlines(), start=1):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                slug = re.sub(r"[^a-z0-9]+", "-", match.group(2).lower()).strip("-")
                records.append(
                    (
                        "deliverable",
                        f"{relative(readme)}#L{line_number}",
                        match.group(2),
                        {"deliverable_id": f"readme-{slug or line_number}"},
                    )
                )
    units = [
        make_unit(object_id, order, kind, location, title, **fields)
        for order, (kind, location, title, fields) in enumerate(records, start=1)
    ]
    return units, visuals


def safety_units_and_visuals() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_id = "assignment-05-safety-supplement"
    path = ASSIGNMENTS_ROOT / "assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf"
    records: list[tuple[str, str, str, dict[str, Any]]] = []
    visuals: list[dict[str, Any]] = []
    checksum = sha256_file(path)
    images_by_page: dict[int, list[dict[str, Any]]] = {}
    for image in pdf_embedded_images(path):
        images_by_page.setdefault(image["page"], []).append(image)
    for page_number, page_text in enumerate(pdf_pages(path), start=1):
        location = f"{relative(path)}#page={page_number}"
        records.append(
            (
                "task",
                location,
                first_page_title(page_text, f"Safety supplement page {page_number}"),
                {"task_id": f"safety-page-{page_number}"},
            )
        )
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-page-{page_number:04d}",
                object_id,
                location,
                [page_number],
                f"What safety/RLHF task, evaluation, or result appears on page {page_number}?",
                [location],
                object_id,
                checksum,
                "Poppler pdftotext/pdfinfo, page audit v1",
            )
        )
        for image in images_by_page.get(page_number, []):
            image_location = f"{location}:image={image['number']}"
            visuals.append(
                source_only_visual(
                    f"{object_id}-visual-image-{image['number']:04d}",
                    object_id,
                    image_location,
                    [page_number],
                    f"What safety/RLHF mechanism does embedded image {image['number']} specify?",
                    [image_location],
                    object_id,
                    checksum,
                    "Poppler pdfimages, embedded-image audit v1",
                )
            )
        for block_index, block in enumerate(re.split(r"\n\s*\n", page_text), start=1):
            first = clean_title(block.splitlines()[0] if block.splitlines() else "", "")
            if not first or len(first) > 150:
                continue
            if not re.search(
                r"(?i)(problem|task|experiment|writeup|deliverable|submit|report|"
                r"evaluation|reproduc|points?|benchmark|test|safety|DPO|RLHF|SFT)",
                block,
            ):
                continue
            kind = "task"
            fields: dict[str, Any] = {"task_id": f"safety-p{page_number}-b{block_index}"}
            if re.search(r"(?i)(writeup|deliverable|submit|report)", block):
                kind = "deliverable"
                fields = {"deliverable_id": f"safety-p{page_number}-b{block_index}"}
            elif re.search(r"(?i)(evaluation|reproduc|points?|benchmark)", block):
                kind = "evaluation-requirement"
                fields = {"criterion_id": f"safety-p{page_number}-b{block_index}"}
            records.append((kind, f"{location}:block={block_index}", first, fields))
    units = [
        make_unit(object_id, order, kind, location, title, **fields)
        for order, (kind, location, title, fields) in enumerate(records, start=1)
    ]
    return units, visuals


def source_only_visual(
    visual_id: str,
    object_id: str,
    source_location: str,
    locations: list[int],
    question: str,
    sequence_locations: list[str],
    anchor: str,
    parent_sha256: str,
    extraction_tool: str,
) -> dict[str, Any]:
    return {
        "id": visual_id,
        "source_object": object_id,
        "source_location": source_location,
        "source_pages": locations,
        "question": question,
        "sequence_members": [
            {"order": order, "source_location": location}
            for order, location in enumerate(sequence_locations, start=1)
        ],
        "disposition": "source-only",
        "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
        "destination_anchor": anchor,
        "reason": (
            "Baseline visual audit: the original is preserved at exact event/page "
            "scope; no canonical textbook placement has yet been reviewed."
        ),
        "attribution": "Stanford CS336 staff, Language Modeling from Scratch, Spring 2026",
        "rights_status": "permission-recorded",
        "rights_holder": "Stanford CS336 source authors and identified upstream asset owners",
        "rights_scope": "Local educational preservation and later attributed textbook reuse",
        "license_identifier": "User permission record (not a license)",
        "license_url": COURSE_URL,
        "rights_evidence": USER_PERMISSION,
        "parent_sha256": parent_sha256,
        "extraction_tool": extraction_tool,
        "rendered_route": "/sources/courses/stanford-cs336-spring-2026/",
        "desktop_evidence": "pending editorial integration; source archive inspected",
        "narrow_evidence": "pending editorial integration; source archive inspected",
        "reviewer": "Codex source audit",
        "checked_at": RETRIEVED_AT,
    }


def build_ledgers(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    revisions = {item["repository"]: item["revision"] for item in lock["repositories"]}
    manifest = source_manifest(lock)
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for number, _, _, _, filename in SCHEDULE:
        if filename is None:
            continue
        path = LECTURES_ROOT / filename
        if path.suffix == ".py":
            new_units, new_visuals = executable_units(number, filename, revisions["lectures"])
        else:
            new_units, new_visuals = pdf_units_and_visuals(f"lecture-{number:02d}", path)
        units.extend(new_units)
        visuals.extend(new_visuals)
    for number, repository, _, _ in ASSIGNMENT_META:
        new_units, new_visuals = assignment_units(number, repository)
        units.extend(new_units)
        visuals.extend(new_visuals)
    safety_units, safety_visuals = safety_units_and_visuals()
    units.extend(safety_units)
    visuals.extend(safety_visuals)
    coverage_rows = [
        {
            "id": f"coverage-{unit['id']}",
            "source_object": unit["source_object"],
            "source_unit": unit["id"],
            "source_location": unit["source_location"],
            "kind": unit["kind"],
            "title": unit["title"],
            "disposition": "source-only",
            "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
            "destination_anchor": unit["source_object"],
            "reason": SOURCE_ONLY_REASON,
            "primary_sources": [unit["source_object"]],
        }
        for unit in units
    ]
    return (
        manifest,
        {"schema_version": 1, "units": units},
        {"schema_version": 1, "rows": coverage_rows},
        {
            "schema_version": 1,
            "audit_method": (
                "Executable lectures: Python AST event inventory. PDFs and assignment "
                "handouts: Poppler page-by-page text/page audit. Each row is one event "
                "or page, never a whole deck."
            ),
            "rows": visuals,
        },
    )


def write_json_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def artifact_inventory() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for root in (LECTURES_ROOT, ASSIGNMENTS_ROOT, METADATA_ROOT):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": relative(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {"schema_version": 1, "generated_at": RETRIEVED_AT, "files": files}


def hub_markdown(lock: dict[str, Any], manifest: dict[str, Any], units: dict[str, Any], visuals: dict[str, Any]) -> str:
    revisions = {item["repository"]: item["revision"] for item in lock["repositories"]}
    objects = {item["id"]: item for item in manifest["objects"]}
    lines = [
        "---",
        "title: Stanford CS336 — Language Modeling from Scratch, Spring 2026",
        "type: source-note",
        "status: verified",
        f"last_verified: {RETRIEVED_AT}",
        "---",
        "",
        "# Stanford CS336 — Language Modeling from Scratch, Spring 2026",
        "",
        "Это закреплённый source layer официального курса, а не основной учебный маршрут.",
        "Последовательное изучение идёт по каноническим главам учебника; здесь сохранены",
        "оригинальные англоязычные материалы, их dependency closure и аудит границ.",
        "",
        "Статус: **inventory complete; editorial integration pending**. Все извлечённые",
        "units пока имеют disposition `source-only`: для них ещё не подтверждены конкретные",
        "destination headings с reciprocal `source_unit_id`.",
        "",
        "## Реестры аудита",
        "",
        "- [source-manifest.yml](source-manifest.yml) — объекты и pinned revisions;",
        "- [source-units.yml](source-units.yml) — extraction index;",
        "- [coverage.yml](coverage.yml) — одна строка покрытия на каждый source unit;",
        "- [visuals.yml](visuals.yml) — event/page-level visual ledger;",
        "- [artifact-inventory.json](artifact-inventory.json) — SHA-256 каждого файла;",
        "- [snapshot-lock.json](snapshot-lock.json) — SHA репозиториев и архивов.",
        "",
        "## Pinned revisions",
        "",
    ]
    for repository in REPOSITORIES:
        lines.append(
            f"- `{repository.name}`: "
            f"[`{revisions[repository.name]}`]"
            f"(https://github.com/{GITHUB_ORG}/{repository.name}/tree/{revisions[repository.name]})"
        )
    lines.extend(["", "## 19 встреч курса", "", "| № | Дата | Тема | Артефакт |", "|---:|---|---|---|"])
    for number, meeting_date, title, lecturer, filename in SCHEDULE:
        if filename:
            artifact = f"[`{filename}`](Lectures/repository/{filename})"
        else:
            artifact = "официальный артефакт не опубликован; слот зафиксирован явно"
        lines.append(f"| {number} | {meeting_date} | {title} — {lecturer} | {artifact} |")
    lines.extend(["", "## Assignments 1–5", ""])
    for number, repository, title, _ in ASSIGNMENT_META:
        lines.append(
            f"- Assignment {number}: [{title}](Assignments/{repository}/README.md), "
            f"pinned at `{revisions[repository]}`."
        )
    lines.extend(
        [
            "- Assignment 5 optional branch: "
            "[safety, instruction tuning, and RLHF supplement]"
            "(Assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf). "
            "Это объект внутри pinned `assignment5-alignment`, а не шестой репозиторий.",
            "",
            "## Объекты source layer",
            "",
        ]
    )
    unit_counts: dict[str, int] = {}
    visual_counts: dict[str, int] = {}
    for unit in units["units"]:
        unit_counts[unit["source_object"]] = unit_counts.get(unit["source_object"], 0) + 1
    for row in visuals["rows"]:
        visual_counts[row["source_object"]] = visual_counts.get(row["source_object"], 0) + 1
    for object_id, source_object in objects.items():
        lines.extend(
            [
                f'<a id="{object_id}"></a>',
                f"### {object_id}: {source_object['title']}",
                "",
                f"- local path: `{source_object['local_path']}`;",
                f"- extracted units: {unit_counts.get(object_id, 0)};",
                f"- visual/event-page rows: {visual_counts.get(object_id, 0)};",
                "- baseline disposition: `source-only` — требуется последующая редакционная интеграция.",
                "",
            ]
        )
    lines.extend(
        [
            "## Права и атрибуция",
            "",
            "Пользователь подтвердил открытое образовательное переиспользование для локального",
            "сохранения и последующего атрибутированного переноса. Это зафиксировано как",
            "`rights_status: permission-recorded`, а не как название лицензии. Явные",
            "LICENSE-файлы репозиториев сохранены в оригинальных snapshot directories;",
            "права на заимствованные upstream figures всё равно проверяются пообъектно.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh() -> None:
    COURSE_ROOT.mkdir(parents=True, exist_ok=True)
    repositories: list[dict[str, Any]] = []
    for repository in REPOSITORIES:
        sha = repository_sha(repository)
        print(f"refresh: {repository.name} @ {sha}")
        repositories.append(replace_repository(repository, sha))
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    course_html = (METADATA_ROOT / "site-repository/index.html").read_bytes()
    (METADATA_ROOT / "course.html").write_bytes(course_html)
    lock = {
        "schema_version": 1,
        "course": "Stanford CS336 Spring 2026",
        "retrieved_at": RETRIEVED_AT,
        "course_page": {
            "url": COURSE_URL,
            "local_path": "Metadata/course.html",
            "sha256": sha256_bytes(course_html),
        },
        "repositories": repositories,
        "guest_slots": [
            {
                "meeting": number,
                "date": meeting_date,
                "title": title,
                "artifact_status": "official artifact unavailable",
                "source": COURSE_URL,
            }
            for number, meeting_date, title, _, filename in SCHEDULE
            if filename is None
        ],
        "rights_record": {
            "status": "permission-recorded",
            "scope": "local educational preservation and later attributed textbook reuse",
            "evidence": USER_PERMISSION,
        },
    }
    write_json_yaml(LOCK_PATH, lock)
    inventory = artifact_inventory()
    write_json_yaml(INVENTORY_PATH, inventory)
    manifest, units, coverage, visuals = build_ledgers(lock)
    for filename, value in zip(
        GENERATED_LEDGER_FILES,
        (manifest, units, coverage, visuals),
        strict=True,
    ):
        write_json_yaml(COURSE_ROOT / filename, value)
    HUB_PATH.write_text(hub_markdown(lock, manifest, units, visuals), "utf-8")
    check()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def read_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"{relative(path)}: {error}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{relative(path)}: expected mapping")
        return {}
    return value


def check() -> None:
    failures: list[str] = []
    for path in (LOCK_PATH, INVENTORY_PATH, HUB_PATH):
        require(path.exists(), f"missing required file: {relative(path)}", failures)
    for filename in GENERATED_LEDGER_FILES:
        require((COURSE_ROOT / filename).exists(), f"missing required ledger: {filename}", failures)
    if failures:
        raise RuntimeError("\n".join(failures))
    lock = read_json(LOCK_PATH, failures)
    inventory = read_json(INVENTORY_PATH, failures)
    manifest = read_json(COURSE_ROOT / "source-manifest.yml", failures)
    source_units = read_json(COURSE_ROOT / "source-units.yml", failures)
    coverage = read_json(COURSE_ROOT / "coverage.yml", failures)
    visuals = read_json(COURSE_ROOT / "visuals.yml", failures)
    revisions = {
        item.get("repository"): item.get("revision")
        for item in lock.get("repositories", [])
        if isinstance(item, dict)
    }
    require(set(revisions) == {repository.name for repository in REPOSITORIES}, "repository lock set is incomplete", failures)
    for repository, revision in revisions.items():
        require(
            isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}", revision) is not None,
            f"{repository}: invalid locked revision",
            failures,
        )
    for number, _, _, _, filename in SCHEDULE:
        if filename:
            require((LECTURES_ROOT / filename).exists(), f"missing lecture {number}: {filename}", failures)
    for dependency in LECTURE_DEPENDENCIES:
        require((LECTURES_ROOT / dependency).exists(), f"missing lecture dependency: {dependency}", failures)
    for _, repository, _, _ in ASSIGNMENT_META:
        require((ASSIGNMENTS_ROOT / repository).is_dir(), f"missing assignment repository: {repository}", failures)
    supplement = ASSIGNMENTS_ROOT / "assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf"
    require(supplement.exists(), "missing Assignment 5 optional safety/RLHF supplement", failures)
    guest_slots = lock.get("guest_slots", [])
    require(
        [slot.get("meeting") for slot in guest_slots if isinstance(slot, dict)] == [18, 19],
        "guest slots 18-19 must be explicit",
        failures,
    )
    inventory_rows = inventory.get("files", [])
    expected_paths: set[str] = set()
    for row in inventory_rows if isinstance(inventory_rows, list) else []:
        if not isinstance(row, dict):
            failures.append("artifact inventory row must be a mapping")
            continue
        path_value = row.get("path")
        if not isinstance(path_value, str):
            failures.append("artifact inventory path is missing")
            continue
        expected_paths.add(path_value)
        path = COURSE_ROOT / path_value
        if not path.exists():
            failures.append(f"missing archived artifact: {path_value}")
            continue
        require(path.stat().st_size == row.get("bytes"), f"size mismatch: {path_value}", failures)
        require(sha256_file(path) == row.get("sha256"), f"SHA-256 mismatch: {path_value}", failures)
    actual_paths = {
        relative(path)
        for root in (LECTURES_ROOT, ASSIGNMENTS_ROOT, METADATA_ROOT)
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual_paths == expected_paths, "artifact inventory does not match archived file set", failures)
    objects = manifest.get("objects", [])
    object_ids = {item.get("id") for item in objects if isinstance(item, dict)}
    require(len([value for value in object_ids if isinstance(value, str) and value.startswith("lecture-")]) == 17, "manifest must contain 17 lectures", failures)
    require(all(f"assignment-{number:02d}" in object_ids for number in range(1, 6)), "manifest must contain Assignments 1-5", failures)
    require("assignment-05-safety-supplement" in object_ids, "manifest must contain the Assignment 5 safety supplement object", failures)
    for item in objects if isinstance(objects, list) else []:
        if not isinstance(item, dict):
            continue
        url = item.get("canonical_url")
        if isinstance(url, str) and "github" in url:
            require(
                re.search(r"/(?:blob|tree)/[a-f0-9]{40}/?", url) is not None,
                f"unpinned GitHub object URL: {url}",
                failures,
            )
    units = source_units.get("units", [])
    rows = coverage.get("rows", [])
    unit_ids = [item.get("id") for item in units if isinstance(item, dict)]
    coverage_ids = [item.get("source_unit") for item in rows if isinstance(item, dict)]
    require(bool(units), "no extraction units were generated", failures)
    require(len(unit_ids) == len(set(unit_ids)), "duplicate extraction unit ids", failures)
    require(len(coverage_ids) == len(set(coverage_ids)), "duplicate coverage source_unit ids", failures)
    require(set(unit_ids) == set(coverage_ids), "every source unit must have exactly one coverage row", failures)
    units_by_object: dict[str, int] = {}
    for unit in units if isinstance(units, list) else []:
        if isinstance(unit, dict) and isinstance(unit.get("source_object"), str):
            object_id = unit["source_object"]
            units_by_object[object_id] = units_by_object.get(object_id, 0) + 1
    for object_id in object_ids:
        if isinstance(object_id, str):
            require(units_by_object.get(object_id, 0) > 0, f"no extracted units for {object_id}", failures)
    visual_rows = visuals.get("rows", [])
    visual_ids = [
        row.get("id") for row in visual_rows if isinstance(row, dict)
    ]
    require(
        len(visual_ids) == len(set(visual_ids)),
        "duplicate visual row ids",
        failures,
    )
    for row in visual_rows if isinstance(visual_rows, list) else []:
        if isinstance(row, dict):
            require(row.get("source_object") in object_ids, f"orphan visual row: {row.get('id')}", failures)
            require(
                isinstance(row.get("source_pages"), list) and bool(row.get("source_pages")),
                f"visual row lacks exact event/page scope: {row.get('id')}",
                failures,
            )
    hub = HUB_PATH.read_text("utf-8")
    require("## 19 встреч курса" in hub, "hub does not expose all 19 meetings", failures)
    require("## Реестры аудита" in hub, "hub does not expose the ledgers", failures)
    require("inventory complete; editorial integration pending" in hub, "hub status is missing", failures)
    for meeting in (18, 19):
        require(f"| {meeting} |" in hub, f"hub is missing guest slot {meeting}", failures)
    if failures:
        raise RuntimeError("Stanford CS336 snapshot check failed:\n- " + "\n- ".join(failures))
    print(
        "Stanford CS336 snapshot OK: "
        f"17 lectures, 2 guest slots, 5 assignments + safety supplement, "
        f"{len(inventory_rows)} artifacts, {len(units)} units, "
        f"{len(coverage_ids)} coverage rows, {len(visual_rows)} visual rows."
    )


def check_upstream_drift() -> None:
    failures: list[str] = []
    lock = read_json(LOCK_PATH, failures)
    if failures:
        raise RuntimeError("\n".join(failures))
    locked = {
        item["repository"]: item["revision"]
        for item in lock.get("repositories", [])
        if isinstance(item, dict)
    }
    drift: list[str] = []
    for repository in REPOSITORIES:
        upstream = repository_sha(repository)
        current = locked.get(repository.name)
        status = "unchanged" if upstream == current else f"drifted from {current}"
        print(f"{repository.name}: {upstream} ({status})")
        if upstream != current:
            drift.append(repository.name)
    site_sha = repository_sha(
        Repository("stanford-cs336.github.io", METADATA_ROOT / "site-repository")
    )
    course_hash = sha256_bytes(
        download(
            "https://raw.githubusercontent.com/"
            f"{GITHUB_ORG}/stanford-cs336.github.io/{site_sha}/index.html"
        )
    )
    locked_course_hash = lock.get("course_page", {}).get("sha256")
    if course_hash != locked_course_hash:
        drift.append("course-page")
        print(f"course-page: {course_hash} (drifted from {locked_course_hash})")
    else:
        print(f"course-page: {course_hash} (unchanged)")
    if drift:
        raise RuntimeError(
            "Upstream drift detected; the frozen Spring 2026 snapshot remains unchanged: "
            + ", ".join(drift)
        )
    print("No upstream drift detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--refresh", action="store_true", help="download and regenerate the frozen source layer")
    modes.add_argument("--check", action="store_true", help="validate the frozen source layer without network access")
    modes.add_argument(
        "--check-upstream-drift",
        action="store_true",
        help="compare current official heads/course page with the lock without changing it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.refresh:
            refresh()
        elif args.check:
            check()
        else:
            check_upstream_drift()
    except (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
