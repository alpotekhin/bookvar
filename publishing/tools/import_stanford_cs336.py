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
import copy
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
SEMANTIC_REVIEW_PATH = COURSE_ROOT / "semantic-review.json"
EDITORIAL_MAP_PATH = COURSE_ROOT / "editorial-map.yml"
RETRIEVED_AT = "2026-09-04"
COURSE_URL = "https://cs336.stanford.edu/"
GITHUB_ORG = "stanford-cs336"
EXTRACTOR_REVISION = "stanford-cs336-semantic-audit-v3"
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


def poppler_version(tool: str) -> str:
    result = subprocess.run(
        [tool, "-v"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = re.search(r"(?:version\s+)?(\d+\.\d+\.\d+)", output)
    if not match:
        raise RuntimeError(f"cannot determine {tool} version")
    return match.group(1)


def edtrace_version() -> str:
    lock_text = (LECTURES_ROOT / "uv.lock").read_text("utf-8")
    match = re.search(
        r'^\[\[package\]\]\s*\nname = "edtrace"\s*\nversion = "([^"]+)"',
        lock_text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError("cannot determine pinned edtrace version")
    return match.group(1)


def extraction_provenance() -> dict[str, str]:
    return {
        "edtrace": edtrace_version(),
        "pdfinfo": poppler_version("pdfinfo"),
        "pdftotext": poppler_version("pdftotext"),
        "pdfimages": poppler_version("pdfimages"),
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": sha256_file(Path(__file__)),
        "semantic_review_sha256": sha256_file(SEMANTIC_REVIEW_PATH),
    }


def semantic_review() -> dict[str, Any]:
    value = json.loads(SEMANTIC_REVIEW_PATH.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"invalid semantic review specification: {SEMANTIC_REVIEW_PATH}")
    return value


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
                            "source_line": source_line,
                            "rendering_type": rendering_type,
                            "text": text_value,
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
                            "source_line": source_line,
                            "rendering_type": rendering_type,
                            "text": text_value,
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
                        "source_line": source_line,
                        "rendering_type": rendering_type,
                        "asset": asset,
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
                        "source_line": source_line,
                        "rendering_type": rendering_type,
                        "text": data if isinstance(data, str) else target,
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
    assignment_review = semantic_review()["assignments"]
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
        handout = COURSE_ROOT / assignment_review[f"assignment-{number:02d}"]["handout"]
        page_count = pdf_page_count(handout)
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


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise RuntimeError(f"cannot form stable semantic id from {value!r}")
    return result


def make_unit(
    object_id: str,
    semantic_id: str,
    kind: str,
    source_location: str,
    title: str,
    sort_key: tuple[int, int, str],
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": f"{object_id}-{slug(semantic_id)}",
        "source_object": object_id,
        "order": 0,
        "source_location": source_location,
        "kind": kind,
        "title": clean_title(title, semantic_id),
        "semantic_id": semantic_id,
        "_sort_key": sort_key,
        **fields,
    }


def finalize_units(units: list[dict[str, Any]], object_id: str) -> list[dict[str, Any]]:
    units.sort(key=lambda unit: tuple(unit["_sort_key"]))
    ids: set[str] = set()
    semantic_ids: set[str] = set()
    for order, unit in enumerate(units, start=1):
        unit.pop("_sort_key")
        unit["order"] = order
        if unit["id"] in ids:
            raise RuntimeError(f"{object_id}: duplicate source-unit id {unit['id']}")
        ids.add(unit["id"])
        semantic = unit.get("semantic_id")
        if semantic in semantic_ids:
            raise RuntimeError(f"{object_id}: duplicate semantic id {semantic}")
        semantic_ids.add(semantic)
    return units


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def page_span(spec: dict[str, Any], page_count: int, label: str) -> list[int]:
    values = spec.get("pages")
    if (
        not isinstance(values, list)
        or len(values) not in (1, 2)
        or not all(isinstance(page, int) for page in values)
    ):
        raise RuntimeError(f"{label}: pages must contain one page or inclusive start/end")
    start = values[0]
    end = values[-1]
    if start < 1 or end < start or end > page_count:
        raise RuntimeError(f"{label}: invalid page span {values} for {page_count} pages")
    return list(range(start, end + 1))


def page_fields(pages: list[int]) -> dict[str, int]:
    if len(pages) == 1:
        return {"page": pages[0]}
    return {"page_start": pages[0], "page_end": pages[-1]}


def assignment_task_location(path: Path, pages: list[int], task_id: str) -> str:
    if len(pages) == 1:
        scope = f"page={pages[0]}"
    else:
        scope = f"pages={pages[0]}-{pages[-1]}"
    return f"{relative(path)}#{scope}:problem={task_id}"


def semantic_page_location(path: Path, pages: list[int], semantic_id: str) -> str:
    if len(pages) == 1:
        scope = f"page={pages[0]}"
    else:
        scope = f"pages={pages[0]}-{pages[-1]}"
    return f"{relative(path)}#{scope}:semantic={semantic_id}"


def require_page_evidence(
    pages_text: list[str],
    pages: list[int],
    evidence: Any,
    label: str,
) -> str:
    if not isinstance(evidence, str) or not evidence.strip():
        raise RuntimeError(f"{label}: evidence_text must be a non-empty string")
    selected = normalized_text("\n".join(pages_text[page - 1] for page in pages))
    if normalized_text(evidence) not in selected:
        raise RuntimeError(f"{label}: evidence_text not found in reviewed page span: {evidence!r}")
    return evidence


def source_only_visual(
    visual_id: str,
    semantic_id: str,
    visual_kind: str,
    object_id: str,
    source_location: str,
    locations: list[int],
    question: str,
    sequence_locations: list[str],
    source_units: list[str],
    parent_sha256: str,
    extraction_tool: str,
    provenance: dict[str, str],
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": visual_id,
        "semantic_id": semantic_id,
        "visual_kind": visual_kind,
        "source_object": object_id,
        "source_units": source_units,
        "source_location": source_location,
        "source_pages": locations,
        "question": question,
        "sequence_members": [
            {"order": order, "source_location": location}
            for order, location in enumerate(sequence_locations, start=1)
        ],
        "disposition": "source-only",
        "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
        "destination_anchor": object_id,
        "reason": (
            "Reviewed semantic visual is preserved at exact event/page scope; "
            "canonical textbook placement remains pending."
        ),
        "attribution": "Stanford CS336 staff, Language Modeling from Scratch, Spring 2026",
        "rights_status": "permission-recorded",
        "rights_holder": "Stanford CS336 source authors and identified upstream asset owners",
        "rights_scope": "Local educational preservation and later attributed textbook reuse",
        "license_identifier": "User permission record (not a license)",
        "license_url": COURSE_URL,
        "rights_evidence": USER_PERMISSION,
        "parent_sha256": parent_sha256,
        "extractor_sha256": provenance["extractor_sha256"],
        "extractor_revision": provenance["extractor_revision"],
        "extraction_tool": extraction_tool,
        "rendered_route": "/sources/courses/stanford-cs336-spring-2026/",
        "desktop_evidence": "pending editorial integration; source archive inspected",
        "narrow_evidence": "pending editorial integration; source archive inspected",
        "reviewer": "Codex source audit",
        "checked_at": RETRIEVED_AT,
        **fields,
    }


def executable_units(
    number: int,
    filename: str,
    sha: str,
    review: dict[str, Any],
    provenance: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_id = f"lecture-{number:02d}"
    trace = trace_path(filename)
    source = LECTURES_ROOT / filename
    events = trace_renderings(filename, sha)
    if not events:
        raise RuntimeError(f"{object_id}: edtrace has no renderings")
    trace_steps = len(trace_document(filename)["steps"])
    headings = [event for event in events if event["kind"] == "section-boundary"]
    administrative_titles = {
        normalized_text(title) for title in review.get("administrative_headings", [])
    }
    matched_administrative: set[str] = set()
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    section_source_ranges = review.get("section_source_ranges", {})
    if not isinstance(section_source_ranges, dict):
        raise RuntimeError(f"{object_id}: section_source_ranges must be an object")
    figure_contexts = review.get("figure_contexts", {})
    if not isinstance(figure_contexts, dict):
        raise RuntimeError(f"{object_id}: figure_contexts must be an object")

    for index, event in enumerate(headings):
        start = event["step"]
        next_start = headings[index + 1]["step"] if index + 1 < len(headings) else trace_steps + 1
        end = max(start, next_start - 1)
        normalized_title = normalized_text(event["title"])
        is_administrative = normalized_title in administrative_titles
        if is_administrative:
            matched_administrative.add(normalized_title)
        semantic_id = (
            f"administrative-{slug(event['title'])}"
            if is_administrative
            else f"section-{start}-{slug(event['title'])}"
        )
        reviewed_ranges = section_source_ranges.get(semantic_id)
        if reviewed_ranges is not None:
            if (
                not isinstance(reviewed_ranges, list)
                or not reviewed_ranges
                or any(
                    not isinstance(span, list)
                    or len(span) != 2
                    or any(type(line) is not int or line < 1 for line in span)
                    or span[0] > span[1]
                    for span in reviewed_ranges
                )
            ):
                raise RuntimeError(
                    f"{object_id}/{semantic_id}: section_source_ranges must contain "
                    "positive [start, end] pairs"
                )
            if any(
                reviewed_ranges[index][0] <= reviewed_ranges[index - 1][1]
                for index in range(1, len(reviewed_ranges))
            ):
                raise RuntimeError(
                    f"{object_id}/{semantic_id}: section_source_ranges must be ordered "
                    "and non-overlapping"
                )
            lines_fragment = ",".join(
                f"{line_start}-{line_end}"
                for line_start, line_end in reviewed_ranges
            )
            location = (
                f"{relative(trace)}#steps={start}-{end}:"
                f"lines={lines_fragment}:semantic={semantic_id}"
            )
        else:
            location = f"{relative(trace)}#steps={start}-{end}:semantic={semantic_id}"
        fields: dict[str, Any] = {
            "event_start": start,
            "event_end": end,
            "heading_level": event["heading_level"],
        }
        if reviewed_ranges is not None:
            fields["source_line_ranges"] = reviewed_ranges
        else:
            fields.update(
                source_line_start=event.get("source_line"),
                source_line_end=headings[index + 1].get("source_line") - 1
                if index + 1 < len(headings)
                and isinstance(headings[index + 1].get("source_line"), int)
                else event.get("source_line"),
            )
        if is_administrative:
            fields.update(
                {
                    "exclusion_reason": (
                        "Course logistics, policy, or assignment administration is preserved "
                        "but excluded from teaching-content coverage."
                    ),
                    "exclusion_evidence": event["title"],
                }
            )
        units.append(
            make_unit(
                object_id,
                semantic_id,
                "administrative" if is_administrative else "section",
                location,
                event["title"],
                (start, 0, semantic_id),
                **fields,
            )
        )
    unmatched = administrative_titles - matched_administrative
    if unmatched:
        raise RuntimeError(f"{object_id}: unmatched administrative headings: {sorted(unmatched)}")
    unmatched_ranges = set(section_source_ranges) - {
        unit["semantic_id"] for unit in units
    }
    if unmatched_ranges:
        raise RuntimeError(
            f"{object_id}: section_source_ranges reference unknown headings: "
            f"{sorted(unmatched_ranges)}"
        )

    for spec in review.get("constructs", []):
        line_start = spec["line_start"]
        line_end = spec["line_end"]
        selected = [
            event
            for event in events
            if isinstance(event.get("source_line"), int)
            and line_start <= event["source_line"] <= line_end
        ]
        if not selected:
            raise RuntimeError(f"{object_id}/{spec['semantic_id']}: no edtrace events in reviewed lines")
        start = min(event["step"] for event in selected)
        end = max(event["step"] for event in selected)
        location = (
            f"{relative(trace)}#steps={start}-{end}:"
            f"lines={line_start}-{line_end}:semantic={spec['semantic_id']}"
        )
        units.append(
            make_unit(
                object_id,
                spec["semantic_id"],
                spec["kind"],
                location,
                spec["title"],
                (start, 1, spec["semantic_id"]),
                event_start=start,
                event_end=end,
                source_line_start=line_start,
                source_line_end=line_end,
                evidence_locations=[event["source_location"] for event in selected],
            )
        )

    claimed_images: set[str] = set()
    for spec in review.get("visual_sequences", []):
        line_start = spec["line_start"]
        line_end = spec["line_end"]
        selected = [
            event
            for event in events
            if isinstance(event.get("source_line"), int)
            and line_start <= event["source_line"] <= line_end
        ]
        if not selected:
            raise RuntimeError(f"{object_id}/{spec['semantic_id']}: no visual sequence events")
        start = min(event["step"] for event in selected)
        end = max(event["step"] for event in selected)
        source_pages = sorted({event["step"] for event in selected})
        location = (
            f"{relative(trace)}#steps={start}-{end}:"
            f"lines={line_start}-{line_end}:semantic={spec['semantic_id']}"
        )
        unit = make_unit(
            object_id,
            spec["semantic_id"],
            spec["kind"],
            location,
            spec["title"],
            (start, 2, spec["semantic_id"]),
            event_start=start,
            event_end=end,
            source_line_start=line_start,
            source_line_end=line_end,
            evidence_locations=[event["source_location"] for event in selected],
        )
        units.append(unit)
        if spec.get("claim_images"):
            claimed_images.update(
                event["source_location"] for event in selected if event["kind"] == "rendered-image"
            )
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-{slug(spec['semantic_id'])}",
                spec["semantic_id"],
                spec["visual_kind"],
                object_id,
                location,
                source_pages,
                spec["question"],
                [event["source_location"] for event in selected],
                [unit["id"]],
                sha256_file(trace),
                f"edtrace {provenance['edtrace']} archived JSON rendering stream",
                provenance,
                source_parent_sha256=sha256_file(source),
                source_lines=[line_start, line_end],
            )
        )

    for event in events:
        if event["kind"] != "rendered-image" or event["source_location"] in claimed_images:
            continue
        match = re.search(r"rendering=(\d+)$", event["source_location"])
        rendering_number = int(match.group(1)) if match else 1
        semantic_id = f"figure-step-{event['step']}-rendering-{rendering_number}"
        nearest_heading = next(
            (heading["title"] for heading in reversed(headings) if heading["step"] <= event["step"]),
            object_id,
        )
        figure_context = figure_contexts.get(str(event["step"]))
        parent_unit_id: str | None = None
        if figure_context is not None:
            if not isinstance(figure_context, dict):
                raise RuntimeError(
                    f"{object_id}/step-{event['step']}: figure context must be an object"
                )
            parent_semantic_id = figure_context.get("parent_semantic_id")
            section_title = figure_context.get("section_title")
            if not isinstance(parent_semantic_id, str) or not isinstance(section_title, str):
                raise RuntimeError(
                    f"{object_id}/step-{event['step']}: figure context requires "
                    "parent_semantic_id and section_title"
                )
            parent_unit = next(
                (unit for unit in units if unit["semantic_id"] == parent_semantic_id),
                None,
            )
            if parent_unit is None:
                raise RuntimeError(
                    f"{object_id}/step-{event['step']}: unknown figure parent "
                    f"{parent_semantic_id}"
                )
            parent_unit_id = parent_unit["id"]
            nearest_heading = section_title
        title = clean_title(
            Path(event.get("asset", "figure")).stem.replace("_", " ").replace("-", " "),
            f"Figure in {nearest_heading}",
        )
        unit = make_unit(
            object_id,
            semantic_id,
            "figure",
            event["source_location"],
            title,
            (event["step"], 3, semantic_id),
            event_start=event["step"],
            event_end=event["step"],
            source_line_start=event.get("source_line"),
            source_line_end=event.get("source_line"),
            asset_url=event["asset_url"],
        )
        units.append(unit)
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-{semantic_id}",
                semantic_id,
                "figure",
                object_id,
                event["source_location"],
                [event["step"]],
                f"How does {title} support the lecture section “{nearest_heading}”?",
                [event["source_location"]],
                [unit["id"], *([parent_unit_id] if parent_unit_id else [])],
                sha256_file(trace),
                f"edtrace {provenance['edtrace']} archived image rendering",
                provenance,
                source_parent_sha256=sha256_file(source),
                asset_url=event["asset_url"],
            )
        )
    unmatched_figure_contexts = set(figure_contexts) - {
        str(event["step"])
        for event in events
        if event["kind"] == "rendered-image" and event["source_location"] not in claimed_images
    }
    if unmatched_figure_contexts:
        raise RuntimeError(
            f"{object_id}: figure_contexts reference missing or claimed images: "
            f"{sorted(unmatched_figure_contexts)}"
        )
    return finalize_units(units, object_id), visuals


def pdf_units_and_visuals(
    object_id: str,
    path: Path,
    review: dict[str, Any],
    provenance: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pages_text = pdf_pages(path)
    checksum = sha256_file(path)
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for spec in review.get("units", []):
        pages = page_span(spec, len(pages_text), f"{object_id}/{spec['semantic_id']}")
        evidence = require_page_evidence(
            pages_text,
            pages,
            spec.get("evidence_text"),
            f"{object_id}/{spec['semantic_id']}",
        )
        fields: dict[str, Any] = {"evidence_text": evidence, **page_fields(pages)}
        if spec["kind"] == "administrative":
            reason = spec.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise RuntimeError(
                    f"{object_id}/{spec['semantic_id']}: administrative reason is required"
                )
            fields.update(
                exclusion_reason=reason,
                exclusion_evidence=evidence,
            )
        units.append(
            make_unit(
                object_id,
                spec["semantic_id"],
                spec["kind"],
                semantic_page_location(path, pages, spec["semantic_id"]),
                spec["title"],
                (pages[0], 0, spec["semantic_id"]),
                **fields,
            )
        )
    units_by_semantic_id = {unit["semantic_id"]: unit for unit in units}
    embedded_images = pdf_embedded_images(path)
    for spec in review.get("visuals", []):
        semantic_id = spec["semantic_id"]
        linked_unit = units_by_semantic_id.get(semantic_id)
        if linked_unit is None:
            raise RuntimeError(f"{object_id}/{semantic_id}: visual lacks a reviewed semantic unit")
        pages = page_span(spec, len(pages_text), f"{object_id}/{semantic_id} visual")
        raster_evidence = [
            image for image in embedded_images if image["page"] in set(pages)
        ]
        location = semantic_page_location(path, pages, semantic_id)
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-{slug(semantic_id)}",
                semantic_id,
                spec["visual_kind"],
                object_id,
                location,
                pages,
                spec["question"],
                [
                    f"{relative(path)}#page={page}:semantic={semantic_id}"
                    for page in pages
                ],
                [linked_unit["id"]],
                checksum,
                (
                    f"Poppler pdftotext {provenance['pdftotext']} semantic page review; "
                    f"pdfimages {provenance['pdfimages']} raster evidence"
                ),
                provenance,
                raster_evidence=raster_evidence,
            )
        )
    raster_summary = {
        "source_object": object_id,
        "source_path": relative(path),
        "parent_sha256": checksum,
        "detected_images": len(embedded_images),
        "treatment": (
            "pdfimages detections are evidence attached to reviewed semantic sequences; "
            "they are not independent teaching visuals."
        ),
    }
    return finalize_units(units, object_id), visuals, raster_summary


def marker_title(text: str, match: re.Match[str], group: int, fallback: str) -> str:
    title = clean_title(match.group(group), "")
    if title:
        return title
    for line in text[match.end() :].splitlines():
        title = clean_title(line, "")
        if title:
            return title
    return fallback


def assignment_units(
    object_id: str,
    repository: str,
    review: dict[str, Any],
    provenance: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = ASSIGNMENTS_ROOT / repository
    handout = COURSE_ROOT / review["handout"]
    if handout.parent != root:
        raise RuntimeError(f"{object_id}: reviewed handout is outside {repository}")
    pages_text = pdf_pages(handout)
    checksum = sha256_file(handout)
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    task_units: list[dict[str, Any]] = []
    task_ids: list[str] = []
    deliverable_count = 0
    current_task: str | None = None
    problem_pattern = re.compile(r"(?m)^\s*Problem \(([^)]+)\):\s*(.*?)\s*$")
    deliverable_pattern = re.compile(r"(?m)^\s*Deliverable:\s*(.*?)\s*$")

    for page_number, page_text in enumerate(pages_text, start=1):
        markers: list[tuple[int, str, re.Match[str]]] = [
            (match.start(), "problem", match) for match in problem_pattern.finditer(page_text)
        ]
        markers.extend(
            (match.start(), "deliverable", match)
            for match in deliverable_pattern.finditer(page_text)
        )
        for offset, marker_kind, match in sorted(markers, key=lambda value: value[0]):
            if marker_kind == "problem":
                task_id = match.group(1).strip()
                current_task = task_id
                task_ids.append(task_id)
                task_unit = make_unit(
                    object_id,
                    f"task-{task_id}",
                    "task",
                    f"{relative(handout)}#page={page_number}:problem={task_id}",
                    marker_title(page_text, match, 2, task_id.replace("_", " ")),
                    (page_number, offset, f"task-{task_id}"),
                    task_id=task_id,
                    page=page_number,
                )
                units.append(task_unit)
                task_units.append(task_unit)
            else:
                deliverable_count += 1
                deliverable_id = (
                    f"{current_task or 'general'}-{deliverable_count:03d}"
                )
                units.append(
                    make_unit(
                        object_id,
                        f"deliverable-{deliverable_id}",
                        "deliverable",
                        (
                            f"{relative(handout)}#page={page_number}:"
                            f"deliverable={deliverable_count}"
                        ),
                        marker_title(
                            page_text,
                            match,
                            1,
                            f"Deliverable for {current_task or 'assignment'}",
                        ),
                        (page_number, offset, f"deliverable-{deliverable_id}"),
                        deliverable_id=deliverable_id,
                        task_id=current_task,
                        page=page_number,
                    )
                )
    expected_task_ids = review.get("expected_task_ids")
    if task_ids != expected_task_ids:
        raise RuntimeError(
            f"{object_id}: named problem closure mismatch; "
            f"expected {expected_task_ids}, extracted {task_ids}"
        )
    if deliverable_count != review.get("expected_deliverables"):
        raise RuntimeError(
            f"{object_id}: deliverable closure mismatch; "
            f"expected {review.get('expected_deliverables')}, extracted {deliverable_count}"
        )
    final_task_page = review.get("final_task_page")
    if (
        not task_units
        or type(final_task_page) is not int
        or final_task_page < task_units[-1]["page"]
        or final_task_page > len(pages_text)
    ):
        raise RuntimeError(
            f"{object_id}: final_task_page must close the final named problem "
            f"within the {len(pages_text)}-page handout"
        )
    task_start_pages = [unit["page"] for unit in task_units]
    for index, unit in enumerate(task_units):
        start = task_start_pages[index]
        end = (
            final_task_page
            if index == len(task_units) - 1
            else max(start, task_start_pages[index + 1] - 1)
        )
        pages = list(range(start, end + 1))
        unit["source_location"] = assignment_task_location(
            handout, pages, unit["task_id"]
        )
        unit["evidence_text"] = f"Problem ({unit['task_id']}):"
        unit.pop("page")
        unit.update(page_fields(pages))

    test_paths = sorted(
        {
            path
            for pattern in review.get("test_globs", [])
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    test_count = 0
    for test_path in test_paths:
        tree = ast.parse(test_path.read_text("utf-8"))
        functions = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
            ),
            key=lambda node: node.lineno,
        )
        for node in functions:
            test_count += 1
            semantic_id = (
                f"test-{slug(test_path.relative_to(root).as_posix())}-"
                f"{node.lineno}-{node.name}"
            )
            units.append(
                make_unit(
                    object_id,
                    semantic_id,
                    "test-interface",
                    f"{relative(test_path)}#L{node.lineno}",
                    f"Test contract: {node.name.replace('_', ' ')}",
                    (len(pages_text) + 1, node.lineno, semantic_id),
                    interface_name=node.name,
                    test_file=test_path.relative_to(root).as_posix(),
                )
            )
    if test_count != review.get("expected_test_interfaces"):
        raise RuntimeError(
            f"{object_id}: test-interface closure mismatch; "
            f"expected {review.get('expected_test_interfaces')}, extracted {test_count}"
        )

    for spec in review.get("units", []):
        pages = page_span(spec, len(pages_text), f"{object_id}/{spec['semantic_id']}")
        evidence = require_page_evidence(
            pages_text, pages, spec.get("evidence_text"), f"{object_id}/{spec['semantic_id']}"
        )
        units.append(
            make_unit(
                object_id,
                spec["semantic_id"],
                spec["kind"],
                semantic_page_location(handout, pages, spec["semantic_id"]),
                spec["title"],
                (pages[0], -1, spec["semantic_id"]),
                evidence_text=evidence,
                **page_fields(pages),
            )
        )
    for spec in review.get("administrative", []):
        pages = page_span(spec, len(pages_text), f"{object_id}/{spec['semantic_id']}")
        evidence = require_page_evidence(
            pages_text, pages, spec.get("evidence_text"), f"{object_id}/{spec['semantic_id']}"
        )
        units.append(
            make_unit(
                object_id,
                spec["semantic_id"],
                "administrative",
                semantic_page_location(handout, pages, spec["semantic_id"]),
                spec["title"],
                (pages[0], -2, spec["semantic_id"]),
                exclusion_reason=spec["reason"],
                exclusion_evidence=evidence,
                evidence_text=evidence,
                **page_fields(pages),
            )
        )
    for spec in review.get("evaluation", []):
        page = spec["page"]
        pages = [page]
        evidence = require_page_evidence(
            pages_text, pages, spec.get("evidence_text"), f"{object_id}/{spec['semantic_id']}"
        )
        units.append(
            make_unit(
                object_id,
                spec["semantic_id"],
                "evaluation-requirement",
                semantic_page_location(handout, pages, spec["semantic_id"]),
                spec["title"],
                (page, -1, spec["semantic_id"]),
                criterion_id=spec["criterion_id"],
                page=page,
                evidence_text=evidence,
            )
        )
    embedded_images = pdf_embedded_images(handout)
    for spec in review.get("visuals", []):
        pages = page_span(spec, len(pages_text), f"{object_id}/{spec['semantic_id']}")
        evidence = require_page_evidence(
            pages_text, pages, spec.get("evidence_text"), f"{object_id}/{spec['semantic_id']}"
        )
        location = semantic_page_location(handout, pages, spec["semantic_id"])
        unit = make_unit(
            object_id,
            spec["semantic_id"],
            spec["kind"],
            location,
            spec["title"],
            (pages[0], -1, spec["semantic_id"]),
            evidence_text=evidence,
            **page_fields(pages),
        )
        units.append(unit)
        visuals.append(
            source_only_visual(
                f"{object_id}-visual-{slug(spec['semantic_id'])}",
                spec["semantic_id"],
                spec["visual_kind"],
                object_id,
                location,
                pages,
                spec["question"],
                [
                    f"{relative(handout)}#page={page}:semantic={spec['semantic_id']}"
                    for page in pages
                ],
                [unit["id"]],
                checksum,
                (
                    f"Poppler pdftotext {provenance['pdftotext']} semantic handout review; "
                    f"pdfimages {provenance['pdfimages']} raster evidence"
                ),
                provenance,
                raster_evidence=[
                    image for image in embedded_images if image["page"] in set(pages)
                ],
            )
        )
    raster_summary = {
        "source_object": object_id,
        "source_path": relative(handout),
        "parent_sha256": checksum,
        "detected_images": len(embedded_images),
        "treatment": (
            "pdfimages detections are evidence attached to reviewed semantic sequences; "
            "they are not independent teaching visuals."
        ),
    }
    return finalize_units(units, object_id), visuals, raster_summary


def coverage_row(unit: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": f"coverage-{unit['id']}",
        "source_object": unit["source_object"],
        "source_unit": unit["id"],
        "source_location": unit["source_location"],
        "kind": unit["kind"],
        "title": unit["title"],
        "primary_sources": [unit["source_object"]],
    }
    if unit["kind"] == "administrative":
        row.update(
            {
                "disposition": "excluded",
                "reason": unit["exclusion_reason"],
                "evidence": unit["exclusion_evidence"],
            }
        )
    else:
        row.update(
            {
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": unit["source_object"],
                "reason": SOURCE_ONLY_REASON,
            }
        )
    return row


def build_ledgers(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    revisions = {item["repository"]: item["revision"] for item in lock["repositories"]}
    review = semantic_review()
    provenance = extraction_provenance()
    executable_ids = {
        f"lecture-{number:02d}"
        for number, _, _, _, filename in SCHEDULE
        if filename and filename.endswith(".py")
    }
    pdf_ids = {
        f"lecture-{number:02d}"
        for number, _, _, _, filename in SCHEDULE
        if filename and filename.endswith(".pdf")
    }
    assignment_ids = {
        *(f"assignment-{number:02d}" for number, _, _, _ in ASSIGNMENT_META),
        "assignment-05-safety-supplement",
    }
    expected_review_sets = {
        "executable_lectures": executable_ids,
        "pdf_lectures": pdf_ids,
        "assignments": assignment_ids,
    }
    for section, expected in expected_review_sets.items():
        actual = set(review.get(section, {}))
        if actual != expected:
            raise RuntimeError(
                f"semantic review {section} closure mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    manifest = source_manifest(lock)
    units: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    raster_summaries: list[dict[str, Any]] = []
    for number, _, _, _, filename in SCHEDULE:
        if filename is None:
            continue
        object_id = f"lecture-{number:02d}"
        path = LECTURES_ROOT / filename
        if path.suffix == ".py":
            new_units, new_visuals = executable_units(
                number,
                filename,
                revisions["lectures"],
                review["executable_lectures"][object_id],
                provenance,
            )
        else:
            new_units, new_visuals, summary = pdf_units_and_visuals(
                object_id,
                path,
                review["pdf_lectures"][object_id],
                provenance,
            )
            raster_summaries.append(summary)
        units.extend(new_units)
        visuals.extend(new_visuals)
    for number, repository, _, _ in ASSIGNMENT_META:
        object_id = f"assignment-{number:02d}"
        new_units, new_visuals, summary = assignment_units(
            object_id,
            repository,
            review["assignments"][object_id],
            provenance,
        )
        units.extend(new_units)
        visuals.extend(new_visuals)
        raster_summaries.append(summary)
    safety_units, safety_visuals, safety_summary = assignment_units(
        "assignment-05-safety-supplement",
        "assignment5-alignment",
        review["assignments"]["assignment-05-safety-supplement"],
        provenance,
    )
    units.extend(safety_units)
    visuals.extend(safety_visuals)
    raster_summaries.append(safety_summary)
    source_units = {
        "schema_version": 1,
        "review_method": review["review_method"],
        "semantic_review_sha256": provenance["semantic_review_sha256"],
        "units": units,
    }
    baseline_coverage = {
        "schema_version": 1,
        "semantic_review_sha256": provenance["semantic_review_sha256"],
        "rows": [coverage_row(unit) for unit in units],
    }
    baseline_visuals = {
        "schema_version": 1,
        "audit_method": (
            "Executable lectures use archived edtrace JSON renderings grouped by reviewed "
            "source-line/event scopes. PDF lectures and assignments use reviewed semantic "
            "page spans; Poppler raster detections are evidence, never standalone visuals."
        ),
        "extraction_provenance": provenance,
        "raster_evidence_summary": raster_summaries,
        "rows": visuals,
    }

    # Validate the immutable source semantics before any editorial destination can
    # affect the generated ledgers. This keeps extraction and editorial judgement
    # as separate, auditable layers.
    closure_failures: list[str] = []
    validate_pdf_page_closure(source_units, baseline_coverage, closure_failures)
    if closure_failures:
        raise RuntimeError(
            "Stanford CS336 source-semantic closure failed before editorial overlay:\n- "
            + "\n- ".join(closure_failures)
        )
    overlay = load_editorial_overlay()
    coverage, editorial_visuals, _ = apply_editorial_overlay(
        manifest,
        source_units,
        baseline_coverage,
        baseline_visuals,
        overlay,
    )
    return manifest, source_units, coverage, editorial_visuals


def document_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


COVERAGE_EDITORIAL_DISPOSITIONS = {
    "integrated",
    "covered-existing",
    "contract-only",
    "source-only",
    "excluded",
}
VISUAL_EDITORIAL_DISPOSITIONS = {
    "integrated",
    "covered-existing",
    "source-only",
    "excluded",
}
EDITORIAL_TOP_LEVEL_FIELDS = {"schema_version", "reviewed_objects", "coverage", "visuals"}
COVERAGE_EDITORIAL_FIELDS = {
    "disposition",
    "destination",
    "destination_anchor",
    "reason",
    "evidence",
}
VISUAL_EDITORIAL_FIELDS = COVERAGE_EDITORIAL_FIELDS | {
    "local_file",
    "transformation",
    "caption",
    "rendered_route",
    "desktop_evidence",
    "narrow_evidence",
    "reviewer",
    "checked_at",
}


def load_editorial_overlay(path: Path = EDITORIAL_MAP_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{relative(path)}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{relative(path)}: expected mapping")
    return value


def editorial_overlay_sha256(overlay: dict[str, Any]) -> str:
    return sha256_bytes(document_bytes(overlay))


def markdown_anchor(text: str) -> str:
    value = text.casefold().strip()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"[ _]+", "-", value).strip("-")


def destination_source_unit_ids(text: str) -> set[str]:
    if not text.startswith("---\n"):
        return set()
    end = text.find("\n---", 4)
    if end == -1:
        return set()
    frontmatter = text[4:end]
    lines = frontmatter.splitlines()
    values: set[str] = set()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"source_unit_id:\s*(.*)", line)
        if match is None:
            continue
        inline = match.group(1).strip()
        if inline:
            if inline.startswith("[") and inline.endswith("]"):
                inline = inline[1:-1]
            values.update(
                item.strip().strip("'\"")
                for item in inline.split(",")
                if item.strip()
            )
            continue
        for following in lines[index + 1 :]:
            item = re.fullmatch(r"\s+-\s+(.+?)\s*", following)
            if item is not None:
                values.add(item.group(1).strip().strip("'\""))
                continue
            if following.startswith((" ", "\t")) or not following.strip():
                continue
            break
    return values


def destination_has_anchor(text: str, anchor: str) -> bool:
    quoted = re.escape(anchor)
    if re.search(rf"<(?:a|span)\b[^>]*(?:id|name)=[\"']{quoted}[\"']", text):
        return True
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE):
        if markdown_anchor(re.sub(r"\s+#+\s*$", "", heading)) == anchor:
            return True
    return False


def resolve_repository_file(repository_root: Path, relative_path: str, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    root = repository_root.resolve()
    destination = (root / relative_path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root: {relative_path}")
    if not destination.is_file():
        raise ValueError(f"{label} does not exist: {relative_path}")
    return destination


def validate_destination(
    repository_root: Path,
    decision_id: str,
    decision: dict[str, Any],
    reciprocal_ids: set[str],
) -> None:
    destination_value = decision.get("destination")
    anchor = decision.get("destination_anchor")
    if not isinstance(destination_value, str) or not destination_value.strip():
        raise ValueError(f"{decision_id}: integrated decision lacks destination")
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError(f"{decision_id}: integrated decision lacks destination_anchor")
    destination = resolve_repository_file(
        repository_root,
        destination_value,
        f"{decision_id}: destination",
    )
    text = destination.read_text("utf-8")
    if not destination_has_anchor(text, anchor):
        raise ValueError(
            f"{decision_id}: destination anchor does not exist: {destination_value}#{anchor}"
        )
    actual_source_ids = destination_source_unit_ids(text)
    missing = sorted(reciprocal_ids - actual_source_ids)
    if missing:
        raise ValueError(
            f"{decision_id}: destination lacks reciprocal source_unit_id values: {missing}"
        )


def validate_contract_only(
    repository_root: Path,
    unit_id: str,
    unit: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    if unit.get("kind") != "test-interface":
        raise ValueError(f"{unit_id}: contract-only is restricted to test-interface units")
    evidence = decision.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"{unit_id}: contract-only decision lacks evidence")
    contract_path, separator, fragment = evidence.partition("#")
    if separator != "#" or fragment != "expected_tests":
        raise ValueError(
            f"{unit_id}: contract-only evidence must point to #expected_tests"
        )
    contract_file = resolve_repository_file(
        repository_root,
        contract_path,
        f"{unit_id}: contract evidence",
    )
    try:
        contract = json.loads(contract_file.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{unit_id}: invalid contract evidence: {error}") from error
    expected_tests = contract.get("expected_tests") if isinstance(contract, dict) else None
    test_file = unit.get("test_file")
    interface_name = unit.get("interface_name")
    expected_test = f"{test_file}::{interface_name}"
    if not isinstance(expected_tests, list) or expected_test not in expected_tests:
        raise ValueError(
            f"{unit_id}: contract evidence does not list {expected_test}"
        )


def apply_editorial_overlay(
    manifest: dict[str, Any],
    source_units: dict[str, Any],
    baseline_coverage: dict[str, Any],
    baseline_visuals: dict[str, Any],
    overlay: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not isinstance(overlay, dict):
        raise ValueError("editorial overlay must be a mapping")
    unknown_top = sorted(set(overlay) - EDITORIAL_TOP_LEVEL_FIELDS)
    if unknown_top:
        raise ValueError(f"unsupported editorial overlay field: {unknown_top[0]}")
    if overlay.get("schema_version") != 1:
        raise ValueError("editorial overlay schema_version must be 1")

    object_ids = {
        item.get("id")
        for item in manifest.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    reviewed_objects = overlay.get("reviewed_objects")
    if not isinstance(reviewed_objects, list) or not all(
        isinstance(item, str) for item in reviewed_objects
    ):
        raise ValueError("editorial overlay reviewed_objects must be a list of ids")
    if len(reviewed_objects) != len(set(reviewed_objects)):
        raise ValueError("editorial overlay reviewed_objects contains duplicates")
    unknown_objects = sorted(set(reviewed_objects) - object_ids)
    if unknown_objects:
        raise ValueError(f"unknown reviewed source object: {unknown_objects[0]}")
    reviewed = set(reviewed_objects)

    unit_rows = source_units.get("units")
    coverage_rows = baseline_coverage.get("rows")
    visual_rows = baseline_visuals.get("rows")
    if not isinstance(unit_rows, list) or not isinstance(coverage_rows, list):
        raise ValueError("source units and baseline coverage must contain row lists")
    if not isinstance(visual_rows, list):
        raise ValueError("baseline visuals must contain a row list")
    units_by_id = {
        row["id"]: row
        for row in unit_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    coverage_by_id = {
        row["source_unit"]: row
        for row in coverage_rows
        if isinstance(row, dict) and isinstance(row.get("source_unit"), str)
    }
    visuals_by_id = {
        row["id"]: row
        for row in visual_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    coverage_overlay = overlay.get("coverage")
    visual_overlay = overlay.get("visuals")
    if not isinstance(coverage_overlay, dict):
        raise ValueError("editorial overlay coverage must be a mapping keyed by source unit id")
    if not isinstance(visual_overlay, dict):
        raise ValueError("editorial overlay visuals must be a mapping keyed by visual id")

    unknown_units = sorted(set(coverage_overlay) - set(units_by_id))
    if unknown_units:
        raise ValueError(f"unknown coverage source unit: {unknown_units[0]}")
    unknown_visuals = sorted(set(visual_overlay) - set(visuals_by_id))
    if unknown_visuals:
        raise ValueError(f"unknown visual id: {unknown_visuals[0]}")

    required_units = {
        unit_id
        for unit_id, unit in units_by_id.items()
        if unit.get("source_object") in reviewed and unit.get("kind") != "administrative"
    }
    required_visuals = {
        visual_id
        for visual_id, visual in visuals_by_id.items()
        if visual.get("source_object") in reviewed
    }
    missing_units = sorted(required_units - set(coverage_overlay))
    if missing_units:
        raise ValueError(f"missing coverage decision for reviewed object: {missing_units[0]}")
    missing_visuals = sorted(required_visuals - set(visual_overlay))
    if missing_visuals:
        raise ValueError(f"missing visual decision for reviewed object: {missing_visuals[0]}")

    for unit_id in coverage_overlay:
        unit = units_by_id[unit_id]
        if unit.get("source_object") not in reviewed:
            raise ValueError(f"coverage decision belongs to an undeclared reviewed object: {unit_id}")
        if (
            unit.get("kind") == "administrative"
            and coverage_overlay[unit_id].get("disposition") not in {"source-only", "excluded"}
        ):
            raise ValueError(
                f"administrative source semantics may only be source-only or excluded: {unit_id}"
            )
    for visual_id in visual_overlay:
        visual = visuals_by_id[visual_id]
        if visual.get("source_object") not in reviewed:
            raise ValueError(f"visual decision belongs to an undeclared reviewed object: {visual_id}")

    coverage = copy.deepcopy(baseline_coverage)
    visuals = copy.deepcopy(baseline_visuals)
    output_coverage_by_id = {row["source_unit"]: row for row in coverage["rows"]}
    output_visuals_by_id = {row["id"]: row for row in visuals["rows"]}

    for unit_id, decision in coverage_overlay.items():
        if not isinstance(decision, dict):
            raise ValueError(f"coverage decision must be a mapping: {unit_id}")
        unit = units_by_id[unit_id]
        unknown = sorted(set(decision) - COVERAGE_EDITORIAL_FIELDS)
        if unknown:
            raise ValueError(f"unsupported coverage editorial field: {unknown[0]}")
        disposition = decision.get("disposition")
        if disposition not in COVERAGE_EDITORIAL_DISPOSITIONS:
            raise ValueError(f"{unit_id}: invalid disposition: {disposition!r}")
        row = output_coverage_by_id[unit_id]
        row.pop("reason", None)
        row.pop("evidence", None)
        if disposition == "excluded":
            row.pop("destination", None)
            row.pop("destination_anchor", None)
        row.update(decision)
        if disposition in {"source-only", "excluded"}:
            for field in ("reason", "evidence"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{unit_id}: {disposition} decision lacks {field}")
        else:
            validate_destination(repository_root, unit_id, row, {unit_id})
            if disposition == "contract-only":
                validate_contract_only(repository_root, unit_id, unit, row)

    for visual_id, decision in visual_overlay.items():
        if not isinstance(decision, dict):
            raise ValueError(f"visual decision must be a mapping: {visual_id}")
        unknown = sorted(set(decision) - VISUAL_EDITORIAL_FIELDS)
        if unknown:
            raise ValueError(f"unsupported visual editorial field: {unknown[0]}")
        disposition = decision.get("disposition")
        if disposition not in VISUAL_EDITORIAL_DISPOSITIONS:
            raise ValueError(f"{visual_id}: invalid disposition: {disposition!r}")
        row = output_visuals_by_id[visual_id]
        row.pop("reason", None)
        row.pop("evidence", None)
        if disposition == "excluded":
            row.pop("destination", None)
            row.pop("destination_anchor", None)
        row.update(decision)
        if disposition in {"source-only", "excluded"}:
            for field in ("reason", "evidence"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{visual_id}: {disposition} decision lacks {field}")
        else:
            reciprocal = {
                item for item in row.get("source_units", []) if isinstance(item, str)
            }
            validate_destination(repository_root, visual_id, row, reciprocal)
            for field in ("local_file", "transformation", "caption", "rendered_route"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{visual_id}: integrated visual lacks {field}")
            resolve_repository_file(repository_root, row["local_file"], f"{visual_id}: local_file")

    overlay_sha = editorial_overlay_sha256(overlay)
    coverage["editorial_overlay_sha256"] = overlay_sha
    visuals["editorial_overlay_sha256"] = overlay_sha
    return coverage, visuals, overlay_sha


def write_json_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(document_bytes(value))


def generated_document_map(
    documents: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return dict(zip(GENERATED_LEDGER_FILES, documents, strict=True))


def document_record_count(filename: str, document: dict[str, Any]) -> int:
    key = "objects" if filename == "source-manifest.yml" else (
        "units" if filename == "source-units.yml" else "rows"
    )
    records = document.get(key)
    if not isinstance(records, list):
        raise RuntimeError(f"{filename}: missing {key} records")
    return len(records)


def generated_audit(
    documents: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    provenance = documents["visuals.yml"]["extraction_provenance"]
    inventory_rows = inventory.get("files", [])
    return {
        "extractor_revision": provenance["extractor_revision"],
        "extractor_sha256": provenance["extractor_sha256"],
        "semantic_review_sha256": provenance["semantic_review_sha256"],
        "editorial_overlay_sha256": documents["coverage.yml"]["editorial_overlay_sha256"],
        "tool_versions": {
            key: provenance[key]
            for key in ("edtrace", "pdfinfo", "pdftotext", "pdfimages")
        },
        "archive": {
            "artifacts": len(inventory_rows),
            "bytes": sum(
                row.get("bytes", 0)
                for row in inventory_rows
                if isinstance(row, dict) and isinstance(row.get("bytes"), int)
            ),
        },
        "ledgers": {
            filename: {
                "sha256": sha256_bytes(document_bytes(document)),
                "records": document_record_count(filename, document),
            }
            for filename, document in documents.items()
        },
    }


def validate_generated_documents(
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    lock: dict[str, Any],
    failures: list[str],
) -> None:
    audit = lock.get("generated_audit")
    if not isinstance(audit, dict):
        failures.append("snapshot lock lacks generated_audit")
        return
    locked_ledgers = audit.get("ledgers")
    if not isinstance(locked_ledgers, dict):
        failures.append("snapshot lock lacks generated ledger hashes/counts")
        return
    for filename in GENERATED_LEDGER_FILES:
        actual_document = actual.get(filename)
        expected_document = expected.get(filename)
        if actual_document != expected_document:
            failures.append(f"{filename}: deterministic extraction mismatch")
        if not isinstance(actual_document, dict) or not isinstance(expected_document, dict):
            continue
        locked = locked_ledgers.get(filename)
        if not isinstance(locked, dict):
            failures.append(f"{filename}: missing locked ledger audit")
            continue
        expected_hash = sha256_bytes(document_bytes(expected_document))
        actual_hash = sha256_bytes(document_bytes(actual_document))
        if locked.get("sha256") != expected_hash:
            failures.append(f"{filename}: locked SHA-256 differs from deterministic extraction")
        if locked.get("sha256") != actual_hash:
            failures.append(f"{filename}: generated ledger SHA-256 mismatch")
        expected_count = document_record_count(filename, expected_document)
        actual_count = document_record_count(filename, actual_document)
        if locked.get("records") != expected_count:
            failures.append(f"{filename}: locked record count differs from deterministic extraction")
        if locked.get("records") != actual_count:
            failures.append(f"{filename}: generated ledger record count mismatch")
    provenance = expected.get("visuals.yml", {}).get("extraction_provenance", {})
    for key in ("extractor_revision", "extractor_sha256", "semantic_review_sha256"):
        if audit.get(key) != provenance.get(key):
            failures.append(f"snapshot lock {key} differs from deterministic extraction")
    expected_overlay_sha = expected.get("coverage.yml", {}).get("editorial_overlay_sha256")
    if audit.get("editorial_overlay_sha256") != expected_overlay_sha:
        failures.append("snapshot lock editorial_overlay_sha256 differs from deterministic overlay")
    if expected.get("visuals.yml", {}).get("editorial_overlay_sha256") != expected_overlay_sha:
        failures.append("coverage and visual ledgers disagree on editorial overlay SHA-256")
    tool_versions = audit.get("tool_versions")
    if not isinstance(tool_versions, dict):
        failures.append("snapshot lock lacks extraction tool versions")
    else:
        for key in ("edtrace", "pdfinfo", "pdftotext", "pdfimages"):
            if tool_versions.get(key) != provenance.get(key):
                failures.append(f"snapshot lock tool version differs for {key}")


def reviewed_pdf_sources() -> dict[str, Path]:
    review = semantic_review()
    sources = {
        f"lecture-{number:02d}": LECTURES_ROOT / filename
        for number, _, _, _, filename in SCHEDULE
        if filename and filename.endswith(".pdf")
    }
    assignment_ids = [
        *(f"assignment-{number:02d}" for number, _, _, _ in ASSIGNMENT_META),
        "assignment-05-safety-supplement",
    ]
    assignment_review = review.get("assignments")
    if not isinstance(assignment_review, dict):
        raise RuntimeError("semantic review assignments must be a mapping")
    for object_id in assignment_ids:
        spec = assignment_review.get(object_id)
        if not isinstance(spec, dict) or not isinstance(spec.get("handout"), str):
            raise RuntimeError(f"{object_id}: semantic review lacks a handout path")
        sources[object_id] = COURSE_ROOT / spec["handout"]
    return dict(sorted(sources.items()))


def source_unit_pages(
    unit: dict[str, Any],
    page_count: int,
    label: str,
    failures: list[str],
) -> list[int]:
    has_page = "page" in unit
    has_start = "page_start" in unit
    has_end = "page_end" in unit
    if not has_page and not has_start and not has_end:
        return []
    if has_page and (has_start or has_end):
        failures.append(f"{label}: ambiguous page and page-range fields")
        return []
    if has_page:
        page = unit.get("page")
        if type(page) is not int or page < 1 or page > page_count:
            failures.append(f"{label}: invalid page {page!r} for {page_count}-page PDF")
            return []
        return [page]
    start = unit.get("page_start")
    end = unit.get("page_end")
    if type(start) is not int or type(end) is not int:
        failures.append(f"{label}: page range must contain integers")
        return []
    if start < 1 or end < start or end > page_count:
        failures.append(
            f"{label}: invalid page range {start}-{end} for {page_count}-page PDF"
        )
        return []
    return list(range(start, end + 1))


def validate_pdf_page_closure(
    source_units: dict[str, Any],
    coverage: dict[str, Any],
    failures: list[str],
) -> None:
    units = source_units.get("units")
    rows = coverage.get("rows")
    if not isinstance(units, list) or not isinstance(rows, list):
        failures.append("PDF page closure requires source-unit and coverage rows")
        return
    coverage_by_unit = {
        row.get("source_unit"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("source_unit"), str)
    }
    for object_id, path in reviewed_pdf_sources().items():
        if not path.is_file():
            failures.append(f"{object_id}: reviewed PDF is missing: {relative(path)}")
            continue
        page_count = pdf_page_count(path)
        expected_location_prefix = f"{relative(path)}#"
        covered_pages: set[int] = set()
        for unit in units:
            if not isinstance(unit, dict) or unit.get("source_object") != object_id:
                continue
            unit_id = unit.get("id")
            label = f"{object_id}/{unit_id or '<missing-id>'}"
            pages = source_unit_pages(unit, page_count, label, failures)
            if not pages:
                continue
            location = unit.get("source_location")
            if not isinstance(location, str) or not location.startswith(
                expected_location_prefix
            ):
                failures.append(
                    f"{label}: page scope does not point to reviewed PDF {relative(path)}"
                )
                continue
            row = coverage_by_unit.get(unit_id)
            if (
                not isinstance(row, dict)
                or row.get("source_object") != object_id
                or row.get("source_location") != location
            ):
                failures.append(f"{label}: page scope lacks matching coverage provenance")
                continue
            if unit.get("kind") == "administrative":
                reason = row.get("reason")
                evidence = row.get("evidence")
                if (
                    row.get("disposition") not in {"excluded", "source-only"}
                    or not isinstance(reason, str)
                    or not reason.strip()
                    or not isinstance(evidence, str)
                    or not evidence.strip()
                ):
                    failures.append(
                        f"{label}: administrative page scope lacks explicit source-only/excluded reason/evidence"
                    )
                    continue
            covered_pages.update(pages)
        missing_pages = sorted(set(range(1, page_count + 1)) - covered_pages)
        if missing_pages:
            failures.append(
                f"{object_id}: semantic page closure missing pages {missing_pages}"
            )


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


def hub_markdown(
    lock: dict[str, Any],
    manifest: dict[str, Any],
    units: dict[str, Any],
    coverage: dict[str, Any],
    visuals: dict[str, Any],
) -> str:
    revisions = {item["repository"]: item["revision"] for item in lock["repositories"]}
    objects = {item["id"]: item for item in manifest["objects"]}
    legacy_anchors = {
        object_id: re.sub(
            r"[^a-z0-9_-]", "", f"{object_id}: {item['title']}".lower().replace(" ", "-")
        )
        for object_id, item in objects.items()
    }
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
        "[Официальная страница курса](https://cs336.stanford.edu/) ·",
        "[[05 Источники/Курсы|Все курсы]]",
        "",
        "Как обучить языковую модель с нуля и понять, на что уходят память, время",
        "и вычисления? Курс Percy Liang и Tatsunori Hashimoto разбирает эту задачу",
        "от токенизатора и устройства Transformer до подготовки корпуса,",
        "распределённого обучения и проверки качества модели. Пять заданий",
        "позволяют реализовать основные части этой системы самостоятельно.",
        "",
        "Ниже — материалы Spring 2026 на английском языке: оригинальные лекции,",
        "слайды и задания. Для каждой темы указаны соответствующие главы Bookvar",
        "и лекции с оригинальным разбором.",
        "",
        "## Где читать тему",
        "",
        "| Тема | Главы Bookvar | Лекции курса |",
        "|---|---|---|",
        "| Токенизатор и реализация Transformer | [[00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]]; [[00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Формы тензоров, параметры и стоимость]] | [1](#lecture-01), [2](#lecture-02), [3](#lecture-03) |",
        "| Альтернативы attention и MoE | [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts]]; [[00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Рекуррентные и гибридные модели]] | [4](#lecture-04) |",
        "| GPU, память и быстрые ядра | [[00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU и иерархия памяти]]; [[00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|Ядра и Triton]] | [5](#lecture-05), [6](#lecture-06) |",
        "| Распределённое обучение | [[00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Параллельное обучение]]; [[00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|Коллективные операции и DDP]] | [7](#lecture-07), [8](#lecture-08) |",
        "| Размер модели, объём данных и бюджет обучения | [[00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]] | [9](#lecture-09), [11](#lecture-11) |",
        "| Генерация и обслуживание запросов | [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Prefill, decode и roofline]]; [[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache и пакетирование]] | [10](#lecture-10) |",
        "| Оценивание и подготовка данных | [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей]]; [[00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]] | [12](#lecture-12), [13](#lecture-13), [14](#lecture-14) |",
        "| Дообучение по примерам, предпочтениям и проверяемой награде | [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT]]; [[00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]; [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR]] | [15](#lecture-15), [16](#lecture-16) |",
        "| Изображения и другие модальности | [[00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Мультимодальные модели]]; [[00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Обучение VLM]] | [17](#lecture-17) |",
        "",
        "## 19 встреч курса",
        "",
        "Открывайте название лекции для краткого описания и ссылок на материалы.",
        "Видеозаписи доступны через [официальное расписание](https://cs336.stanford.edu/#schedule).",
        "У двух гостевых встреч в сохранённой версии расписания нет ссылки на материалы.",
        "",
        "| № | Дата | Тема | Преподаватель |",
        "|---:|---|---|---|",
    ]
    for number, meeting_date, title, lecturer, filename in SCHEDULE:
        topic = f"[{title}](#lecture-{number:02d})" if filename else title
        lines.append(f"| {number} | {meeting_date} | {topic} | {lecturer} |")
    descriptions = {
        "lecture-01": "От байтов и Unicode к словарю токенов: почему разбиение текста влияет на длину последовательности и стоимость модели. Сравниваются простые способы токенизации и BPE; обучение токенизатора отделено от применения готового словаря.",
        "lecture-02": "Формы тензоров, операции PyTorch и einops, градиенты и шаг оптимизатора. На этих операциях разбирается подсчёт параметров, FLOPs и памяти: важно учитывать не только веса, но и активации, градиенты и состояние оптимизатора.",
        "lecture-03": "Выбор компонентов Transformer: нормализация, позиционные представления, функции активации и устройство attention. Архитектурные решения рассматриваются вместе с гиперпараметрами обучения, а не как независимый список приёмов.",
        "lecture-04": "Как уменьшить стоимость обработки длинных последовательностей и увеличить число параметров без пропорционального роста вычислений. Лекция сопоставляет альтернативы полному attention и разреженные MoE-модели с выбором экспертов для каждого токена.",
        "lecture-05": "Устройство GPU и TPU, матричные вычисления и движение данных между уровнями памяти. Пропускная способность памяти и вычислительная мощность ограничивают разные операции; это объясняет, почему число FLOPs само по себе не предсказывает время работы.",
        "lecture-06": "От операции PyTorch к отдельному GPU-ядру: измерение времени, объединение операций и разбиение работы на блоки в Triton. Примеры связывают организацию вычислений с числом обращений к памяти и фактическим ускорением.",
        "lecture-07": "Процессы, обмен тензорами и коллективные операции, необходимые для обучения на нескольких GPU. Разбирается, какие данные нужно передавать между устройствами и как согласовать локальные вычисления с синхронизацией.",
        "lecture-08": "Способы распределить данные, параметры и слои модели между устройствами. Сравнение учитывает память, объём обменов и простой устройств: одного увеличения числа GPU недостаточно, чтобы обучение ускорилось пропорционально.",
        "lecture-09": "Как по небольшим экспериментам оценить эффект увеличения модели и обучающего корпуса. Законы масштабирования связывают функцию потерь, число параметров, объём данных и вычислительный бюджет.",
        "lecture-10": "Чем обработка запроса отличается от последовательной генерации токенов и почему KV-cache меняет стоимость attention. Далее рассматриваются квантизация, сжатие, спекулятивное декодирование и обслуживание запросов разной длины.",
        "lecture-11": "Продолжение темы масштабирования: выбор соотношения размера модели и числа обучающих токенов. Экстраполяция требует нескольких измерений и проверки предположений, особенно когда меняются данные или режим обучения.",
        "lecture-12": "Что измеряет тест языковой модели и когда его результат можно сравнивать с другими моделями. Обсуждаются наборы задач, метрики, загрязнение тестовых данных и ограничения оценок, полученных с помощью другой языковой модели.",
        "lecture-13": "Откуда берутся данные для предобучения: веб, книги, научные тексты, код и специализированные наборы. Происхождение корпуса определяет его состав, доступность и ограничения использования.",
        "lecture-14": "Как превратить собранные документы в обучающий корпус: фильтрация, удаление повторов и выбор пропорций источников. Синтетические данные рассматриваются вместе с проверкой их качества, а не как автоматически полезное увеличение корпуса.",
        "lecture-15": "Как после предобучения научить модель выполнять инструкции и учитывать предпочтения. Обсуждаются обучающие примеры для SFT, сравнения ответов и оптимизация поведения модели после обучения модели награды.",
        "lecture-16": "Обучение с наградой, которую можно вычислить по проверке результата, например ответа на математическую задачу. Разбирается связь между генерацией решений, проверяющей программой и обновлением модели в RLVR.",
        "lecture-17": "Как связать текстовую модель с изображениями и другими модальностями. Разбираются представления входных данных, архитектурные способы их объединения и задачи обучения, согласующие разные представления.",
        "assignment-01": "Реализовать BPE-токенизатор, Transformer и оптимизатор, затем обучить небольшую языковую модель. Задание связывает формулы с работающим кодом: готовую реализацию модели вместо собственной использовать не предполагается.",
        "assignment-02": "Измерить узкие места модели из первого задания, реализовать FlashAttention-2 на Triton и распределённое обучение с экономией памяти. Результат проверяется не только корректностью, но и измерениями времени и памяти.",
        "assignment-03": "Спланировать серию обучений, подобрать закон масштабирования и предсказать подходящий размер модели при заданном бюджете. В README отдельно описаны работа через учебный API и самостоятельный запуск для читателей вне Stanford.",
        "assignment-04": "Подготовить данные для предобучения из Common Crawl: извлечь текст, отфильтровать документы и удалить повторы. Эффект обработки корпуса проверяется по обученной модели, а не только по числу оставшихся документов.",
        "assignment-05": "Дообучить модель на математических задачах с помощью SFT и обучения с подкреплением. Задание включает проверяемую награду и GRPO; в репозитории есть тесты, к которым нужно подключить собственную реализацию.",
        "assignment-05-safety-supplement": "Необязательное продолжение пятого задания: безопасность, выполнение инструкций и обучение по предпочтениям, в том числе DPO. Это дополнение к Assignment 5, а не отдельное шестое задание.",
    }
    lines.extend(["", "## Лекции: что читать и смотреть", ""])
    for number, _, title, _, filename in SCHEDULE:
        object_id = f"lecture-{number:02d}"
        if object_id in legacy_anchors:
            lines.append(f'<a id="{legacy_anchors[object_id]}"></a>')
        lines.extend([f'<a id="{object_id}"></a>', f"### {number}. {title}", ""])
        if not filename:
            lines.extend(["В сохранённом расписании ссылка на материалы не опубликована.", ""])
            continue
        lines.extend([descriptions[object_id], ""])
        if filename.endswith(".py"):
            lines.append(
                f"[Оригинальная лекция с кодом и иллюстрациями]"
                f"(https://cs336.stanford.edu/lectures/?trace=lecture_{number:02d}) · "
                f"[сохранённый Python-файл](Lectures/repository/{filename})"
            )
        else:
            lines.append(
                f"[Слайды PDF](Lectures/repository/{filename}) · "
                f"[оригинал в репозитории]({objects[object_id]['canonical_url']})"
            )
        lines.append("")
    lines.extend(["", "## Assignments 1–5", ""])
    for number, repository, title, _ in ASSIGNMENT_META:
        object_id = f"assignment-{number:02d}"
        lines.extend([
            f'<a id="{legacy_anchors[object_id]}"></a>',
            f'<a id="{object_id}"></a>',
            f"### Assignment {number}: {title}",
            "",
            descriptions[object_id],
            "",
            f"[Условия и запуск](Assignments/{repository}/README.md) · "
            f"[репозиторий задания]({objects[object_id]['canonical_url']})",
            "",
        ])
    lines.extend(
        [
            f'<a id="{legacy_anchors["assignment-05-safety-supplement"]}"></a>',
            '<a id="assignment-05-safety-supplement"></a>',
            "### Assignment 5: optional safety supplement",
            "",
            descriptions["assignment-05-safety-supplement"],
            "",
            "[Условия дополнения, PDF](Assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)",
            "",
            "## Об оригиналах и заимствованиях",
            "",
            "Тексты лекций и заданий сохранены на английском языке. Авторство и ссылки",
            "на источники сохраняются при переносе материала в главы. Условия использования",
            "кода, лекционных слайдов и рисунков из других публикаций могут различаться;",
            "доступность файла в интернете сама по себе не устанавливает его лицензию.",
            "",
            "<details>",
            "<summary>Для редакторов: состав архива, версии и проверка переноса</summary>",
            "",
            "## Реестры аудита",
            "",
            "Архив материалов сохранён; тематическое дополнение и редактура глав продолжаются.",
            "Технический статус: **inventory complete; editorial integration active**.",
            "Наличие записи в реестре не заменяет проверку качества объяснения или рисунка.",
            "",
            "- [source-manifest.yml](source-manifest.yml) — перечень оригинальных материалов;",
            "- [semantic-review.json](semantic-review.json) — границы смысловых разделов;",
            "- [editorial-map.yml](editorial-map.yml) — решения о переносе в главы;",
            "- [source-units.yml](source-units.yml) и [coverage.yml](coverage.yml) — фрагменты источников и их назначение;",
            "- [visuals.yml](visuals.yml) — иллюстрации и последовательности слайдов;",
            "- [artifact-inventory.json](artifact-inventory.json) — контрольные суммы файлов;",
            "- [snapshot-lock.json](snapshot-lock.json) — версии репозиториев и инструментов проверки.",
            "",
            "### Сохранённые версии репозиториев",
            "",
        ]
    )
    for repository in REPOSITORIES:
        lines.append(
            f"- `{repository.name}`: "
            f"[`{revisions[repository.name]}`]"
            f"(https://github.com/{GITHUB_ORG}/{repository.name}/tree/{revisions[repository.name]})"
        )
    lines.extend(
        [
            "",
            "### Права и атрибуция",
            "",
            "Запись `rights_status: permission-recorded` фиксирует пользовательское",
            "подтверждение образовательного переиспользования, а не название лицензии",
            "правообладателя. Файлы LICENSE сохранены вместе с репозиториями. Условия",
            "переноса сторонних рисунков проверяются отдельно для каждой иллюстрации.",
            "",
            "</details>",
            "",
        ]
    )
    return "\n".join(lines)


def write_generated_ledgers(lock: dict[str, Any], inventory: dict[str, Any]) -> None:
    documents_tuple = build_ledgers(lock)
    documents = generated_document_map(documents_tuple)
    closure_failures: list[str] = []
    validate_pdf_page_closure(
        documents["source-units.yml"],
        documents["coverage.yml"],
        closure_failures,
    )
    if closure_failures:
        raise RuntimeError(
            "Stanford CS336 reviewed-PDF closure failed:\n- "
            + "\n- ".join(closure_failures)
        )
    lock["generated_audit"] = generated_audit(documents, inventory)
    write_json_yaml(LOCK_PATH, lock)
    for filename, document in documents.items():
        write_json_yaml(COURSE_ROOT / filename, document)
    manifest, units, coverage, visuals = documents_tuple
    HUB_PATH.write_text(hub_markdown(lock, manifest, units, coverage, visuals), "utf-8")


def regenerate_ledgers() -> None:
    failures: list[str] = []
    lock = read_json(LOCK_PATH, failures)
    inventory = read_json(INVENTORY_PATH, failures)
    if failures:
        raise RuntimeError("\n".join(failures))
    write_generated_ledgers(lock, inventory)
    check()


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
    inventory = artifact_inventory()
    write_json_yaml(INVENTORY_PATH, inventory)
    write_generated_ledgers(lock, inventory)
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
    for path in (LOCK_PATH, INVENTORY_PATH, HUB_PATH, EDITORIAL_MAP_PATH):
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
    locked_archive = lock.get("generated_audit", {}).get("archive", {})
    require(
        isinstance(locked_archive, dict)
        and locked_archive.get("artifacts") == len(inventory_rows),
        "locked archive artifact count mismatch",
        failures,
    )
    archive_bytes = sum(
        row.get("bytes", 0)
        for row in inventory_rows
        if isinstance(row, dict) and isinstance(row.get("bytes"), int)
    )
    require(
        isinstance(locked_archive, dict)
        and locked_archive.get("bytes") == archive_bytes,
        "locked archive byte count mismatch",
        failures,
    )
    actual_documents = {
        "source-manifest.yml": manifest,
        "source-units.yml": source_units,
        "coverage.yml": coverage,
        "visuals.yml": visuals,
    }
    expected_documents = generated_document_map(build_ledgers(lock))
    validate_generated_documents(actual_documents, expected_documents, lock, failures)
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
    validate_pdf_page_closure(source_units, coverage, failures)
    unit_ids = [item.get("id") for item in units if isinstance(item, dict)]
    coverage_ids = [item.get("source_unit") for item in rows if isinstance(item, dict)]
    require(bool(units), "no extraction units were generated", failures)
    require(len(unit_ids) == len(set(unit_ids)), "duplicate extraction unit ids", failures)
    require(len(coverage_ids) == len(set(coverage_ids)), "duplicate coverage source_unit ids", failures)
    require(set(unit_ids) == set(coverage_ids), "every source unit must have exactly one coverage row", failures)
    forbidden_proxy_kinds = {"page", "rendered-text", "rendered-link"}
    require(
        forbidden_proxy_kinds.isdisjoint(
            {
                unit.get("kind")
                for unit in units
                if isinstance(unit, dict)
            }
        ),
        "proxy page/rendered-text/rendered-link units are forbidden",
        failures,
    )
    semantic_locations: set[tuple[Any, Any]] = set()
    units_by_id: dict[str, dict[str, Any]] = {}
    coverage_by_unit = {
        row.get("source_unit"): row
        for row in rows
        if isinstance(row, dict)
    }
    for unit in units if isinstance(units, list) else []:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("id")
        if isinstance(unit_id, str):
            units_by_id[unit_id] = unit
        location_key = (unit.get("source_object"), unit.get("source_location"))
        require(
            location_key not in semantic_locations,
            f"duplicate semantic source location: {location_key}",
            failures,
        )
        semantic_locations.add(location_key)
        if unit.get("kind") == "administrative":
            coverage_row_value = coverage_by_unit.get(unit_id)
            require(
                isinstance(coverage_row_value, dict)
                and coverage_row_value.get("disposition") in {"excluded", "source-only"}
                and isinstance(coverage_row_value.get("reason"), str)
                and bool(coverage_row_value.get("reason", "").strip())
                and isinstance(coverage_row_value.get("evidence"), str)
                and bool(coverage_row_value.get("evidence", "").strip()),
                f"administrative unit lacks source-only/excluded coverage reason/evidence: {unit_id}",
                failures,
            )
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
    visual_locations: set[tuple[Any, Any]] = set()
    has_multi_event = False
    has_multi_page = False
    for row in visual_rows if isinstance(visual_rows, list) else []:
        if isinstance(row, dict):
            require(row.get("source_object") in object_ids, f"orphan visual row: {row.get('id')}", failures)
            location_key = (row.get("source_object"), row.get("source_location"))
            require(
                location_key not in visual_locations,
                f"duplicate semantic visual location: {location_key}",
                failures,
            )
            visual_locations.add(location_key)
            require(
                isinstance(row.get("source_pages"), list) and bool(row.get("source_pages")),
                f"visual row lacks exact event/page scope: {row.get('id')}",
                failures,
            )
            linked_units = row.get("source_units")
            require(
                isinstance(linked_units, list) and bool(linked_units),
                f"visual row lacks semantic source-unit linkage: {row.get('id')}",
                failures,
            )
            for unit_id in linked_units if isinstance(linked_units, list) else []:
                linked_unit = units_by_id.get(unit_id)
                require(
                    linked_unit is not None
                    and linked_unit.get("source_object") == row.get("source_object"),
                    f"visual row links an unknown or foreign source unit: {row.get('id')}",
                    failures,
                )
            members = row.get("sequence_members")
            require(
                isinstance(members, list) and bool(members),
                f"visual row lacks sequence members: {row.get('id')}",
                failures,
            )
            require(
                isinstance(row.get("semantic_id"), str)
                and isinstance(row.get("visual_kind"), str),
                f"visual row lacks semantic classification: {row.get('id')}",
                failures,
            )
            require(
                row.get("extractor_sha256")
                == visuals.get("extraction_provenance", {}).get("extractor_sha256"),
                f"visual row extractor hash mismatch: {row.get('id')}",
                failures,
            )
            require(
                isinstance(row.get("parent_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", row["parent_sha256"]) is not None,
                f"visual row parent hash is invalid: {row.get('id')}",
                failures,
            )
            if isinstance(members, list) and len(members) > 1:
                if "var/traces" in str(row.get("source_location")):
                    has_multi_event = True
                if "#pages=" in str(row.get("source_location")):
                    has_multi_page = True
    require(has_multi_event, "visual ledger lacks a reviewed multi-event sequence", failures)
    require(has_multi_page, "visual ledger lacks a reviewed multi-page sequence", failures)
    lecture_six_tables = [
        row
        for row in visual_rows
        if isinstance(row, dict)
        and row.get("source_object") == "lecture-06"
        and row.get("semantic_id") == "accelerator-memory-hierarchy-table"
    ]
    require(
        len(lecture_six_tables) == 1
        and lecture_six_tables[0].get("visual_kind") == "table"
        and lecture_six_tables[0].get("source_pages") == list(range(8, 21))
        and len(lecture_six_tables[0].get("sequence_members", [])) == 13,
        "Lecture 6 accelerator table must be one 13-event sequence over steps 8-20",
        failures,
    )
    assignment_two_semantics = {
        row.get("semantic_id")
        for row in visual_rows
        if isinstance(row, dict) and row.get("source_object") == "assignment-02"
    }
    require(
        {"nsight-systems-trace", "distributed-rank-topology"}.issubset(
            assignment_two_semantics
        ),
        "Assignment 2 orphan raster semantics are missing",
        failures,
    )
    hub = HUB_PATH.read_text("utf-8")
    require("## 19 встреч курса" in hub, "hub does not expose all 19 meetings", failures)
    require("## Реестры аудита" in hub, "hub does not expose the ledgers", failures)
    require("inventory complete; editorial integration active" in hub, "hub status is missing", failures)
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
    modes.add_argument(
        "--regenerate-ledgers",
        action="store_true",
        help="regenerate semantic ledgers from the frozen archive without network access",
    )
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
        elif args.regenerate_ledgers:
            regenerate_ledgers()
        elif args.check:
            check()
        else:
            check_upstream_drift()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
