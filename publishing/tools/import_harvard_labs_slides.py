#!/usr/bin/env python3
"""Publish Harvard CS249r Marimo labs and slide sources as readable Markdown.

The generated page keeps the complete original Python/TeX source and adds a
readable layer extracted from Marimo markdown cells or Beamer frames.  This is
an archival source layer, not a translation or a replacement for canonical
Bookvar chapters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path

from harvard_beamer import parse_deck, project_tex, code_block


COMMIT = "45ecc8d82fcae70c149cdce550d3b3d3411df913"
REPOSITORY = "https://github.com/harvard-edge/cs249r_book"
LICENSE = "CC BY-NC-SA 4.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_from_path(path: Path) -> str:
    stem = re.sub(r"^\d+_", "", path.stem)
    stem = re.sub(r"^lab_\d+_", "", stem)
    return stem.replace("_", " ").strip().title()


def inside_markdown_math(text: str) -> bool:
    """Track dollar math while ignoring escaped dollars and Markdown code."""
    math = None
    code = None
    i = 0
    while i < len(text):
        if text[i] == '`':
            end = i
            while end < len(text) and text[end] == '`':
                end += 1
            run = end - i
            if code is None:
                code = run
            elif code == run:
                code = None
            i = end
        elif code is not None:
            i += 1
        elif text[i] == '\\':
            i += 2
        elif text[i] == '$':
            delimiter = '$$' if text.startswith('$$', i) else '$'
            if math is None:
                math = delimiter
            elif math == delimiter:
                math = None
            i += len(delimiter)
        else:
            i += 1
    return math is not None


def inline_code(text: str) -> str:
    runs = [len(match.group()) for match in re.finditer(r'`+', text)]
    fence = '`' * (max(runs, default=0) + 1)
    return f'{fence} {text} {fence}'


def escape_math_text_labels(markdown: str) -> str:
    """Escape literal underscores in display-math text, not in original code."""
    lines = []
    in_code = False
    in_math = False
    for line in markdown.splitlines():
        if line.lstrip().startswith('```'):
            in_code = not in_code
        elif not in_code and line.strip() == '$$':
            in_math = not in_math
        elif not in_code and in_math:
            line = re.sub(r'(\\text(?:tt)?\{)([^{}]*)(\})',
                lambda match: match[1] + re.sub(r'(?<!\\)_', r'\\_', match[2]) + match[3], line)
        lines.append(line)
    return '\n'.join(lines)


def extract_marimo_markdown(source: str) -> list[str]:
    """Decode authored string literals; name runtime fields without evaluating them.

    AST constants implement Python's escaping rules, including raw strings and
    doubled f-string braces. A lab's Python code, imports and expressions are
    never executed. Calls with a runtime-only argument remain in the full source.
    """
    calls = sorted((node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'mo'
        and node.func.attr == 'md' and node.args), key=lambda node: (node.lineno, node.col_offset))
    sections = []
    for call in calls:
        argument = call.args[0]
        fields = []
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            body = argument.value
        elif isinstance(argument, ast.JoinedStr):
            body = ''
            for part in argument.values:
                if isinstance(part, ast.Constant):
                    body += part.value
                elif isinstance(part, ast.FormattedValue):
                    number = len(fields) + 1
                    # Preserve the expression, conversion and format specification
                    # as Python source, not as an invented numeric result.
                    fields.append(ast.unparse(ast.JoinedStr(values=[part])))
                    body += (rf'\mathrm{{value}}_{{{number}}}' if inside_markdown_math(body)
                             else f'⟨value {number}⟩')
        else:
            continue
        body = textwrap.dedent(body).strip()
        if fields:
            body += '\n\nRuntime fields (not evaluated):\n\n' + '\n'.join(
                f'- value_{number}: {inline_code(expression)}'
                for number, expression in enumerate(fields, start=1))
        if body:
            sections.append(body)
    return sections


def clean_tex(text: str) -> str:
    return project_tex(text)


def extract_beamer_frames(source: str) -> list[tuple[str, str]]:
    return [(slide.title, slide.body) for slide in parse_deck(source)]


def slide_image_resolver(root: Path, relative: str):
    manifest = root / '05 Источники/Courses/Harvard ML Systems/slide-assets-manifest.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    lookup = {ref['tex_reference']: row for row in records for ref in row['references']
              if ref['deck'] == relative}

    def resolve(name: str):
        row = lookup.get(name)
        if not row or row['state'] != 'local-original':
            return None
        local = root / row['local_path']
        if hashlib.sha256(local.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Original SVG hash mismatch: ' + row['local_path'])
        path = row['local_path']
        url = f"{REPOSITORY}/blob/{COMMIT}/{row['source']}"
        caption = (f"Harvard Edge ML Systems Book contributors, [{Path(row['source']).name}]({url}), "
                   f"[{LICENSE}]({REPOSITORY}/blob/{COMMIT}/LICENSE.md). "
                   'Original SVG, copied without redrawing or cropping.')
        return path, caption
    return resolve, lookup


def render_slides(source: str, relative: str, root: Path) -> tuple[str, dict]:
    path = Path(relative)
    title = f"{path.parents[1].name.upper()} Slides — {title_from_path(path.parent)}"
    resolver, images = slide_image_resolver(root, relative)
    slides = parse_deck(source, resolver)
    review_file = root / '05 Источники/Courses/Harvard ML Systems/slide-source-review.json'
    reviews = json.loads(review_file.read_text()) if review_file.exists() else []
    reviews = [row for row in reviews if row['source'] == relative]
    applied_reviews = []
    sequence = []
    warnings = []
    for number, slide in enumerate(slides, 1):
        source_url = f'{REPOSITORY}/blob/{COMMIT}/{relative}#L{slide.line}'
        # Keep headings stable where possible; no generated lecture/script text
        # is promoted to an editorial claim about the canonical textbook.
        sequence.append(f'## {slide.title}\n\n[Original slide {number}]({source_url})\n\n{slide.body}')
        for review in (row for row in reviews if row['title'] == slide.title):
            if (review['source_commit'] != COMMIT or
                    hashlib.sha256(slide.original.encode()).hexdigest() != review['slide_sha256']):
                raise ValueError('Stale source review: ' + relative + ': ' + slide.title)
            sequence.append('> [!warning] ' + review['heading'] + '\n> ' + review['text'])
            applied_reviews.append(slide.title)
        if slide.notes:
            notes = '\n\n'.join(code_block(note).strip() for note in slide.notes)
            sequence.append('<details>\n<summary>Original speaker notes (TeX)</summary>\n\n' + notes + '\n\n</details>')
        if slide.warnings:
            lines = []
            for warning in slide.warnings:
                if warning.startswith('Unresolved image: '):
                    name = warning.removeprefix('Unresolved image: ')
                    row = images.get(name)
                    if row:
                        lines.append(f"- [{name}]({row['source_url']}): raster source link retained; individual provenance not established.")
                    else:
                        lines.append('- ' + inline_code(warning))
                else:
                    lines.append('- ' + inline_code(warning))
            sequence.append('<details>\n<summary>Projection limits for this slide</summary>\n\n'
                            + '\n'.join(lines) + '\n\n</details>')
            warnings.append({'slide': number, 'line': slide.line, 'title': slide.title,
                             'items': slide.warnings})
    text = (frontmatter(title, relative, 'beamer-slides')
            + '> [!note] Static slide archive\n'
            + '> Text and equations retain the original language. Layout wrappers are simplified;\n'
            + '> speaker notes are separate, and original SVGs are not redrawn or cropped.\n'
            + '> Unsupported TeX remains visible and is listed beside the affected slide.\n'
            + '> This archive is not a checked transcript or a finished Bookvar chapter.\n\n'
            + f'# {title}\n\n## Readable slide sequence\n\n' + '\n\n'.join(sequence)
            + '\n\n## Complete original Beamer source\n\n'
            + f'```tex\n{source.rstrip()}\n```\n')
    if len(applied_reviews) != len(reviews):
        raise ValueError('Unmatched source review: ' + relative)
    return text, {'source': relative, 'explicit_frames': sum(s.kind == 'frame' for s in slides),
                  'title_slides': sum(s.kind == 'mlsystitle' for s in slides),
                  'focus_slides': sum(s.kind == 'mlsysfocus' for s in slides),
                  'speaker_notes': sum(len(s.notes) for s in slides),
                  'source_issues': applied_reviews,
                  'projection_limits': warnings}


def rebuild_archived_slides(root: Path, check: bool = False) -> None:
    from prepare_harvard_slide_assets import source_from_page
    manifest = root / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'
    rows = [r for r in json.loads(manifest.read_text()) if r['kind'] == 'slides']
    pages = []
    report = []
    for row in rows:
        page, audit = render_slides(source_from_page(root, row), row['source'], root)
        pages.append((root / row['output'], page))
        report.append(audit)
    report_path = root / '05 Источники/Courses/Harvard ML Systems/slide-projection-audit.json'
    pages.append((report_path, json.dumps(report, ensure_ascii=False, indent=2) + '\n'))
    for path, expected in pages:
        if check:
            if path.read_text() != expected:
                raise ValueError('Stale slide projection: ' + str(path.relative_to(root)))
        else:
            path.write_text(expected, encoding='utf-8')
    print(f"{'Verified' if check else 'Rebuilt'} {len(rows)} Harvard slide archives; "
          f"{sum(r['explicit_frames'] for r in report)} explicit frames, "
          f"{sum(r['focus_slides'] for r in report)} focus slides, original hashes unchanged.")


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
> License: [{LICENSE}]({REPOSITORY}/blob/{COMMIT}/LICENSE.md).
> The readable layer below is mechanically extracted from the original;
> the complete unmodified source follows on the same page.

"""


def render_lab(source: str, relative: str) -> str:
    path = Path(relative)
    title = f"{path.parent.name.upper()} Lab — {title_from_path(path)}"
    prose = extract_marimo_markdown(source)
    # Keep unconditional-looking OOM/success messages out of the explanation.
    # All single-line UI strings and unevaluated templates remain accessible,
    # together with the complete original source and its actual control flow.
    narrative = [body for body in prose if '\n' in body and 'Runtime fields (not evaluated):' not in body]
    templates = [body for body in prose if body not in narrative]
    readable = "\n\n---\n\n".join(escape_math_text_labels(body) for body in narrative) if narrative else (
        "This lab is primarily executable code; use the complete source below."
    )
    if templates:
        readable += ('\n\n<details>\n<summary>UI text and runtime templates (not executed)</summary>\n\n'
                     + '\n\n---\n\n'.join(escape_math_text_labels(body) for body in templates) + '\n\n</details>')
    return (
        frontmatter(title, relative, "marimo-lab")
        + '> [!note] Static text, not a running lab\n'
        + '> Python string literals are decoded without executing the lab.\n'
        + '> Literal underscores in display-math text labels are escaped for KaTeX.\n'
        + '> Runtime fields are named below each excerpt; they are not measured results.\n'
        + '> The explanation shows literal multiline Markdown; other UI text and\n'
        + '> unevaluated templates are collapsed separately. Conditional branches\n'
        + '> can appear together there. Runtime-only\n'
        + '> messages, HTML interfaces and plots remain in the complete source.\n\n'
        + f"# {title}\n\n## Readable lab narrative\n\n{readable}\n\n"
        + "## Complete original Marimo source\n\n"
        + f"```python\n{source.rstrip()}\n```\n"
    )


def archived_lab_source(root: Path, record: dict[str, str]) -> str:
    """Recover the existing byte-verified original; do not re-download a course."""
    page = (root / record['output']).read_text('utf-8')
    marker = '## Complete original Marimo source\n\n```python\n'
    if page.count(marker) != 1 or not page.endswith('\n```\n'):
        raise ValueError(f"Missing complete original source: {record['output']}")
    source = page.split(marker, 1)[1].removesuffix('\n```\n') + '\n'
    if hashlib.sha256(source.encode()).hexdigest() != record['sha256']:
        raise ValueError(f"Original source hash mismatch: {record['source']}")
    return source


def rebuild_archived_labs(root: Path, check: bool = False) -> None:
    manifest = root / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'
    records = [row for row in json.loads(manifest.read_text('utf-8')) if row['kind'] == 'lab']
    # Validate all originals and prepare all pages before changing any output.
    rendered = [(root / row['output'], render_lab(archived_lab_source(root, row), row['source']))
                for row in records]
    for path, expected in rendered:
        if check:
            if path.read_text('utf-8') != expected:
                raise ValueError(f"Stale static extraction: {path.relative_to(root)}")
        else:
            path.write_text(expected, encoding='utf-8')
    print(f"{'Verified' if check else 'Rebuilt'} {len(rendered)} Harvard lab pages; original source hashes unchanged.")


def import_lab(root: Path, source_root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(source_root).as_posix()
    volume = path.parent.name
    title = f"{volume.upper()} Lab — {title_from_path(path)}"
    source = path.read_text(encoding="utf-8")
    output = (
        root
        / "05 Источники/Courses/Harvard ML Systems"
        / "labs"
        / volume
        / f"{path.stem}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_lab(source, relative), encoding="utf-8")
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
    output = (
        root
        / "05 Источники/Courses/Harvard ML Systems"
        / "slides"
        / volume
        / f"{lecture}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    page, _ = render_slides(source, relative, root)
    output.write_text(page, encoding='utf-8')
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
        f"Original English archives from commit `{COMMIT}`:",
        '34 Marimo labs and 35 Beamer decks. Each page retains the complete, hash-verified',
        'source alongside a static projection. The labs do not execute on this site;',
        'their runtime interfaces and plots remain in the Python source.',
        '',
        'The slide archives include 269 unmodified source SVGs, equations, tables and',
        'separate speaker-note drawers. Fourteen raster references retain direct links',
        'to the pinned originals. Projection limits are listed beside the affected',
        'slides. These are course-source archives, not independently checked textbook',
        'explanations or a claim that every course topic has been integrated into Bookvar.',
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
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--rebuild-labs", action="store_true",
                        help="Rebuild static lab prose from the existing hash-verified originals")
    parser.add_argument('--rebuild-slides', action='store_true',
                        help='Rebuild slide text/math/notes and native figure links from verified originals')
    parser.add_argument("--check", action="store_true", help="Check archived extractions without writing")
    parser.add_argument("--update-navigation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.rebuild_labs or args.rebuild_slides:
        if args.source_root or args.update_navigation:
            parser.error('Archive rebuilds do not use --source-root or --update-navigation')
        if args.rebuild_labs:
            rebuild_archived_labs(root, check=args.check)
        if args.rebuild_slides:
            rebuild_archived_slides(root, check=args.check)
        return
    if not args.source_root or args.check:
        parser.error('Use --source-root for a new import, or --rebuild-labs/--rebuild-slides [--check]')
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
