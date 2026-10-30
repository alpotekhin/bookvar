#!/usr/bin/env python3
"""Fetch only referenced native SVGs from the pinned Harvard course tree.

No TeX is executed and no figure is redrawn. Photographs retain direct source
links: a repository-wide notice does not establish their individual provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import posixpath
import re
import urllib.request
import xml.etree.ElementTree as ET

from harvard_beamer import parse_deck, strip_comments

COMMIT = '45ecc8d82fcae70c149cdce550d3b3d3411df913'
BASE = 'https://raw.githubusercontent.com/harvard-edge/cs249r_book/' + COMMIT + '/'
MANIFEST = '05 Источники/Courses/Harvard ML Systems/slide-assets-manifest.json'


def source_from_page(root: Path, row: dict) -> str:
    page = (root / row['output']).read_text('utf-8')
    source = page.split('## Complete original Beamer source\n\n```tex\n', 1)[1].removesuffix('\n```\n') + '\n'
    if hashlib.sha256(source.encode()).hexdigest() != row['sha256']:
        raise ValueError('Original TeX hash mismatch: ' + row['source'])
    return source


def inspect_svg(data: bytes) -> None:
    node = ET.fromstring(data)
    if node.tag.rsplit('}', 1)[-1] != 'svg':
        raise ValueError('Not SVG')
    for item in node.iter():
        if item.tag.rsplit('}', 1)[-1] in {'script', 'foreignObject'}:
            raise ValueError('Active content in course SVG')
        for key, value in item.attrib.items():
            if key.lower().startswith('on'):
                raise ValueError('Event handler in SVG')
            if key.rsplit('}', 1)[-1] == 'href' and not value.startswith(('#', 'data:')):
                raise ValueError('External dependency in SVG: ' + value)


def prepare(root: Path, tree: list[dict], check: bool = False) -> None:
    available = {r['path']: r for r in tree}
    slides = json.loads((root / '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json').read_text())
    refs = {}
    for row in slides:
        if row['kind'] != 'slides':
            continue
        source = source_from_page(root, row)
        for slide in parse_deck(source):
            for match in re.finditer(r'\\(?:safeimg|includegraphics)(?:\[[^\]]*\])?\{([^{}]+)\}', strip_comments(slide.original)):
                name = match[1]
                parent = str(Path(row['source']).parent)
                candidates = [posixpath.normpath(parent + '/images/' + name), posixpath.normpath(parent + '/' + name)]
                if name.endswith('.pdf'):
                    candidates = [str(Path(c).with_suffix('.svg')) for c in candidates] + candidates
                found = next((c for c in candidates if c in available), None)
                if found is None:
                    raise ValueError('Missing pinned image: ' + row['source'] + ': ' + name)
                record = refs.setdefault(found, {'source': found, 'git_blob_sha1': available[found]['sha'],
                    'source_commit': COMMIT, 'source_url': BASE + found,
                    'license': 'CC BY-NC-SA 4.0 (repository notice)', 'references': []})
                ref = {'deck': row['source'], 'tex_reference': name}
                if ref not in record['references']:
                    record['references'].append(ref)

    def fetch(record: dict) -> dict:
        source = record['source']
        if not source.endswith('.svg'):
            return {**record, 'state': 'external-source-link',
                    'reason': 'Photograph/raster provenance not independently established; original link retained.'}
        target = Path('Assets/Sources/Harvard ML Systems') / source
        path = root / target
        if path.exists():
            data = path.read_bytes()
        elif check:
            raise ValueError('Missing local original: ' + str(target))
        else:
            request = urllib.request.Request(record['source_url'], headers={'User-Agent': 'Bookvar-source-archive/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != record['git_blob_sha1']:
            raise ValueError('Pinned Git blob mismatch: ' + source)
        inspect_svg(data)
        if not check and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {**record, 'state': 'local-original', 'local_path': target.as_posix(),
                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, [refs[key] for key in sorted(refs)]))
    target = root / MANIFEST
    serialized = json.dumps(records, ensure_ascii=False, indent=2) + '\n'
    if check:
        if target.read_text() != serialized:
            raise ValueError('Stale slide asset manifest')
    else:
        target.write_text(serialized, encoding='utf-8')
    local = [r for r in records if r['state'] == 'local-original']
    print(f'{len(local)} byte-verified original SVGs ({sum(r["bytes"] for r in local)} bytes); '
          f'{len(records) - len(local)} raster references remain explicit source links.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--tree', type=Path,
                        help='Pinned Git tree response; defaults to the archived course metadata')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    tree = args.tree or root / '05 Источники/Courses/Harvard ML Systems/slide-assets-tree.json'
    prepare(root, json.loads(tree.read_text()), args.check)
