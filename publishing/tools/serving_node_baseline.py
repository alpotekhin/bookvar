"""Exact removal of the independently checked host placement/loading addition."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def before_serving_node(text, source):
    row = json.loads((ROOT / 'docs/audit/2026-10-29-serving-node-additions.json').read_text())
    source = Path(source).relative_to(ROOT).as_posix() if Path(source).is_absolute() else str(source)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    if source != row['source'] or digest(text) == row['before_sha256']:
        return text
    pattern = r'<!-- serving-node-completion: host-loading start -->[\s\S]*?<!-- serving-node-completion: host-loading end -->\n\n'
    blocks = re.findall(pattern, text)
    if len(blocks) != 1 or digest(blocks[0]) != row['block_sha256']:
        raise AssertionError(f'Unreviewed serving-node addition: {source}')
    return text.replace(blocks[0], '', 1)
