"""Restore two separately reviewed MLOps teaching blocks, not arbitrary edits."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def before_serving_decisions(text, source):
    row = json.loads((ROOT / 'docs/audit/2026-10-29-serving-decisions-additions.json').read_text())
    source = Path(source).relative_to(ROOT).as_posix() if Path(source).is_absolute() else str(source)
    if source != row['source']:
        return text
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    if digest(text) == row['before_sha256']:
        return text
    for block in row['blocks']:
        pattern = (r'<!-- serving-decisions-completion: ' + block['label'] + r' start -->'
                   r'[\s\S]*?<!-- serving-decisions-completion: ' + block['label'] + r' end -->\n\n')
        matches = re.findall(pattern, text)
        if len(matches) != 1 or digest(matches[0]) != block['sha256']:
            raise AssertionError(f'Unreviewed serving decision addition: {source}/{block["label"]}')
        text = text.replace(matches[0], '', 1)
    return text
