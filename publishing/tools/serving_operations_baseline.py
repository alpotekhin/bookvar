"""Bounded restoration for pre-existing chapter arithmetic/source checks."""
import hashlib
import json
from pathlib import Path
import re
from serving_decisions_baseline import before_serving_decisions
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]


def before_serving_operations(text, source):
    text = before_topic_consolidation(text, source)
    text = before_serving_decisions(text, source)
    rows = json.loads((ROOT / 'docs/audit/2026-10-29-serving-operations-additions.json').read_text())['chapters']
    source = Path(source).relative_to(ROOT).as_posix() if Path(source).is_absolute() else str(source)
    row = next((row for row in rows if row['source'] == source), None)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    if row is None or digest(text) == row['before_sha256']:
        return text
    pattern = (r'<!-- serving-operations-completion: ' + row['label'] + r' start -->'
               r'[\s\S]*?<!-- serving-operations-completion: ' + row['label'] + r' end -->\n\n')
    blocks = re.findall(pattern, text)
    if len(blocks) != 1 or digest(blocks[0]) != row['block_sha256']:
        raise AssertionError(f'Unreviewed serving operations addition: {source}')
    return text.replace(blocks[0], '', 1)
