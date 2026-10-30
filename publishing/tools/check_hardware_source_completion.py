#!/usr/bin/env python3
"""Check the actual hardware chapter's CPU examples and bounded addition.

No CUDA, TPU, NPU, memory-migration, energy or performance experiment is run.
"""
import hashlib
import itertools
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / 'docs/audit/2026-10-29-hardware-source-comparison.json'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def before_hardware_source_completion(text, source):
    row = json.loads(JOURNAL.read_text())['addition']
    if str(source) != row['source'] or digest(text) == row['before_sha256']:
        return text
    pattern = (r'<!-- hardware-source-completion: mapping:start -->[\s\S]*?'
               r'<!-- hardware-source-completion: mapping:end -->\n\n')
    matches = re.findall(pattern, text)
    assert len(matches) == 1 and digest(matches[0]) == row['block_sha256']
    return text.replace(matches[0], '', 1)


def main():
    row = json.loads(JOURNAL.read_text())['addition']
    text = (ROOT / row['source']).read_text()
    assert digest(text) == row['after_sha256']
    assert digest(before_hardware_source_completion(text, row['source'])) == row['before_sha256']
    examples = {}
    for ident, code in re.findall(
            r'<!-- checked-example: (hardware-(?:layout|im2col)) -->\n```python\n(.*?)\n```',
            text, re.S):
        namespace = {}
        exec(compile(code, f"{row['source']}:{ident}", 'exec'), namespace)
        examples[ident] = namespace
    assert set(examples) == {'hardware-layout', 'hardware-im2col'}
    layout = examples['hardware-layout']
    import torch
    # Exhaustively check all addresses, not only the five rows in the table.
    x, y = layout['x'], layout['y']
    physical_y = y.permute(0, 2, 3, 1).reshape(-1)
    for index in itertools.product(range(2), range(3), range(2), range(4)):
        assert physical_y[layout['offset'](index, y.stride())] == x[index]
    # Distinguish a view's shape change from a storage-format conversion.
    assert layout['nhwc_view'].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert layout['packed_nhwc'].untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    print(json.dumps({'displayed_examples': list(examples), 'checked_layout_elements': 48,
                      'torch': torch.__version__, 'device': 'cpu',
                      'scope': 'values, shapes, strides and exact text preservation only'}, indent=2))


if __name__ == '__main__':
    main()
