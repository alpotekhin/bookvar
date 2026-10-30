#!/usr/bin/env python3
"""Execute explicitly selected displayed CPU programs; never a hardware benchmark."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / 'docs/audit/2026-10-29-en-transformer-models-parity.json'


def blocks(text):
    return [m.group(2) for m in re.finditer(
        r'^(```|~~~)python\n(.*?)\n\1', text, re.S | re.M)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter', default='')
    args = parser.parse_args()
    results = []
    journal = json.loads(JOURNAL.read_text())
    for row in journal['entries']:
        if not row['complete'] or args.chapter not in row['path']:
            continue
        text = (ROOT / row['path']).read_text()
        source = (ROOT / row['canonical']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == row['after_sha256']
        assert hashlib.sha256(source.encode()).hexdigest() == row['canonical_sha256']
        en, canonical = blocks(text), blocks(source)
        assert len(en) == len(canonical)
        selected = [index for group in row['code_groups'] for index in group]
        assert sorted(selected) == list(range(len(en)))
        # Comments may be translated; executable statements remain identical.
        for actual, expected in zip(en, canonical):
            assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
        ran = []
        for group in row['code_groups']:
            ns = {'__name__': '__main__'}
            for index in group:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(en[index], row['path'] + f':block{index}', 'exec'), ns)
                ran.append({'block': index, 'stdout': output.getvalue()})
        results.append({'path': row['path'],
                        'sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'executed_cpu_blocks': ran,
                        'not_executed': row['not_executed']})
    assert results, 'No completed chapter matched'
    print(json.dumps({'scope': 'Displayed CPU programs only; no accelerator timing',
                      'chapters': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
