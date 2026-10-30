#!/usr/bin/env python3
"""Check the printed feature-history examples; --sql runs them in DuckDB 1.4.1.

No feature-store service, backfill deployment or model is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
SOURCES = ROOT / '05 Источники/Originals/mlops-2026-10-20'
ASSETS = ROOT / '00 Учебник/Assets/Figures/curated/feature-stores-2026-10-26/assets.json'
OLD_SHA = '0e63a2dde76d8eae9630036aac9542f63c48cf0f15cb98a659ca2dd5fa76f454'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return ' '.join(value.split())


def excerpt(chapter, key):
    match = re.search(r'<!-- source-excerpt: ' + key + r' -->\n(.*?)\n<!-- /source-excerpt -->',
                      chapter, re.S)
    assert match, key
    return '\n'.join(re.sub(r'^> ?', '', line) for line in match[1].splitlines())


def main(run_sql=False):
    current_chapter = CHAPTER.read_text()
    reviewed_chapter = before_serving_operations(current_chapter, CHAPTER)
    # Later independently checked extension; its own checker restores our accepted SHA.
    chapter = re.sub(r'<!-- BEGIN REVIEW: mlops (platform-resources|finops|fleet-alerts) 2026-10-26 -->\n'
                     r'.*?<!-- END REVIEW: mlops \1 2026-10-26 -->\n\n',
                     '', reviewed_chapter, flags=re.S)
    old = re.sub(r'<!-- BEGIN REVIEW: mlops (feature-history|feature-migration) 2026-10-26 -->\n'
                 r'.*?<!-- END REVIEW: mlops \1 2026-10-26 -->\n\n', '', chapter, flags=re.S)
    old = old.replace('last_updated: 2026-10-26', 'last_updated: 2026-10-20')
    old = old.replace('last_verified: 2026-10-26', 'last_verified: 2026-10-20')
    assert sha(old.encode()) == OLD_SHA, 'accepted text changed outside the additions'

    vol1 = (SOURCES / 'harvard-ml-ops.qmd').read_text()
    vol2 = (SOURCES / 'harvard-ops-scale.qmd').read_text()
    rendered_reference = vol1.replace('@sec-ml-operations-technical-debt-system-complexity-2762',
                                      'the technical debt section')
    assert compact(excerpt(chapter, 'harvard-feature-definition-skew')) in compact(rendered_reference)
    for key in ['harvard-feature-version-incident', 'harvard-incremental-backfill']:
        assert compact(excerpt(chapter, key)) in compact(vol2), key
    sql = re.findall(r'^```sql\n(.*?)^```', chapter, re.M | re.S)
    original_sql = re.search(r'```\{\.sql\}\n(.*?)\n```', vol2[vol2.index('### Point-in-time correctness'):], re.S)[1]
    assert len(sql) == 3 and sql[0].rstrip() == original_sql
    table = excerpt(chapter, 'harvard-feature-lineage-table').splitlines()[2:]
    source_table = vol2[vol2.index('| **Component**'):].split('\n\n', 1)[0].splitlines()[2:]
    cells = lambda lines: [[compact(re.sub(r'<[^>]+>', '', cell).replace('**', '')) for cell in line.strip('|').split('|')]
                           for line in lines]
    assert cells(table) == cells(source_table)

    additions = re.findall(r'<!-- BEGIN REVIEW: mlops .*?-->\n(.*?)<!-- END REVIEW: mlops .*?-->',
                            chapter, re.S)
    blocks = re.findall(r'^```python\n(.*?)^```', '\n'.join(additions), re.M | re.S)
    assert len(blocks) == 2
    namespace = {}
    for i, block in enumerate(blocks):
        exec(compile(block, str(CHAPTER) + f':extension-{i}', 'exec'), namespace)
    assert namespace['retained_bytes'] == 100_000_000 * 1000 * 8760 * 100
    gate = namespace['backfill_blockers']
    rng = random.Random(20261026)
    for _ in range(300):
        days = [f'day-{i}' for i in range(rng.randrange(1, 20))]
        receipts, expected = {}, []
        for day in days:
            state = rng.choice(['missing', 'different-inputs', 'failed', 'passed'])
            if state != 'missing':
                receipts[day] = {'definition': 'v2', 'snapshot': 'A', 'passed': state != 'failed'}
                if state == 'different-inputs':
                    receipts[day][rng.choice(['definition', 'snapshot'])] = 'other'
            if state != 'passed':
                expected.append((day, state))
        assert gate(days, receipts, 'v2', 'A') == expected
    for args in [([], {}, 'v2', 'A'), (['d', 'd'], {}, 'v2', 'A'),
                 (['d'], {}, '', 'A'), (['d'], {}, 'v2', '')]:
        try:
            gate(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid migration contract accepted')
    assert gate(['d'], {'d': {'definition': 'v2', 'snapshot': 'A', 'passed': 1}},
                'v2', 'A') == [('d', 'failed')]
    assert (40 - 30) / 10 == 1 and (40 - 0) / (100 - 0) == .4
    assert (40 - 30) / 10 > .8 > (40 - 0) / (100 - 0)

    assets = json.loads(ASSETS.read_text())
    for asset in assets:
        for path_key, hash_key in [('asset', 'sha256'), ('source_asset', 'source_asset_sha256'),
                                   ('source_page_snapshot', 'source_page_sha256'),
                                   ('license_evidence', 'license_evidence_sha256')]:
            assert sha((ROOT / asset[path_key]).read_bytes()) == asset[hash_key]
        assert (ROOT / asset['asset']).read_bytes() == (ROOT / asset['source_asset']).read_bytes()
        assert asset['asset'] in chapter
    assert len(assets) == 3
    assert 'feature-store-architecture.svg' not in chapter

    report = {'chapter_sha256': sha(current_chapter.encode()), 'accepted_chapter_preserved': True,
              'executed_new_python_blocks': len(blocks), 'partition_cases': 300,
              'retained_bytes': namespace['retained_bytes'],
              'source_excerpts': 5, 'unchanged_source_figures': 3,
              'sql_status': 'not-run; supply --sql with duckdb==1.4.1',
              'scope': 'Teaching examples only; no Feast backend, online store, model or backfill job'}
    if run_sql:
        import duckdb
        assert duckdb.__version__ == '1.4.1', duckdb.__version__
        connection = duckdb.connect(':memory:')
        connection.execute(sql[1])
        source_rows = sorted(connection.execute(sql[0]).fetchall())
        assert source_rows[0][:3] == (101, 720, 0) and source_rows[0][3] in (4, 6)
        assert source_rows[1][3:] == (8, 80) and source_rows[2][3:] == (8, 80)
        assert source_rows[3][3:] == (None, None)
        expected_fixture = [(101, 720, 0, 3, 30), (101, 780, 1, 8, 80),
                            (101, 900, 0, None, None), (102, 720, 1, None, None)]
        assert connection.execute(sql[2]).fetchall() == expected_fixture

        # Independent in-memory selection oracle, including other versions/snapshots,
        # equal-time boundaries, missing entities, late rows and correction ties.
        for _ in range(200):
            connection.execute('DELETE FROM features')
            keys, rows = set(), []
            for i in range(rng.randrange(1, 40)):
                user = rng.choice([101, 102, 103])
                event_time = rng.choice([660, 690, 720, 780, 840, 900, rng.randrange(600, 940)])
                available = event_time + rng.randrange(0, 140)
                definition, snapshot = rng.choice(['clicks:v1', 'other']), rng.choice(['source:A', 'other'])
                key = (user, event_time, available, definition, snapshot)
                if key not in keys:
                    keys.add(key)
                    rows.append((*key, i, i * 10))
            connection.executemany('INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
            expected = []
            for user, at, label in [(101, 720, 0), (101, 780, 1), (101, 900, 0), (102, 720, 1)]:
                eligible = [r for r in rows if r[0] == user and r[3:5] == ('clicks:v1', 'source:A')
                            and at - 60 <= r[1] <= at and r[2] <= at]
                selected = max(eligible, key=lambda r: (r[1], r[2])) if eligible else None
                expected.append((user, at, label, *(selected[5:7] if selected else (None, None))))
            assert connection.execute(sql[2]).fetchall() == expected
        connection.close()
        report.update(sql_status='executed', duckdb_version=duckdb.__version__,
                      visible_sql_blocks=3, random_histories=200, oracle_prediction_rows=800)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql', action='store_true')
    main(parser.parse_args().sql)
