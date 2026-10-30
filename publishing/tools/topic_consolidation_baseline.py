"""Restore only hash-pinned, reviewed topic-consolidation changes for old checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNALS = (
    '2026-10-30-clean-checkout-additions.json',
    '2026-10-30-publication-delivery-additions.json',
    '2026-10-30-figure-replacements-additions.json',
    '2026-10-30-atlas-release-label-additions.json',
    '2026-10-30-atlas-gqa-label-additions.json',
    '2026-10-30-en-scheduler-navigation-additions.json',
    '2026-10-30-scheduler-practice-link-additions.json',
    '2026-10-30-decoding-prerequisites-additions.json',
    '2026-10-30-reader-provenance-additions.json',
    '2026-10-30-modern-route-editorial-additions.json',
    '2026-10-30-posttraining-order-additions.json',
    '2026-10-30-locale-navigation-additions.json',
    '2026-10-30-reference-index-metadata-additions.json',
    '2026-10-30-bleu-reconciliation-additions.json',
    '2026-10-30-tot-reconciliation-additions.json',
    '2026-10-30-agent-reconciliation-additions.json',
    '2026-10-30-nanogpt-additions.json',
    '2026-10-29-synthetic-recipes-additions.json',
    '2026-10-29-semi-extensions-additions.json',
    '2026-10-29-consistency-training-additions.json',
    '2026-10-29-corpus-releases-additions.json',
    '2026-10-29-tool-rewards-additions.json',
    '2026-10-29-constitutional-ai-additions.json',
    '2026-10-29-structured-dropout-additions.json',
    '2026-10-29-adaptive-clipping-additions.json',
    '2026-10-29-retention-additions.json',
    '2026-10-29-bert-objectives-additions.json',
    '2026-10-29-small-data-finetuning-additions.json',
    '2026-10-29-peft-additions.json',
    '2026-10-29-scheduler-lab-additions.json',
    '2026-10-29-coconut-additions.json',
    '2026-10-29-masked-diffusion-additions.json',
    '2026-10-29-security-prose-additions.json',
    '2026-10-29-topic-navigation-followup-additions.json',
    '2026-10-29-topic-navigation-additions.json',
    '2026-10-29-architecture-integration-additions.json',
    '2026-10-29-atlas-consolidation-additions.json',
    '2026-10-29-security-consolidation-additions.json',
    '2026-10-29-attention-consolidation-additions.json',
    '2026-10-29-training-consolidation-additions.json',
    '2026-10-29-inference-consolidation-additions.json',
)


def before_topic_consolidation(text, source, stop_before_journal=None):
    source = Path(source)
    source = source.relative_to(ROOT).as_posix() if source.is_absolute() else source.as_posix()
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    if stop_before_journal and stop_before_journal not in JOURNALS:
        raise AssertionError(f'Unknown topic consolidation journal: {stop_before_journal}')
    for name in JOURNALS:
        if name == stop_before_journal:
            break
        path = ROOT / 'docs/audit' / name
        if not path.exists():
            continue
        journal = json.loads(path.read_text())
        rows = journal.get('files', journal.get('entries', journal.get('records', [])))
        row = next((row for row in rows if row['source'] == source), None)
        if row is None or row.get('created') or digest(text) == row['before_sha256']:
            continue
        if digest(text) != row['after_sha256']:
            raise AssertionError(f'Unreviewed topic consolidation revision: {source}')
        for change in reversed(row['changes']):
            if not change['new'] or text.count(change['new']) != 1:
                raise AssertionError(f'Ambiguous topic consolidation change: {source}')
            text = text.replace(change['new'], change['old'], 1)
        if digest(text) != row['before_sha256']:
            raise AssertionError(f'Topic consolidation does not restore its preceding revision: {source}')
    return text
