import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');
const journals = [
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
];

// Historical preservation contracts still apply to the exact revision they
// reviewed. Reverse only independently recorded additions, never normalize prose.
export function beforeTopicConsolidation(text, source, root = defaultRoot, stopBeforeJournal) {
  if (stopBeforeJournal && !journals.includes(stopBeforeJournal))
    throw new Error('Unknown topic consolidation journal: ' + stopBeforeJournal);
  for (const name of journals) {
    if (name === stopBeforeJournal) break;
    const path = resolve(root, 'docs/audit', name);
    if (!existsSync(path)) continue;
    const journal = JSON.parse(readFileSync(path, 'utf8'));
    const row = (journal.files ?? journal.entries ?? journal.records).find(row => row.source === source);
    if (!row || row.created || hash(text) === row.before_sha256) continue;
    if (hash(text) !== row.after_sha256)
      throw new Error('Unreviewed topic consolidation revision: ' + source);
    for (const change of [...row.changes].reverse()) {
      if (!change.new || text.split(change.new).length !== 2)
        throw new Error('Ambiguous topic consolidation change: ' + source);
      text = text.replace(change.new, () => change.old);
    }
    if (hash(text) !== row.before_sha256)
      throw new Error('Topic consolidation does not restore its preceding revision: ' + source);
  }
  return text;
}
