import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const chapter = '00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts.md';
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');
const body = read(chapter);

describe('small-data encoder fine-tuning inside the canonical explanation', () => {
  it('adds the mechanisms to the existing native-English chapter and both topic indexes', () => {
    expect(body).toContain('source_language: en');
    expect(body.match(/^```python$/gm)).toHaveLength(18);
    for (const anchor of ['small-data-fine-tuning', 'fine-tuning-bias-correction',
      'fine-tuning-reinitialization', 'fine-tuning-update-budget', 'intermediate-task-transfer']) {
      expect(body).toContain('id="' + anchor + '"');
      for (const p of ['01 Справочник/_index.md', 'en/01 Reference/_index.md'])
        expect(read(p)).toContain('#' + anchor + '|');
    }
  });
  it('preserves the complete original licensed repository rasters and their archive bytes', () => {
    const assets = JSON.parse(read('00 Учебник/Assets/Figures/curated/small-data-finetuning-2026-10-29/assets.json'));
    expect(assets).toHaveLength(2);
    for (const a of assets) {
      expect(body).toContain(a.asset);
      expect(hash(readFileSync(resolve(root, a.asset)))).toBe(a.sha256);
      expect(hash(readFileSync(resolve(root, a.source_asset)))).toBe(a.sha256);
      expect(hash(readFileSync(resolve(root, a.license_evidence)))).toBe(a.license_evidence_sha256);
      expect(a.derivation).toBe('source-raster-copy');
    }
    for (const [published, original] of [
      ['asapp-finetuning-0aa4f4e-LICENSE.txt', 'asapp-LICENSE'],
      ['mosbach-finetuning-3cf27e8-Apache-2.0.txt', 'mosbach-LICENSE'],
    ]) expect(read('site/public/licenses/' + published))
      .toBe(read('05 Источники/Originals/small-data-finetuning-2026-10-29/' + original));
  });
  it('keeps the original ASAPP Overview excerpt without translating or rewriting it', () => {
    const normalize = (s: string) => s.replace(/^> ?/gm, '').replace(/\s+/g, ' ').trim();
    const quote = body.match(/(?:^>[^\n]*(?:\n|$))+/gm)!
      .find(s => s.startsWith('> In this paper, we study the problem of few-sample'));
    expect(quote).toBeDefined();
    expect(normalize(read('05 Источники/Originals/small-data-finetuning-2026-10-29/asapp-README.md')))
      .toContain(normalize(quote!));
  });
  it('retains the worked reasoning and distinguishes reported trials from the synthetic checks', () => {
    const prose = body.replace(/\s+/g, ' ');
    for (const fact of ['0.6931', '3.162', '6.528', '96 updates', '225 in three epochs',
      '50% failure rate', 'Table 2', 'test halves', 'not results of training BERT',
      'resetting only the classifier', 'Reinitialization happens before constructing'])
      expect(prose.toLowerCase().includes(fact.toLowerCase()), fact).toBe(true);
    expect(body).toContain('chosen = [] if count == 0 else list(blocks)[-count:]');
    expect(body).toContain('MNLI uses three classes, while RTE');
    expect(body).not.toContain('```mermaid');
  });
  it('restores the exact preceding revision and refuses unreviewed additions', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-small-data-finetuning-additions.json'));
    expect(journal.files).toHaveLength(4);
    for (const row of journal.files) {
      let text = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-small-data-finetuning-additions.json');
      expect(hash(text)).toBe(row.after_sha256);
      for (const change of [...row.changes].reverse()) {
        expect(text.split(change.new)).toHaveLength(2);
        text = text.replace(change.new, () => change.old);
      }
      expect(hash(text)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-peft-additions.json')).toBe(text);
      expect(() => beforeTopicConsolidation(read(row.source) + '\nUnreviewed prose.\n',
        row.source, root, '2026-10-29-peft-additions.json')).toThrow();
    }
  });
});
