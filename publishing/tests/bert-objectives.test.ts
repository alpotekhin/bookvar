import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const ru = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md';
const en = 'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03 BERT, RoBERTa, and DeBERTa.md';
const read = (p: string) => readFileSync(resolve(root,p),'utf8');
const hash = (s: string) => createHash('sha256').update(s).digest('hex');
const anchors = ['whole-word-masking','spanbert-boundary-objective',
  'spanbert-span-sampling','sentence-order-prediction'];
const normalize = (s: string) => s.replace(/\s+/g,' ').trim();

describe('source-faithful grouped masking and sentence-order additions', () => {
  it('keeps one unchanged English explanation in both existing chapters', () => {
    const fragment = (body: string) => body.slice(body.indexOf('<a id="whole-word-masking">'),
      body.indexOf('\n## DeBERTa:',body.indexOf('<a id="whole-word-masking">')));
    expect(fragment(read(ru))).toBe(fragment(read(en)));
    expect(fragment(read(ru)).length).toBeGreaterThan(15000);
    for (const p of [ru,en]) {
      expect(read(p).match(/^```python$/gm)).toHaveLength(8);
      for (const a of anchors) expect(read(p).split('id="'+a+'"')).toHaveLength(2);
      expect(read(p)).not.toContain('```mermaid');
    }
    expect(read(en)).toContain('canonical_revision: "'+hash(read(ru))+'"');
    for (const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
      for (const a of anchors) expect(read(p)).toContain('#'+a+'|');
  });
  it('preserves the original README passage and the ALBERT swap operators', () => {
    const body = read(ru);
    const quote = body.match(/> The training is identical[\s\S]*?multiple WordPieces\./)![0];
    expect(normalize(read('05 Источники/Originals/bert-objectives-2026-10-29/bert-README.md')))
      .toContain(normalize(quote.replace(/^> ?/gm,'')));
    const excerpt = body.match(/```python title="Original ALBERT pair swap — Apache-2.0"\n([\s\S]*?)\n```/)![1];
    expect(read('05 Источники/Originals/bert-objectives-2026-10-29/albert-create_pretraining_data.py'))
      .toContain(excerpt);
    for (const [published,archived] of [
      ['google-bert-eedf5716-Apache-2.0.txt','bert-LICENSE'],
      ['google-albert-b772393d-Apache-2.0.txt','albert-LICENSE'],
    ]) expect(read('site/public/licenses/'+published))
      .toBe(read('05 Источники/Originals/bert-objectives-2026-10-29/'+archived));
  });
  it('keeps the causal distinctions, source versions and experimental boundaries', () => {
    const body = normalize(read(ru));
    for (const fact of ['external boundary states', 'does not read',
      'direct inputs to the SBO head', 'final selected word can be cut short',
      '80/10/10', '1.2M-step ablation', 'final 2.4M-step test results',
      'random-document fallback', 'next_sentence_labels', 'Table 5',
      '0.28768', '52.0', '86.5', 'same vocabulary-sized prediction head'])
      expect(body, fact).toContain(fact);
  });
  it('reverses only the five exact reviewed deltas and rejects unreviewed text', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-bert-objectives-additions.json'));
    expect(journal.files).toHaveLength(5);
    for (const row of journal.files) {
      const current = beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-bert-objectives-additions.json');
      expect(hash(current)).toBe(row.after_sha256);
      const restored = beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-small-data-finetuning-additions.json');
      expect(hash(restored)).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(read(row.source)+'\nUnreviewed sentence.\n',
        row.source,root,'2026-10-29-small-data-finetuning-additions.json')).toThrow();
    }
  });
});
