import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe, expect, it} from 'vitest';
import {beforeCs336Residuals} from '../tools/cs336-residuals-baseline.mjs';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p: string) => readFileSync(resolve(root, p), 'utf8');
const hash = (s: string) => createHash('sha256').update(s).digest('hex');
const ledger = JSON.parse(read('docs/audit/2026-10-29-cs336-residual-additions.json'));

describe('CS336 residual mechanism and original figures', () => {
  it('preserves every byte outside the five recorded additions and exact V4 summary edits', () => {
    for (const row of ledger.files) {
      const current = beforeTopicConsolidation(read(row.source), row.source, root);
      expect(hash(current)).toBe(row.after_sha256);
      expect(hash(beforeCs336Residuals(current, row.source, root))).toBe(row.before_sha256);
    }
  });

  it('does not hide an unreviewed change inside an addition', () => {
    const row = ledger.files[0];
    const historical = beforeTopicConsolidation(read(row.source), row.source, root);
    expect(historical).toContain('### Compressed Sparse Attention');
    const changed = historical.replace('### Compressed Sparse Attention', '### Changed Sparse Attention');
    expect(changed).not.toBe(historical);
    expect(() => beforeCs336Residuals(changed, row.source, root)).toThrow('Unreviewed CS336 residual');
  });

  it('separates 61 decoder layers from the MTP compression rate', () => {
    const config = JSON.parse(read('05 Источники/Architecture/DeepSeek V4 2026-10-29/config.json'));
    const rates = config.compress_ratios;
    expect(config.num_hidden_layers).toBe(61);
    expect(config.num_nextn_predict_layers).toBe(1);
    expect(rates).toHaveLength(62);
    expect(rates.slice(0, 61).filter((r: number) => r === 4)).toHaveLength(30);
    expect(rates.slice(0, 61).filter((r: number) => r === 128)).toHaveLength(31);
    expect(rates.slice(61)).toEqual([0]);
    const chapter = read(ledger.files[0].source);
    expect(chapter).toContain('plus one MTP layer');
    expect(chapter).toContain('not the total cache stored');
  });

  it('distinguishes historical examples, taxonomy coverage and model performance', () => {
    const evaluation = read(ledger.files[1].source);
    const responsible = read(ledger.files[2].source);
    expect(evaluation).toContain('не как пример успешно решённой задачи');
    expect(evaluation).toContain('64×64 с 16 цветами');
    expect(responsible).toContain('не схема алгоритма GCG');
    expect(responsible).toContain('23 из 45 категорий дают 51%');
    expect(responsible).toContain('не доля\nбезопасных ответов модели');
  });
});
