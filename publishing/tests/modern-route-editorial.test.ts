import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const ffn = '00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md';
const long = '00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md';
const distributed = '00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md';
const journalName = '2026-10-30-modern-route-editorial-additions.json';

function teachingMaterial(text: string) {
  return {
    code: text.match(/^(?:```|~~~)[^\n]*\n[\s\S]*?^(?:```|~~~)\s*$/gm) ?? [],
    math: text.match(/^\$\$\n[\s\S]*?^\$\$$/gm) ?? [],
    figures: text.match(/!\[\[[^\]]+\]\]/g) ?? [],
    sourceExcerpts: text.match(/<!-- BEGIN SOURCE:[\s\S]*?<!-- END SOURCE:[^>]+-->/g) ?? [],
  };
}

describe('modern route editorial corrections', () => {
  it('keeps FFN as the sole explanation rather than promising a second formula page', () => {
    const alias = parse(read('01 Справочник/FFN и MoE/Dense FFN и gated activations.md').split('---')[1]);
    expect(alias.status).toBe('redirect');
    expect(alias.canonical_topic.replace(/\s+/g, ' ')).toBe(ffn);
    expect(read(ffn)).not.toContain('Краткие формулы собраны отдельно');
    expect(read(ffn)).toContain('**Дальше:**');
  });

  it('introduces distributed training through its actual specialist prerequisites', () => {
    const text = read(distributed);
    expect(text).not.toContain('Предыдущая [[43 Scaling laws');
    expect(text).toContain('> [!note] Prerequisites');
    expect(text).toContain('05 Transformer с нуля — формы, параметры и стоимость|Transformer tensor shapes and resources');
    expect(text).toContain('04 Арифметика Transformer и MoE|Transformer and MoE arithmetic');
    expect(text).toContain('[[43 Scaling laws|Scaling laws]]');
  });

  it('retains all five long-context examples without the stale four-block claim', () => {
    const text = read(long);
    expect(text.match(/^~~~python$/gm)).toHaveLength(5);
    expect(text).not.toContain('Все четыре блока NumPy');
    expect(text).toContain('Примеры NumPy в главе можно');
  });

  it('uses the same canonical body in both interfaces, without creating translations', () => {
    const pages = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.flatMap(s => s.pages);
    for (const source of [ffn, long, distributed]) {
      expect(pages.find(p => p.source === source)?.sourceEn).toBeUndefined();
      expect(parse(read(source).split('---')[1]).last_updated).toBe('2026-10-30');
    }
  });

  it('reverses only the three recorded corrections and preserves all teaching material', () => {
    const journal = JSON.parse(read('docs/audit/' + journalName));
    expect(journal.files.map((r: { source: string }) => r.source)).toEqual([ffn, long, distributed]);
    expect(journal.files[0].before_sha256).toBe('50f2528d8a0cba6231692f81353c6817f2626aa54945aa51a0b6e14aeccbffc3');
    expect(journal.files[1].before_sha256).toBe('c15d27a0f6565299948cf7c3aa0005428743735058ff8d9998707bced97c1b85');
    for (const row of journal.files) {
      const current = read(row.source);
      const prior = beforeTopicConsolidation(current, row.source, root,
        '2026-10-30-posttraining-order-additions.json');
      expect(hash(current), row.source).toBe(row.after_sha256);
      expect(hash(prior), row.source).toBe(row.before_sha256);
      expect(teachingMaterial(current), row.source).toEqual(teachingMaterial(prior));
      expect(() => beforeTopicConsolidation(current + '\nunknown prose edit\n', row.source, root,
        '2026-10-30-posttraining-order-additions.json')).toThrow('Unreviewed topic consolidation revision');
    }
  });
});
