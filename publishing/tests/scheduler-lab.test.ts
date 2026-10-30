import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const practice = '06 Практика/14b Планировщик запросов и KV-память.md';

describe('interactive scheduler is a published canonical exercise', () => {
  it('wires the exercise into the manifest and index without replacing the neural-engine exercise', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const pages = manifest.sections.find(section => section.id === 'practice')!.pages;
    expect(pages.find(page => page.source === practice)?.route).toBe('practice/inference-scheduler');
    expect(read('06 Практика/_index.md')).toContain(practice.replace(/\.md$/, ''));
    const content = read(practice);
    expect(content).toContain('14 Собрать минимальный inference engine');
    expect(content).toContain('55b Scheduling');
  });

  it('retains the worked trace, distinct memory/token budgets and limits of the equal-iteration model', () => {
    const content = read(practice);
    for (const phrase of ['семи шагов', '131\\,072', 'P+N-1', 'не исполняет нейросеть',
      'Физический номер блока', 'намеренно не вытесняет', 'CC BY-NC-SA 4.0']) expect(content).toContain(phrase);
    expect(content).not.toContain('```mermaid');
    expect(content).toContain('hf-continuous-batching.png');
    expect(content).toContain('https://huggingface.co/blog/continuous_batching');
  });

  it('links both scheduling editions to the interactive exercise with its actual limits', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-30-scheduler-practice-link-additions.json'));
    expect(journal.files).toHaveLength(2);
    for (const row of journal.files) {
      const current = read(row.source);
      expect(current.split(practice.replace(/\.md$/, ''))).toHaveLength(2);
      expect(current).toContain('Interactive scheduler and KV-memory lab');
      expect(current).toContain('does not run a neural network or predict GPU latency');
    }
    expect(read(journal.files[1].source)).toContain(`canonical_revision: "${hash(read(journal.files[0].source))}"`);
  });

  it('preserves every teaching paragraph outside the exact practice-link and metadata delta', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-30-scheduler-practice-link-additions.json'));
    for (const row of journal.files) {
      const current = read(row.source);
      const restored = beforeTopicConsolidation(current, row.source, root,
        '2026-10-30-decoding-prerequisites-additions.json');
      expect(hash(current)).toBe(row.after_sha256);
      expect(hash(restored)).toBe(row.before_sha256);
      for (const pattern of [/^```[\s\S]*?^```$/gm, /\$\$[\s\S]*?\$\$/g, /^!\[\[.*\]\]$/gm])
        expect(current.match(pattern)).toEqual(restored.match(pattern));
      expect(() => beforeTopicConsolidation(current + '\nUnreviewed change.\n', row.source, root,
        '2026-10-30-decoding-prerequisites-additions.json')).toThrow();
    }
  });

  it('reverses only the recorded publication additions, preserving historical navigation contracts', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-scheduler-lab-additions.json'));
    for (const row of journal.files) {
      let restored = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-scheduler-lab-additions.json');
      expect(hash(restored)).toBe(row.after_sha256);
      for (const change of [...row.changes].reverse()) {
        expect(restored.split(change.new)).toHaveLength(2);
        restored = restored.replace(change.new, () => change.old);
      }
      expect(hash(restored)).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(read(row.source) + '\nUnreviewed change.\n', row.source, root)).toThrow();
    }
  });

  it('reuses the original lab in an EN wrapper without a new translation or RU redirect', () => {
    const wrapper = read('site/src/pages/en/practice/inference-scheduler.astro');
    expect(wrapper).toContain("getEntry('docs', 'en/practice/inference-scheduler')");
    expect(wrapper).toContain("../../../components/InferenceSchedulerLab.astro");
    expect(wrapper).toContain('<div lang="ru">');
    expect(wrapper).toContain('<Content />');
    const config = read('site/astro.config.mjs');
    expect(config).not.toContain("'/en/practice/inference-scheduler':");
    expect(config).toContain("'/en/practice/causal-self-attention': causalSelfAttentionHref");
  });

  it('records the bounded adapter and wrapper change without accepting unknown edits', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-30-en-scheduler-navigation-additions.json'));
    for (const row of journal.files) {
      const atReview = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-30-en-scheduler-navigation-additions.json');
      expect(hash(atReview)).toBe(row.after_sha256);
      if (row.created) continue;
      expect(hash(beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-30-scheduler-practice-link-additions.json'))).toBe(row.before_sha256);
      expect(() => beforeTopicConsolidation(read(row.source) + '\nUnreviewed change.\n', row.source, root,
        '2026-10-30-scheduler-practice-link-additions.json')).toThrow();
    }
  });
});
