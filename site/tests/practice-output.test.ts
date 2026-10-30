import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('published causal-attention practice', () => {
  const html = readFileSync(resolve(import.meta.dirname, '../dist/practice/causal-self-attention/index.html'), 'utf8');

  it('publishes the interactive lab and the complete canonical exercise together', () => {
    expect(html).toContain('id="past-score"');
    expect(html).toContain('Сила связи «читает → Она»');
    expect(html).toContain('Полностью воспроизводимый пример');
    for (const token of ['scaled_dot_product_attention', 'assert_close', 'contiguous', 'float64']) {
      expect(html).toContain(token);
    }
    expect(html).toContain('Проверки реализации');
    expect(html).toContain('Annotated Transformer');
    expect(html.match(/<h1(?:\s|>)/g)).toHaveLength(1);
  });

  it('includes the lab and exercise in the page table of contents', () => {
    expect(html).toContain('href="#lab-title"');
    expect(html).toContain('href="#полностью-воспроизводимый-пример"');
  });
});

describe('published interactive scheduler', () => {
  const html = readFileSync(resolve(import.meta.dirname, '../dist/practice/inference-scheduler/index.html'), 'utf8');

  it('renders one page title, the live lab and the complete canonical worked exercise', () => {
    expect(html.match(/<h1(?:\s|>)/g)).toHaveLength(1);
    for (const id of ['scheduler-budget', 'scheduler-slots', 'scheduler-blocks', 'scheduler-chunked',
      'scheduler-step', 'scheduler-back', 'scheduler-run', 'scheduler-timeline', 'scheduler-memory'])
      expect(html).toContain('id="' + id + '"');
    expect(html).toContain('Orca, §3');
    expect(html).toContain('CC BY-NC-SA 4.0');
    expect(html).toContain('hf-continuous-batching.png');
    expect(html).toContain('не исполняет нейросеть');
    expect(html).not.toContain('katex-error');
  });

  it('keeps attribution, accessible controls and theory in no-JS HTML', () => {
    expect(html).toContain('<noscript>');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain('role="alert"');
    expect(html).toContain('id="scheduler-lab-title"');
    expect(html).toContain('href="#scheduler-lab-title"');
    expect(html).toContain('полный разбор примера');
  });
});
