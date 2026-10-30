import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-29-en-transformer-models-parity.json'));
const fences = (text: string) => [...text.matchAll(/^(\x60{3}|~{3})([^\n]*)\n([\s\S]*?)^\1/gm)]
  .map(match => ({ language: match[2], text: match[3] }));
const prose = (text: string) => text.replace(/^(\x60{3}|~{3})[^\n]*\n[\s\S]*?^\1\s*$/gm, '');
const figures = (text: string) => [...text.matchAll(/!\[\[([^\]]+)\]\]/g)]
  .map(match => match[1].replace(/^02 Areas\/ML & DL\//, ''));
const equations = (text: string) => [...text.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)]
  .map(match => match[1].replaceAll('\\text{или}', '\\text{or}').replaceAll('\\text{и}', '\\text{and}'));
const urls = (text: string) => new Set([...text.matchAll(/https?:\/\/[^\s<>)\]]+/g)].map(match => match[0]));

describe('English Transformer and model-family full content parity', () => {
  it('accounts for all nine assigned chapters, not canonical fallbacks', () => {
    expect(journal.entries).toHaveLength(9);
    expect(journal.entries.every((entry: any) => entry.complete && entry.status === 'current')).toBe(true);
    expect(journal.totals.before_figures).toBe(35);
    expect(journal.totals.after_figures).toBe(53);
    expect(journal.totals.display_equations).toBe(79);
    expect(journal.totals.standalone_python_blocks).toBe(33);
  });

  for (const entry of journal.entries) {
    describe(entry.route, () => {
      // This receipt predates the separately reviewed source-native objectives.
      // Preserve its exact evidence; current additions have their own contracts.
      const historical = (path: string) => beforeTopicConsolidation(read(path), path,
        root, '2026-10-29-small-data-finetuning-additions.json');
      const english = historical(entry.path);
      const canonical = historical(entry.canonical);

      it('pins the exact fully read English and canonical revisions', () => {
        expect(sha(english)).toBe(entry.after_sha256);
        expect(sha(canonical)).toBe(entry.canonical_sha256);
        const frontmatter = parse(english.match(/^---\n([\s\S]*?)\n---/)![1]);
        expect(frontmatter.locale).toBe('en');
        expect(frontmatter.translation_status).toBe('current');
        expect(frontmatter.translation_of).toBe(entry.canonical);
        expect(frontmatter.canonical_revision).toBe(entry.canonical_sha256);
        expect(frontmatter.translation_reviewed).toBe('2026-10-29');
        expect(entry.full_en_before_read && entry.full_canonical_read && entry.full_en_after_read).toBe(true);
        expect(entry.outstanding_substantive_omissions).toEqual([]);
        expect(entry.sections.length).toBeGreaterThan(10);
      });

      it('preserves the ordered source figures and exact asset bytes', () => {
        expect(figures(english)).toEqual(figures(canonical));
        expect(figures(english)).toEqual(entry.figures);
        expect(entry.figure_inventory).toHaveLength(entry.figure_count);
        for (const figure of entry.figure_inventory) {
          expect(sha(readFileSync(resolve(root, figure.path)))).toBe(figure.sha256);
        }
      });

      it('preserves every display equation with documented language/reflow changes only', () => {
        let restored = english;
        if (entry.math_reflow) {
          expect(restored).toContain(entry.math_reflow.new);
          restored = restored.replace(entry.math_reflow.new, () => entry.math_reflow.old);
        }
        expect(equations(restored)).toEqual(equations(canonical));
        expect(equations(english)).toHaveLength(entry.display_math_count);
        expect(prose(english)).not.toMatch(/^\s*\$\s*$/m);
        expect((prose(english).match(/^\$\$$/gm) ?? []).length).toBe(entry.display_math_count * 2);
      });

      it('pins every displayed code/text block and distinguishes execution from source excerpts', () => {
        const current = fences(english), original = fences(canonical);
        expect(current).toHaveLength(entry.code_block_count);
        expect(current.map(block => block.language)).toEqual(original.map(block => block.language));
        for (const [index, block] of current.entries()) {
          const record = entry.code_inventory[index];
          expect(sha(block.text)).toBe(record.sha256);
          expect(sha(original[index].text)).toBe(record.canonical_sha256);
          if (block.language.startsWith('python title=')) {
            expect(block.text).toBe(original[index].text);
            expect(record.execution).toBe('not-run-context-dependent-original-source');
          }
          if (block.language === 'python') expect(record.execution).toBe('cpu-passed');
        }
        expect(entry.canonical_python_ast_equal).toBe(true);
        expect(entry.execution_evidence.returncode).toBe(0);
        expect(entry.execution_evidence.artifact_sha256).toBe(entry.after_sha256);
        expect(entry.execution_evidence.executed_standalone_python_blocks).toHaveLength(entry.python_blocks);
      });

      it('retains primary links, explicit stable fragment aliases, and verbatim licensed quotations', () => {
        const present = urls(english);
        for (const url of urls(canonical)) expect(present.has(url), url).toBe(true);
        for (const id of entry.canonical_explicit_anchors) expect(english).toContain('<span id="' + id + '"></span>');
        for (const quote of entry.source_native_quotations) {
          expect(canonical).toContain(quote.text);
          expect(english).toContain(quote.text);
          expect(quote.preserved_verbatim).toBe(true);
        }
      });
    });
  }

  it('does not call CPU examples source-runtime reproductions or visual acceptance', () => {
    expect(journal.entries.reduce((sum: number, e: any) => sum + e.python_blocks, 0)).toBe(33);
    expect(journal.entries.reduce((sum: number, e: any) => sum + e.not_executed.length, 0)).toBe(3);
    expect(journal.verification.source_runtime_or_accelerator_benchmarks).toContain('not run');
    expect(journal.verification.build_and_visual_qa).toContain('parent-owned');
  });

  it('keeps the specific distinctions recovered by full chapter reconciliation', () => {
    const chapter = (route: string) => read(journal.entries.find((e: any) => e.route.endsWith(route)).path);
    expect(chapter('masking-heads-shapes')).toContain('In a full-sequence self-attention pass');
    expect(chapter('t5')).toContain('Changing the source also requires clearing decoder self-KV');
    expect(chapter('t5')).toContain('there may be no\nseparate closing sentinel afterward');
    expect(chapter('t5')).toContain('222,903,552');
    expect(chapter('gpt-2')).toContain('P+G-1');
    expect(chapter('bert-roberta-deberta')).toContain('89.875');
  });
});
