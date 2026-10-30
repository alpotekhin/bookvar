import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const ledger = JSON.parse(readFileSync(resolve(root, 'docs/audit/2026-10-29-english-reconciliation.json'), 'utf8'));
const entries = ledger.entries as Array<{
  path: string; route: string; canonical: string; status: string;
  after_sha256: string; canonical_sha256: string; figures: string[];
  en_body_read: boolean; outstanding_expansions: string[];
}>;
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
function matter(text: string) {
  return parse(text.match(/^---\n([\s\S]*?)\n---/)![1]);
}

describe('existing English textbook reconciliation', () => {
  it('reorders Constitutional AI in both locales without losing teaching material', () => {
    const name = '2026-10-30-posttraining-order-additions.json';
    const journal = JSON.parse(read('docs/audit/' + name));
    expect(journal.files).toHaveLength(7);
    for (const row of journal.files) {
      const current = read(row.source);
      const prior = beforeTopicConsolidation(current, row.source, root,
        '2026-10-30-locale-navigation-additions.json');
      expect(hash(current), row.source).toBe(row.after_sha256);
      expect(hash(prior), row.source).toBe(row.before_sha256);
      const preserved = (text: string) => ({
        code: text.match(/^```[^\n]*\n[\s\S]*?^```\s*$/gm) ?? [],
        math: text.match(/^\$\$\n[\s\S]*?^\$\$$/gm) ?? [],
        figures: text.match(/!\[\[[^\]]+\]\]/g) ?? [],
      });
      expect(preserved(current), row.source).toEqual(preserved(prior));
      expect(() => beforeTopicConsolidation(current + '\nunknown edit\n',
        row.source, root, '2026-10-30-locale-navigation-additions.json')).toThrow();
    }
    expect(matter(read('en/00 Textbook/_index.md')).canonical_revision)
      .toBe(hash(read('00 Учебник/_index.md')));
    const cai = read('00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md');
    expect(cai).not.toContain('can be read before SFT');
    expect(cai).toContain('#bradley-terry|Reward modeling');
  });

  it('adds only two compatibility anchors to the English contents without changing prose', () => {
    const name = '2026-10-30-locale-navigation-additions.json';
    const journal = JSON.parse(read('docs/audit/' + name));
    expect(journal.files).toHaveLength(1);
    const row = journal.files[0];
    expect(row.source).toBe('en/00 Textbook/_index.md');
    const current = beforeTopicConsolidation(read(row.source), row.source, root, name);
    const prior = beforeTopicConsolidation(read(row.source), row.source, root,
      '2026-10-30-reference-index-metadata-additions.json');
    expect(hash(current)).toBe(row.after_sha256);
    expect(hash(prior)).toBe(row.before_sha256);
    const anchors = ['углублённые-модули', 'контрастивное-и-маскированное-предобучение'];
    let body = current.split('\n---\n')[1];
    for (const anchor of anchors) body = body.replace('<a id="' + anchor + '"></a>\n\n', '');
    expect(body).toBe(prior.split('\n---\n')[1]);
    expect(matter(current).translation_status).toBe(matter(prior).translation_status);
    const canonicalAtReview = beforeTopicConsolidation(read('00 Учебник/_index.md'),
      '00 Учебник/_index.md', root, name);
    expect(matter(current).canonical_revision).toBe(hash(canonicalAtReview));
    const oldCanonical = beforeTopicConsolidation(read('00 Учебник/_index.md'),
      '00 Учебник/_index.md', root, '2026-10-29-scheduler-lab-additions.json');
    expect(hash(oldCanonical)).toBe(matter(prior).canonical_revision);
    expect(matter(current).translation_reviewed).toBe('2026-10-30');
    expect(matter(prior).translation_reviewed).toBe('2026-10-29');
    expect(() => beforeTopicConsolidation(current + '\nunknown edit\n', row.source, root,
      '2026-10-30-reference-index-metadata-additions.json')).toThrow();
  });
  it('changes only layout in the twenty final mobile formula reflows', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-en-parity-render-reflow.json'));
    const normalize = (s: string) => s
      .replace(/\\(?:begin|end)\{(?:aligned|gathered)\}/g, '')
      .replace(/\\(?:qquad|quad|left|right|Bigg)/g, '')
      .replace(/\\\\/g, '').replace(/\{\}|&|\s/g, '').replace(/[.,]$/, '');
    let count = 0;
    for (const row of journal.files) {
      const current = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-small-data-finetuning-additions.json');
      let restored = current;
      for (const change of row.changes) {
        expect(normalize(change.new), row.source).toBe(normalize(change.old));
        expect(restored, row.source).toContain(change.new);
        restored = restored.replace(change.new, () => change.old);
        count++;
      }
      expect(hash(current), row.source).toBe(row.after_sha256);
      expect(hash(restored), row.source).toBe(row.before_sha256);
      expect(restored.match(/```[\s\S]*?```/g)).toEqual(current.match(/```[\s\S]*?```/g));
    }
    expect(count).toBe(20);
  });
  it('retains historical math-reflow evidence and the current self-attention reflows', () => {
    const reflow = JSON.parse(read('docs/audit/2026-10-29-english-math-reflow.json'));
    const originalHashes = [
      '4223813e97a1d172c2468a7105ef6e07b8b23d3d13caf9bc53140941ca5c5402',
      '1392955ececc792893b7484ff7742646efa144edff0930c0c64c5a149fd611b5',
    ];
    const normalize = (s: string) => s.replace(/\\(?:begin|end)\{(?:aligned|gathered)\}/g, '')
      .replace(/\\(?:qquad|quad)/g, '').replace(/\\\\/g, '').replace(/\{\}|&|\s/g, '');
    let count = 0;
    for (const [i, row] of reflow.files.entries()) {
      let restored = beforeTopicConsolidation(read(row.source), row.source, root);
      for (const change of row.changes) {
        expect(normalize(change.new)).toBe(normalize(change.old));
        if (i === 0) {
          expect(restored).toContain(change.new);
          restored = restored.replace(change.new, () => change.old);
        }
        count++;
      }
      // Self-attention did not change after this reflow. FlashAttention later
      // received a full content expansion, recorded separately; restoring its
      // displays now include more intermediate derivation steps. The journal
      // remains evidence for the earlier revision, not a constraint to remove
      // those steps from the expanded chapter.
      if (i === 0) expect(hash(restored)).toBe(originalHashes[i]);
      expect(restored.match(/```[\s\S]*?```/g)).toEqual(read(row.source).match(/```[\s\S]*?```/g));
    }
    expect(count).toBe(8);
  });
  it('accounts for all existing EN textbook bodies without calling fallbacks translations', () => {
    const pages = loadManifest(resolve(root, 'publishing/navigation.yml')).sections
      .find(section => section.id === 'textbook')!.pages.filter(page => page.sourceEn);
    expect(entries).toHaveLength(29);
    expect(entries.map(row => row.path).sort()).toEqual(pages.map(page => page.sourceEn).sort());
    expect(entries.filter(row => row.status === 'current')).toHaveLength(29);
    expect(entries.filter(row => row.status === 'partial')).toHaveLength(0);
    for (const row of entries) {
      const page = pages.find(page => page.sourceEn === row.path)!;
      expect(row.canonical).toBe(page.source);
      expect(row.route).toBe(page.route);
      expect(row.en_body_read).toBe(true);
      expect(row.outstanding_expansions.length === 0).toBe(row.status === 'current');
    }
  });

  it('pins an actual canonical revision and verifies the exact reviewed EN files', () => {
    for (const row of entries) {
      const text = read(row.path), frontmatter = matter(text);
      expect(frontmatter.locale).toBe('en');
      expect(frontmatter.translation_status).toBe(row.status);
      expect(frontmatter.translation_of).toBe(row.canonical);
      expect(frontmatter.canonical_revision).toMatch(/^[a-f0-9]{64}$/);
      // This receipt predates PEFT navigation and the source-native BERT additions.
      // Restore their exact journaled deltas; unrecorded prose still fails.
      const reviewed = (value: string, source: string) => beforeTopicConsolidation(
        value, source, root, '2026-10-29-scheduler-lab-additions.json');
      expect(matter(reviewed(text, row.path)).translation_reviewed).toBe('2026-10-29');
      expect(matter(reviewed(text, row.path)).canonical_revision).toBe(row.canonical_sha256);
      expect(hash(reviewed(text, row.path)), row.path).toBe(row.after_sha256);
      if (row.status === 'current')
        expect(hash(reviewed(read(row.canonical), row.canonical))).toBe(row.canonical_sha256);
    }
  });

  it('retains locally resolvable source figures in every existing EN chapter', () => {
    for (const row of entries) {
      if (row.route !== 'textbook/index') expect(row.figures.length, row.path).toBeGreaterThan(0);
      for (const figure of row.figures) expect(existsSync(resolve(root, figure)), figure).toBe(true);
    }
  });

  it('does not restore the impossible self-attention projections', () => {
    const text = read(entries.find(row => row.route === 'textbook/transformer/self-attention')!.path);
    expect(text).toContain('Set $X=I_3$');
    expect(text).toContain('x_3=x_1+x_2');
    expect(text).toContain('v_3=v_1+v_2');
    expect(text).toContain('X = torch.eye(3, dtype=dtype, requires_grad=True)');
    expect(text).toContain('X.grad, dQ @ WQ.T + dK @ WK.T + dV @ WV.T');
    expect(text).toContain('Different weights, the same output');
    expect(text.match(/!\[\[/g)).toHaveLength(5);
  });

  it('retains every canonical figure and valid display-math delimiters', () => {
    const figures = (text: string) => [...text.matchAll(/!\[\[([^\]|]+)/g)]
      .map(match => match[1].replace(/^02 Areas\/ML & DL\//, ''));
    for (const row of entries) {
      const english = read(row.path), canonical = read(row.canonical);
      const present = new Set(figures(english));
      for (const figure of figures(canonical)) expect(present.has(figure), row.path + ': ' + figure).toBe(true);
      const prose = english.replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, '');
      expect(prose.match(/^\$$/gm) ?? [], row.path).toHaveLength(0);
      expect((prose.match(/^\$\$$/gm) ?? []).length % 2, row.path).toBe(0);
    }
  });

  it('preserves source-native licensed explanations instead of replacing them with summaries', () => {
    const row = entries.find(row => row.route === 'textbook/inference/serving-engines')!;
    const original = read(row.canonical), english = read(row.path);
    const normalize = (text: string) => text
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/^#+\s+/gm, '')
      .replace(/\s+/g, ' ').trim();
    for (const id of ['vllm-processes', 'vllm-worker-model', 'vllm-weight-loading',
      'tensorrt-architecture', 'flashinfer-ragged', 'flashinfer-pages', 'flashinfer-faq']) {
      const start = '<!-- source-excerpt:' + id + ':start -->';
      const end = '<!-- source-excerpt:' + id + ':end -->';
      const body = original.split(start)[1]?.split(end)[0];
      expect(body, id).toBeDefined();
      expect(normalize(english), id).toContain(normalize(body!));
    }
    const flash = entries.find(row => row.route === 'textbook/inference/flashattention')!;
    const marker = /<!-- source-excerpt: fa3-pipeline-practicalities:start -->([\s\S]*?)<!-- source-excerpt: fa3-pipeline-practicalities:end -->/;
    expect(read(flash.path).match(marker)![1]).toBe(read(flash.canonical).match(marker)![1]);
  });
});

