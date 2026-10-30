import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import matter from 'gray-matter';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { parseMarkdownHeadings, normalizeObsidianHeading } from '../adapter/build.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
const pages = manifest.sections.flatMap(section => section.pages);
const aliases = JSON.parse(read('docs/audit/2026-10-29-topic-aliases.json')).entries;
const targets = (text: string) => new Set([...text.matchAll(/(?<!!)\[\[([^\]|#]+)/g)]
  .map(match => match[1].replace(/^02 Areas\/ML & DL\//, '').replace(/\.md$/, '')));

describe('one primary explanation per published topic', () => {
  it('covers every textbook subject in both topic indexes without writing second explanations', () => {
    const book = pages.filter(page => page.route.startsWith('textbook/')
      && !['textbook/index', 'textbook/how-to-use'].includes(page.route));
    expect(book).toHaveLength(152);
    for (const path of ['01 Справочник/_index.md', 'en/01 Reference/_index.md']) {
      const content = read(path), links = targets(content);
      for (const page of book) expect(links.has(page.source.replace(/\.md$/, '')), page.source).toBe(true);
      expect(content).not.toMatch(/^\$\$|^```|!\[\[/m);
    }
  });

  it('preserves each original article and route while removing independently maintained copies', () => {
    expect(aliases).toHaveLength(49);
    for (const row of aliases) {
      const original = read(row.snapshot);
      expect(hash(original), row.source).toBe(row.beforeSha256);
      const current = read(row.source), parsed = matter(current);
      expect(hash(current), row.source).toBe(row.afterSha256);
      expect(parsed.data.status).toBe('redirect');
      expect(parsed.data.canonical_topic).toBe(row.canonicalSource);
      expect(parsed.content).not.toMatch(/^##|^\$\$|^```|!\[\[/m);
      expect(pages.find(page => page.source === row.source)?.route).toBe(row.route);
      expect(pages.some(page => page.source === row.canonicalSource)).toBe(true);
      const primary = matter(read(row.canonicalSource));
      expect(primary.data.status).not.toBe('redirect');
      if (row.canonicalHeading) {
        expect(parseMarkdownHeadings(primary.content, true).some(heading =>
          heading.explicitAnchor === row.canonicalHeading
          || heading.normalized === normalizeObsidianHeading(row.canonicalHeading))).toBe(true);
      }
      for (const id of row.retainedFragmentAnchors)
        expect(parsed.content, row.source + '#' + id).toContain('id="' + id + '"');
    }
  });

  it('keeps original implementation studies in the library without altering their text, code or images', () => {
    const moves = JSON.parse(read('docs/audit/2026-10-29-implementation-studies-move.json')).entries;
    for (const move of moves) {
      const alias = aliases.find((row: any) => row.source === move.source);
      expect(alias).toBeDefined();
      const original = matter(read(alias.snapshot));
      const study = matter(read(move.canonicalSource));
      expect(study.data.type).toBe('source-note');
      expect(study.content.trimEnd()).toBe(original.content.trimEnd());
      expect(pages.find(page => page.source === move.canonicalSource)?.route).toMatch(/^sources\/implementations\//);
    }
  });

  it('reverses only recorded additions and rejects an unrelated edit in a reviewed changed chapter', () => {
    for (const name of ['attention', 'training', 'inference', 'security']) {
      const path = 'docs/audit/2026-10-29-' + name + '-consolidation-additions.json';
      const journal = JSON.parse(read(path));
      for (const row of journal.files ?? journal.records ?? journal.entries) {
        if (row.created) continue;
        const current = read(row.source);
        expect(() => beforeTopicConsolidation(current + '\nUnreviewed change.\n', row.source, root)).toThrow();
      }
    }
  });

  it('keeps Python preservation checks strict after the same reviewed additions', () => {
    const result = execFileSync('python3', ['-c', `
import json, sys
from pathlib import Path
sys.path.insert(0, 'publishing/tools')
from topic_consolidation_baseline import before_topic_consolidation
root = Path.cwd()
journal = json.loads((root / 'docs/audit/2026-10-29-security-consolidation-additions.json').read_text())
for row in journal['entries']:
    current = (root / row['source']).read_text()
    assert before_topic_consolidation(current, row['source']) != current
    try:
        before_topic_consolidation(current + '\\nUnreviewed change.\\n', row['source'])
    except AssertionError:
        pass
    else:
        raise AssertionError('Accepted an unrecorded change')
print('Reviewed additions restored; unrelated edits rejected.')
`], { cwd: root, encoding: 'utf8' });
    expect(result).toContain('unrelated edits rejected');
  });

  it('retains the transferred source figures and the full new explanations', () => {
    const figure = 'transformer_self-attention_visualization.png';
    for (const path of [
      '00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md',
      'en/00 Textbook/05 Attention and Transformer/02 Self-Attention — Q, K, V.md'
    ]) expect(read(path)).toContain(figure);
    for (const source of [
      '00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md',
      '00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем.md',
      '00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем.md',
    ]) expect(existsSync(resolve(root, source))).toBe(true);
    expect(read('00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md')).toContain('feature_teacher.detach()');
    const evaluation = read('00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md');
    for (const section of ['mmlu-benchmark', 'humaneval-benchmark', 'helm-taxonomy-figure2.png'])
      expect(evaluation).toContain(section);
  });

  it('keeps display-math delimiters intact across the entire canonical textbook', () => {
    for (const page of pages.filter(page => page.route.startsWith('textbook/'))) {
      const body = matter(read(page.source)).content
        .replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1\s*$/gm, '');
      expect(body.match(/^\$$/gm) ?? [], page.source).toEqual([]);
      expect((body.match(/^\$\$$/gm) ?? []).length % 2, page.source).toBe(0);
    }
  });
});
