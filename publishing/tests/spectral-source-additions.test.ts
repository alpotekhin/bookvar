import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-11-03-spectral-source-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-10-31-clustering-course-additions.json';
const prior = beforeTopicConsolidation(current, row.source, root, stop);

describe('spectral chapter source explanations and figures', () => {
  it('restores the exact preceding chapter and registry and rejects unrelated changes', () => {
    expect(row.before_sha256).toBe('6184d698155c4e8571f256ffbc4f013d91cf1e5451d67db9109ccb5643d27b34');
    expect(sha(current)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root, stop)).toThrow();
    const registry = journal.files[1];
    expect(sha(beforeTopicConsolidation(read(registry.source), registry.source, root, stop)))
      .toBe(registry.before_sha256);
  });

  it('retains all four preceding programs and figures alongside the full new worked examples', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(fences(current)).toHaveLength(6);
    expect(fences(prior)).toHaveLength(4);
    expect(fences(current).filter(code => fences(prior).includes(code))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(6);
    expect(images(current).filter(path => images(prior).includes(path))).toEqual(images(prior));
    for (const text of ['chain_order', 'centered_heat', '0{,}999994', 'L_pinv',
      'commute_embedding', 'H_{23}=21', '1/\\sqrt{\\lambda_r}',
      'Геометрия взвешенной нормы', 'бесконечно', 'Ng–Jordan–Weiss'])
      expect(current).toContain(text);
  });

  it('indexes four canonical mechanisms in both locales without separate explanations', () => {
    const stem = row.source.replace(/\.md$/, '');
    for (const source of ['01 Справочник/_index.md', 'en/01 Reference/_index.md']) {
      const index = beforeTopicConsolidation(read(source), source, root,
        '2026-11-03-spectral-source-additions.json');
      for (const anchor of ['normalized-cut', 'degree-weighted-constraint', 'commute-time', 'njw-row-normalization']) {
        expect(current).toContain('<a id="' + anchor + '"></a>');
        expect(index).toContain('[[' + stem + '#' + anchor + '|');
      }
      const record = journal.files.find((entry: { source: string }) => entry.source === source);
      expect(sha(index)).toBe(record.after_sha256);
      expect(sha(beforeTopicConsolidation(read(source), source, root, stop))).toBe(record.before_sha256);
    }
  });

  it('compares the original source-language passage with the archived PDF', { tags: ['source-archive'] }, () => {
    const pdf = resolve(root, '05 Источники/Originals/spectral-2026-10-25/wiskott-laplacian.pdf');
    const source = execFileSync('pdftotext', ['-f','5','-l','5',pdf,'-'], {encoding:'utf8'});
    const start = current.indexOf('> This is essentially how');
    const end = current.indexOf('one heat diffusion mode into account.', start)
      + 'one heat diffusion mode into account.'.length;
    const normalize = (s: string) => s.replace(/^> /gm, '').replace(/\s+/g, ' ').trim();
    expect(start).toBeGreaterThan(0);
    expect(normalize(source)).toContain(normalize(current.slice(start, end)));
  });

  it('retains the exact complete source crops and source review metadata', () => {
    for (const [name, digest] of [
      ['wiskott-connected-heat.png', 'a5060cb9bddaaea7d3f45bfa43e25e52e9180aff0137ea2e59f75af611d2659e'],
      ['wiskott-weighted-constraint.png', '4615778b7c706d94775f2c2f0fc5c08ebf8e5fd12fc132879f1602500d6c3416'],
    ]) expect(sha(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/spectral-2026-10-31/' + name)))).toBe(digest);
    expect(journal.sourceComparison.fullyRead).toHaveLength(3);
  });
});

