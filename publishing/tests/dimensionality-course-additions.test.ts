import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p: string) => readFileSync(resolve(root,p),'utf8');
const sha = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-11-03-dimensionality-course-additions.json'));
const row = journal.files[0], current = read(row.source);
const stop = '2026-11-03-spectral-source-additions.json';
const prior = beforeTopicConsolidation(current,row.source,root,stop);

describe('dimensionality chapter original-course additions', () => {
  it('reverses the precise chapter, registry and index changes and rejects unrelated edits', () => {
    expect(row.before_sha256).toBe('0487fe879457379241a0fa9565ec8cac5eb51280ae820e50a02dc3896a929e5d');
    for (const entry of journal.files) {
      const text = beforeTopicConsolidation(read(entry.source),entry.source,root,
        '2026-11-03-dimensionality-course-additions.json');
      expect(sha(text)).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(() => beforeTopicConsolidation(text+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });

  it('retains seven earlier programs and nine image assets, with all mechanisms intact', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(fences(current)).toHaveLength(9);
    expect(fences(prior)).toHaveLength(7);
    expect(fences(current).filter(x=>fences(prior).includes(x))).toEqual(fences(prior));
    expect(images(current)).toHaveLength(24);
    expect(images(current).filter(x=>images(prior).includes(x))).toEqual(images(prior));
    for (const text of ['fixed_projector','noise_tradeoff','digit_denoiser','cumulative_variance',
      'Gaussian','16,112','7,269','не является энтропией','оценка градиента смещена',
      '0.5.12 при обычном','2.5.4','69278069a6d4ef16513501b8839f4eb836c7b231'])
      expect(current).toContain(text);
    expect(current).not.toContain('smooth_knn_dist.py_func');
  });

  it('retains exact source-language paragraphs and all fifteen complete original images', () => {
    const source = read('05 Источники/Courses/Advanced Unsupervised and Classical ML/umap-parameters.rst.md');
    const plain = (s: string) => s.replace(/^> /gm,'')
      .split(String.fromCharCode(96)).join('').replace(/\s+/g,' ').trim();
    for (const [start,end] of [
      ['This parameter controls how UMAP balances','getting the broader of the data.'],
      ['Here we see that with','detailed topological structure.'],
    ]) {
      const first = current.indexOf('> '+start);
      const last = current.indexOf(end,first)+end.length;
      expect(first).toBeGreaterThan(0);
      expect(plain(source)).toContain(plain(current.slice(first,last)));
    }
    const rows = JSON.parse(read('00 Учебник/Assets/Figures/curated/dimensionality-2026-11-03/assets.json'));
    expect(rows).toHaveLength(15);
    for (const entry of rows)
      expect(sha(readFileSync(resolve(root,entry.asset)))).toBe(entry.sha256);
    expect(journal.sourceComparison.fullyRead).toHaveLength(3);
  });

  it('keeps eight terms owned by this chapter in both reference locales', () => {
    const stem = row.source.replace(/\.md$/,'');
    const anchors = ['pca-image-basis','pca-cumulative-variance','pca-noise-filtering','pca-whitening',
      'tsne-perplexity','tsne-map-interpretation','umap-neighbors-min-dist','trustworthiness'];
    for (const anchor of anchors) {
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for (const source of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(source)).toContain('[['+stem+'#'+anchor+'|');
    }
  });
});

