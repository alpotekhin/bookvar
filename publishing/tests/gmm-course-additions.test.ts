import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p:string) => readFileSync(resolve(root,p),'utf8');
const sha = (s:string|Buffer) => createHash('sha256').update(s).digest('hex');
const own = '2026-11-03-gmm-course-additions.json';
const stop = '2026-11-03-dimensionality-course-additions.json';
const journal = JSON.parse(read('docs/audit/'+own));
const row = journal.files[0];
const current = beforeTopicConsolidation(read(row.source),row.source,root,own);
const prior = beforeTopicConsolidation(current,row.source,root,stop);

describe('GMM chapter source additions',()=>{
  it('reverses only pinned chapter, registry and index additions',()=>{
    expect(row.before_sha256).toBe('c02a1bd7955b8920a1842f89d34827cd126a90400cd65e45b49e8ca933c20a3d');
    for(const entry of journal.files){
      const text=beforeTopicConsolidation(read(entry.source),entry.source,root,own);
      expect(sha(text)).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(read(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text+'\nUNREVIEWED\n',entry.source,root,stop)).toThrow();
    }
  });
  it('retains all eight programs, seven images and prior explanatory paragraphs',()=>{
    const fences=(s:string)=>[...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m=>m[1]);
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(fences(prior)).toHaveLength(8);
    expect(fences(current)).toHaveLength(10);
    expect(fences(current).filter(x=>fences(prior).includes(x))).toEqual(fences(prior));
    expect(images(prior)).toHaveLength(7);
    expect(images(current)).toHaveLength(12);
    expect(images(current).filter(x=>images(prior).includes(x))).toEqual(images(prior));
    for(const term of ['gaussian_logpdf','fit_em_1d','bound_value','regularized_score',
      'initialization_results','generated_digit_pixels','126\\,419','−8,058',
      'не задаёт невырожденную плотность','не повторяет перебор AIC'])
      expect(current).toContain(term);
  });
  it('compares the original English passages with the archived notebook', { tags: ['source-archive'] }, ()=>{
    const notebook=JSON.parse(read('05 Источники/Originals/gmm-2026-10-25/jake-gmm.ipynb'));
    const plain=(s:string)=>s.replace(/^> ?/gm,'').replace(/\s+/g,' ').trim();
    for(const cell of [33,39]){
      const passage=notebook.cells[cell].source.join('').replace(/^##[^\n]*\n\n/,'');
      expect(plain(current)).toContain(plain(passage));
    }
  });
  it('retains the exact saved image bytes and source review metadata',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/gmm-2026-11-03/assets.json'));
    expect(rows).toHaveLength(5);
    for(const asset of rows)expect(sha(readFileSync(resolve(root,asset.asset)))).toBe(asset.sha256);
    expect(journal.sourceComparison.fullyRead).toHaveLength(3);
  });
  it('indexes eight stable mechanisms from the same chapter in both locales',()=>{
    const anchors=['gmm-responsibilities','em-bound','gmm-covariance','gmm-covariance-collapse',
      'gmm-initialization','gmm-density-components','gmm-model-selection','gmm-pca-generation'];
    const stem=row.source.replace(/\.md$/,'');
    for(const anchor of anchors){
      expect(current).toContain('<a id="'+anchor+'"></a>');
      for(const source of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(source)).toContain('[['+stem+'#'+anchor+'|');
    }
  });
});
