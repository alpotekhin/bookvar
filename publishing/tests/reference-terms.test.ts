import { describe, expect, it } from 'vitest';
import { referenceTerms } from '../adapter/reference-terms.js';

const term = { label: 'Метод', label_en: 'Method', anchor: 'method' };
const headings = [{ explicitAnchor: 'method' }];

describe('chapter-owned subject-reference terms', () => {
  it('keeps old chapters valid without additional terms', () => {
    expect(referenceTerms('chapter.md', undefined, [])).toEqual([]);
  });
  it('returns a bilingual term linked to a unique explicit chapter anchor', () => {
    expect(referenceTerms('chapter.md', [term], headings)).toEqual([term]);
  });
  it.each([
    ['not a list', term, headings],
    ['absent anchor', [term], []],
    ['ambiguous anchor', [term], [...headings, ...headings]],
    ['missing English label', [{ ...term, label_en: '' }], headings],
    ['wiki syntax in a label', [{ ...term, label: 'a|b' }], headings],
    ['duplicate term', [term, term], headings],
    ['invalid anchor syntax', [{ ...term, anchor: '../elsewhere' }], headings],
  ])('rejects %s with the source path', (_label, input, available) => {
    expect(() => referenceTerms('chapter.md', input, available as typeof headings)).toThrow('chapter.md');
  });
});
