import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { classProbabilities, continuationScores, entailmentInput } from '../../site/src/lib/entailment-lab';
import { loadManifest } from '../adapter/manifest.js';

describe('GPT entailment lab', () => {
  it('keeps the test pair unchanged and omits its answer in every prompt', () => {
    const probe = 'Premise: A cat sleeps.\nHypothesis: An animal rests.\nRelation:';
    expect(entailmentInput('gpt1', 0)).toBe('[Start] A cat sleeps [Delim] An animal rests [Extract]');
    expect(entailmentInput('gpt2', 0)).toBe(probe);
    const [zero, one, two] = [0, 1, 2].map(n => entailmentInput('gpt3', n));
    expect(zero).toBe('Choose entailment, contradiction, or neutral.\n\n' + probe);
    expect(one).toContain('Relation: contradiction\n\n' + probe);
    expect(one).not.toContain('A robin is singing.');
    expect(two).toContain('Relation: entailment\n\nPremise: The door is open.');
    for (const [shots, input] of [zero, one, two].entries()) {
      expect(input.endsWith(probe)).toBe(true);
      expect(input.match(/Relation:/g)).toHaveLength(shots + 1);
      expect(entailmentInput('llama', shots)).toBe(input);
    }
    expect(() => entailmentInput('gpt3', 3)).toThrow(RangeError);
  });

  it('separates class softmax, full-label scoring and the exact tie', () => {
    expect(classProbabilities.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 15);
    expect(classProbabilities[0]).toBeCloseTo(.6652409558, 9);
    expect(continuationScores('gpt2', 500)).toEqual({
      factors: [.4, .5], scores: [.2, .25, .1], winners: ['neutral'],
    });
    expect(continuationScores('gpt2', 625).winners).toEqual(['entailment', 'neutral']);
    expect(continuationScores('gpt3', 650).winners).toEqual(['entailment']);
    expect(continuationScores('llama', 650)).toEqual({
      factors: [.4, .65, .8], scores: [.208, .25, .1], winners: ['neutral'],
    });
    expect(continuationScores('llama', 1000).winners).toEqual(['entailment']);
    for (const bad of [-1, 1001, .5, NaN]) {
      expect(() => continuationScores('gpt2', bad)).toThrow(RangeError);
    }
  });

  it('registers both editions and preserves their link to the original illustration', () => {
    const root = resolve(import.meta.dirname, '../..');
    const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
    const source = '06 Практика/03a Оценить текстовую метку от GPT-1 до LLaMA.md';
    const sourceEn = 'en/06 Practice/03a Score a label from GPT-1 to LLaMA.md';
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    expect(manifest.sections.find(s => s.id === 'practice')?.pages.find(p => p.route === 'practice/gpt-entailment'))
      .toMatchObject({ source, sourceEn });
    expect(read(sourceEn)).toContain(createHash('sha256').update(read(source)).digest('hex'));
    for (const page of [source, sourceEn]) {
      const text = read(page);
      expect(text).toContain('min-figure2-demonstrations.png');
      expect(text).toContain('https://aclanthology.org/2022.emnlp-main.759.pdf#page=2');
      expect(text).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(text).toContain('#gpt3-to-llama1');
    }
  });
});
