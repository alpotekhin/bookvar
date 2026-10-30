import { describe, expect, it } from 'vitest';
import { attentionWeights, checkAttention } from '../src/lib/causal-attention';

describe('causal attention laboratory', () => {
  const settings = { pastScore: 3, futureScore: 5, causal: true };

  it('changes the labelled past link, not a future link', () => {
    const before = attentionWeights(settings);
    const after = attentionWeights({ ...settings, pastScore: 6 });
    expect(before[1][0]).toBeCloseTo(.731058579);
    expect(after[1][0]).toBeCloseTo(.982013790);
    expect(after[1][2]).toBe(0);
    expect(after.filter((_, i) => i !== 1)).toEqual(before.filter((_, i) => i !== 1));
  });

  it('actually perturbs the future score when checking independence', () => {
    expect(attentionWeights({ ...settings, futureScore: -2 }))
      .toEqual(attentionWeights({ ...settings, futureScore: 6 }));
    const unmasked = { ...settings, causal: false };
    expect(attentionWeights(unmasked)[1][2]).toBeCloseTo(.84013791);
    expect(attentionWeights({ ...unmasked, futureScore: 6 })[1][2])
      .toBeGreaterThan(attentionWeights(unmasked)[1][2]);
    expect(checkAttention(settings).map(([, passed]) => passed)).toEqual([true, true, true, true, true]);
    expect(checkAttention(unmasked).map(([, passed]) => passed)).toEqual([true, true, false, false, false]);
  });

  it('normalizes all control positions with and without masking', () => {
    for (let pastScore = -2; pastScore <= 6; pastScore += .5) {
      for (let futureScore = -2; futureScore <= 6; futureScore += .5) {
        for (const causal of [true, false]) {
          const rows = attentionWeights({ pastScore, futureScore, causal });
          for (const [i, row] of rows.entries()) {
            expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
            expect(row.every(value => Number.isFinite(value) && value >= 0)).toBe(true);
            if (causal) expect(row.slice(i + 1).every(value => value === 0)).toBe(true);
          }
        }
      }
    }
  });
});
