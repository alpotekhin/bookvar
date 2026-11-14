import { describe, expect, it } from 'vitest';

const choose = (n: number, k: number) => {
  if (k < 0 || k > n) return 0n;
  let value = 1n;
  for (let i = 1; i <= k; i++) value = value * BigInt(n - i + 1) / BigInt(i);
  return value;
};

describe('capacity and selective audit in the imbalanced classification chapter', () => {
  it('counts every size-50 boundary draw and recovers the published means', () => {
    const population = choose(200, 50);
    let draws = 0n;
    let truePositives = 0n;
    for (let hits = 0; hits <= 30; hits++) {
      const ways = choose(30, hits) * choose(170, 50 - hits);
      draws += ways;
      truePositives += BigInt(60 + hits) * ways;
    }
    expect(draws).toBe(population);
    expect(2n * truePositives).toBe(135n * population);
    const tp = Number(truePositives) / Number(population);
    const fp = 150 - tp, fn = 100 - tp;
    expect(fp + 100 * fn).toBeCloseTo(3332.5, 8);
    expect(tp / 150).toBeCloseTo(.45, 12);
    expect(tp / 100).toBeCloseTo(.675, 12);
  });

  it('computes the audit estimate and its chance of observing no event', () => {
    const pi = 197 / 9850;
    expect(pi).toBe(.02);
    expect(1 / pi).toBe(50);
    expect(70 / (70 + 1 / pi)).toBeCloseTo(.5833333333, 9);
    const scale = 10n ** 14n;
    const scaledZero = choose(9820, 197) * scale / choose(9850, 197);
    expect(Number(scaledZero) / Number(scale)).toBeCloseTo(.544991916938, 10);
  });

  it('uses inverse inclusion weights for totals, not as an unbiased recall ratio', () => {
    // Two fully reviewed positives. Outside that queue, one positive is
    // always audited and a second has inclusion probability 1/2.
    const missed = [1, 1 + 1 / .5];
    expect((missed[0] + missed[1]) / 2).toBe(2);
    const meanRatio = missed.reduce((acc, m) => acc + 2 / (2 + m), 0) / 2;
    expect(meanRatio).toBeCloseTo(8 / 15, 12);
    expect(meanRatio).not.toBeCloseTo(2 / (2 + 2), 12);
  });
});
