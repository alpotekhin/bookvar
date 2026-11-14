import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';

const text = readFileSync(new URL('../../00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем.md', import.meta.url), 'utf8');

describe('worked privacy-audit examples', () => {
  it('computes TPR, FPR and positive precision on the same records', () => {
    const rows = [...text.matchAll(/^\| (?:Мягкий|Строгий) \| (\d+) \| (\d+) \| ([\d,]+)% \|$/gm)];
    expect(rows).toHaveLength(2);
    const thresholds = rows.map(([, tp, fp, precision]) => {
      const a = Number(tp), b = Number(fp);
      expect(100 * a / (a + b)).toBeCloseTo(Number(precision.replace(',', '.')), 1);
      return {tpr: a / 100, fpr: b / 10_000};
    });
    expect(thresholds).toEqual([{tpr: 0.8, fpr: 0.01}, {tpr: 0.3, fpr: 0.001}]);
    const {tpr, fpr} = thresholds[1];
    const precision = (prior: number) => prior * tpr / (prior * tpr + (1 - prior) * fpr);
    expect(precision(100 / 10_100)).toBeCloseTo(0.75, 10);
    expect(precision(0.5)).toBeCloseTo(0.997, 3);
    const upper = 1 - Math.pow(0.05, 1 / 1000);
    expect(Math.pow(1 - upper, 1000)).toBeCloseTo(0.05, 10);
    expect(100 * upper).toBeCloseTo(0.3, 2);
  });

  it('separates per-record LiRA likelihoods even at the same observed score', () => {
    const density = (z: number, mu: number, sigma: number) =>
      Math.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
    const ratio = (out: number, inside: number) => density(2, inside, 0.5) / density(2, out, 0.5);
    expect(Math.log(ratio(2, 2.25))).toBeCloseTo(-0.125, 12);
    expect(Math.log(ratio(0.5, 2))).toBeCloseTo(4.5, 12);
    expect(ratio(0.5, 2)).toBeCloseTo(90.02, 2);
  });

  it('counts ties in canary rank and compares exposure at fixed search size', () => {
    const losses = [1.5, 3, 2.25, 2.25, 5];
    expect(losses.filter(x => x <= losses[2])).toHaveLength(3);
    const exposure = (rank: number) => Math.log2(1_000_000) - Math.log2(rank);
    expect(exposure(20)).toBeCloseTo(15.61, 2);
    expect(exposure(20_000)).toBeCloseTo(5.64, 2);
    expect(20 / 1_000_000).toBe(1 / 50_000);
  });
});
