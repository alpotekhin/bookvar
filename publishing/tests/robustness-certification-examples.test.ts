import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';

const chapter = readFileSync(new URL('../../00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем.md', import.meta.url), 'utf8');

function upperTail(n: number, k: number, p: number) {
  // Log terms avoid overflow from large binomial coefficients.
  let logCoefficient = 0;
  let tail = 0;
  for (let i = 0; i <= n; i++) {
    if (i >= k) tail += Math.exp(logCoefficient + i * Math.log(p) + (n - i) * Math.log1p(-p));
    if (i < n) logCoefficient += Math.log(n - i) - Math.log(i + 1);
  }
  return tail;
}

function lowerBound(n: number, k: number, alpha: number) {
  if (k === 0) return 0;
  let lo = 0, hi = 1;
  for (let i = 0; i < 50; i++) {
    const p = (lo + hi) / 2;
    if (upperTail(n, k, p) < alpha) lo = p;
    else hi = p;
  }
  return (lo + hi) / 2;
}

// Midpoint integration is independent of the quantile implementation used
// to calculate the example and accurate at these moderate z scores.
function normalCdf(z: number) {
  const steps = 2048;
  let integral = 0;
  for (let i = 0; i < steps; i++)
    integral += Math.exp(-0.5 * (z * (i + 0.5) / steps) ** 2);
  return 0.5 + z * integral / (steps * Math.sqrt(2 * Math.PI));
}

describe('robustness chapter worked examples', () => {
  it('recovers a class prior without changing the class-conditional error rates', () => {
    const q = (0.31 - 0.1) / (0.8 - 0.1);
    expect(q).toBeCloseTo(0.3, 12);
    const populations = [0.1, q].map(prior => [0.8 * prior, 0.1 * (1 - prior)]);
    for (const [i, [trueAlert, falseAlert]] of populations.entries()) {
      expect(trueAlert + falseAlert).toBeCloseTo([0.17, 0.31][i], 12);
      expect(trueAlert / (trueAlert + falseAlert)).toBeCloseTo([80 / 170, 240 / 310][i], 12);
    }
    expect(chapter).toContain('$240/310');
  });

  it('inverts the binomial tail and separates certification from the raw majority', () => {
    const majority = lowerBound(1000, 700, 0.001);
    const abstain = lowerBound(1000, 530, 0.001);
    expect(majority).toBeCloseTo(0.6535, 4);
    expect(abstain).toBeCloseTo(0.4807, 4);
    expect(abstain).toBeLessThan(0.5);
    expect(normalCdf(0.0987 / 0.25)).toBeCloseTo(majority, 4);
    expect(normalCdf(0.1311 / 0.25)).toBeCloseTo(0.7, 4);
    for (const n of [20, 1000])
      expect(lowerBound(n, n, 0.001)).toBeCloseTo(0.001 ** (1 / n), 10);
    for (const printed of ['0.6535', '0.4807', '0.0987', '0.1311'])
      expect(chapter.includes(printed + '$'), 'missing example value ' + printed).toBe(true);
  });

  it('counts certified, disproved, and unresolved examples separately', () => {
    const cases = {total: 100, certifiedCorrect: 72, counterexamples: 12};
    expect(cases.total - cases.certifiedCorrect - cases.counterexamples).toBe(16);
    expect(cases.certifiedCorrect / cases.total).toBe(0.72);
    expect(1 - cases.counterexamples / cases.total).toBe(0.88);
    expect(chapter).toContain('между 72% и 88%');
  });
});
