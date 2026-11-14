import { expect, it } from 'vitest';

function slope(f: (z: number) => number, z: number) {
  const h = 1e-5;
  return (f(z + h) - f(z - h)) / (2 * h);
}

it('reproduces the three caption objectives and their second-logit gradients', () => {
  // Three logits, with the second as the independent variable.
  const softmax = (z: number) => {
    const unnormalized = [Math.log(8), z, 0].map(Math.exp);
    const total = unnormalized.reduce((a, b) => a + b);
    return unnormalized.map((v) => v / total);
  };
  const first = (z: number) => -Math.log(softmax(z)[0]);
  const either = (z: number) => {
    const [p1, p2] = softmax(z);
    return -Math.log(p1 + p2);
  };
  const each = (z: number) => {
    const [p1, p2] = softmax(z);
    return -(Math.log(p1) + Math.log(p2)) / 2;
  };
  for (const [objective, loss, gradient] of [
    [first, 0.2231, 0.1], [either, 0.1054, -1 / 90], [each, 1.2629, -0.4],
  ] as const) {
    expect(objective(0)).toBeCloseTo(loss, 4);
    expect(slope(objective, 0)).toBeCloseTo(gradient, 7);
  }
});

it('relabeling a sigmoid pair reverses its gradient without changing another pair term', () => {
  const pair = (z: number, y: number) => Math.log1p(Math.exp(-y * z));
  const total = (z: number, y: number) => pair(1.1, 1) + pair(z, y) + pair(-0.8, -1);
  expect(slope((z) => total(z, -1), 0)).toBeCloseTo(0.5, 7);
  expect(slope((z) => total(z, 1), 0)).toBeCloseTo(-0.5, 7);
  expect(slope((other) => pair(other, 1) + pair(0, -1), 1.1))
    .toBeCloseTo(slope((other) => pair(other, 1) + pair(0, 1), 1.1), 7);
});

it('reproduces LAVIS smoothing with two equal image IDs and with only a diagonal target', () => {
  const ids = [4, 4, 9];
  const targets = (sameImage: boolean) => {
    const valid = ids.map((id, i) => sameImage ? Number(id === ids[0]) : Number(i === 0));
    return valid.map(x => 0.9 * x / valid.reduce((a, b) => a + b) + 0.1 / valid.length);
  };
  const q = targets(true);
  expect(q[0]).toBeCloseTo(29 / 60, 12);
  expect(q[1]).toBeCloseTo(29 / 60, 12);
  expect(q[2]).toBeCloseTo(1 / 30, 12);
  const objective = (z: number, weights: number[]) => {
    const logits = [Math.log(8), z, 0];
    const normalizer = Math.log(logits.map(Math.exp).reduce((a, b) => a + b));
    return -weights.reduce((sum, w, i) => sum + w * (logits[i] - normalizer), 0);
  };
  expect(slope(z => objective(z, q), 0)).toBeCloseTo(-0.3833333333, 7);
  expect(slope(z => objective(z, targets(false)), 0)).toBeCloseTo(0.0666666667, 7);
});

it('counts separate text and image K/V for the three connector examples', () => {
  const bytes = (positions: number, layers: number) => 2 * 32 * 128 * 2 * positions * layers;
  const mib = (b: number) => b / 2 ** 20;
  for (const [input, cross, expected] of [
    [576, 0, [308, 0]], [32, 0, [36, 0]], [0, 64, [20, 8]],
  ] as const) {
    const before = bytes(40 + input, 32);
    expect([mib(before), mib(bytes(cross, 8))]).toEqual(expected);
    expect(mib(bytes(41 + input, 32) - before)).toBe(0.5);
  }
});

it('matches the strided MRoPE frequency assignment including the extra time pairs', () => {
  const sections = [4, 2, 2];
  const contiguous = sections.flatMap((n, axis) => Array(n).fill('THW'[axis]));
  const interleaved = Array(8).fill('T');
  for (const axis of [1, 2])
    for (let j = axis; j < sections[axis] * 3; j += 3) interleaved[j] = 'THW'[axis];
  expect(contiguous.join(' ')).toBe('T T T T H H W W');
  expect(interleaved.join(' ')).toBe('T H W T H W T T');
  const widthIndices = (axes: string[]) => axes.flatMap((axis, i) => axis === 'W' ? [i] : []);
  expect(widthIndices(contiguous)).toEqual([6, 7]);
  expect(widthIndices(interleaved)).toEqual([2, 5]);
  expect(Math.cos(0.5)).toBeCloseTo(0.8776, 4);
  expect(Math.cos(0.05)).toBeCloseTo(0.9988, 4);
});

it('compares overlapping tile boxes only after putting them in the same frame', () => {
  const a = [32, 16, 232, 116], b = [16, 8, 216, 108], clipped = [16, 8, 128, 108];
  const padding = [12, 6];
  const toOriginal = (box: number[], origin: number[]) =>
    box.map((v, i) => (v + origin[i % 2] - padding[i % 2]) / 0.5);
  const iou = ([x1, y1, x2, y2]: number[], [u1, v1, u2, v2]: number[]) => {
    const overlap = Math.max(0, Math.min(x2, u2) - Math.max(x1, u1))
      * Math.max(0, Math.min(y2, v2) - Math.max(y1, v1));
    return overlap / ((x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - overlap);
  };
  const aFull = toOriginal(a, [80, 40]), bFull = toOriginal(b, [96, 48]);
  expect(aFull).toEqual([200, 100, 600, 300]);
  expect(aFull).toEqual(bFull);
  expect(iou(a, b)).toBeCloseTo(0.7337, 4);
  expect(iou(aFull, bFull)).toBe(1);
  expect(toOriginal(clipped, [96, 48])).toEqual([200, 100, 424, 300]);
  expect(iou(aFull, toOriginal(clipped, [96, 48]))).toBe(0.56);
});

it('recovers the full weighted update across uneven microbatches and replica averaging', () => {
  const batches = [{ targets: 4, loss: 2, gradient: 1 }, { targets: 16, loss: 1, gradient: -0.75 }];
  const meanOfMeans = (batches[0].gradient + batches[1].gradient) / 2;
  expect(meanOfMeans).toBe(0.125);
  for (const [gamma, expectedLoss, expectedGradient] of [
    [0, 1.2, -0.4], [0.5, 4 / 3, -1 / 6], [1, 1.5, 0.125],
  ]) {
    const masses = batches.map(b => b.targets ** (1 - gamma));
    const w = masses[0] + masses[1];
    const weightedSum = (key: 'loss' | 'gradient') =>
      batches.reduce((sum, b, i) => sum + masses[i] * b[key], 0);
    expect(weightedSum('loss') / w).toBeCloseTo(expectedLoss, 12);
    expect(weightedSum('gradient') / w).toBeCloseTo(expectedGradient, 12);
    const rankGradients = batches.map((b, i) => (2 / w) * masses[i] * b.gradient);
    expect((rankGradients[0] + rankGradients[1]) / 2).toBeCloseTo(expectedGradient, 12);
  }
});

it('separates item and pair accuracy when the visual answer changes', () => {
  const measure = (pairs: boolean[][]) => [
    pairs.flat().filter(Boolean).length / (2 * pairs.length),
    pairs.filter(p => p.every(Boolean)).length / pairs.length,
  ];
  expect(measure([[true, false], [true, true], [true, true], [false, true]])).toEqual([0.75, 0.5]);
  expect(measure(Array.from({ length: 4 }, () => [true, false]))).toEqual([0.5, 0]);
});
