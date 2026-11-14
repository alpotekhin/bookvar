type Vec2 = readonly [number, number];

export const ropeLabDefaults = {
  queryPosition: 2, keyPosition: 5, shift: 0, pair: 32, distance: 1000,
} as const;
export const ropeLabHeadings = {
  ru: ['Поверните запрос и ключ', 'Сравните частоты Llama 3.1'],
  en: ['Rotate the query and key', 'Compare Llama 3.1 frequencies'],
} as const;
export const llama31PairCount = 64;
export const llama31MaxDistance = 131071;
export const labQuery = [1, 2] as const;
export const labKey = [3, 4] as const;
const omega = Math.PI / 6;

function rotate([x, y]: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

const dot = ([x, y]: Vec2, [u, v]: Vec2) => x * u + y * v;

export function rotationExperiment(queryPosition: number, keyPosition: number, shift: number) {
  const query = queryPosition + shift;
  const key = keyPosition + shift;
  const delta = key - query;
  const rotatedQuery = rotate(labQuery, query * omega);
  const rotatedKey = rotate(labKey, key * omega);
  return {
    query, key, delta, rotatedQuery, rotatedKey,
    absoluteDot: dot(rotatedQuery, rotatedKey),
    relativeDot: dot(labQuery, rotate(labKey, delta * omega)),
  };
}

export function llama31Experiment(pair: number, distance: number) {
  if (!Number.isInteger(pair) || pair < 0 || pair >= llama31PairCount)
    throw new RangeError('Llama 3.1 8B pair must be in 0..63');
  if (!Number.isInteger(distance) || distance < 0 || distance > llama31MaxDistance)
    throw new RangeError('Distance must be in 0..131071 token positions');

  const original = 500000 ** (-pair / llama31PairCount);
  const period = 2 * Math.PI / original;
  const turns = 8192 / period;
  const band: 'unchanged' | 'slow' | 'blend' = turns >= 4 ? 'unchanged' : turns <= 1 ? 'slow' : 'blend';
  const weight = Math.min(1, Math.max(0, (turns - 1) / 3));
  const scaled = (1 - weight) * original / 8 + weight * original;
  const phase = original * distance;
  const scaledPhase = scaled * distance;
  return {
    period, turns, band, original, scaled, phase, scaledPhase,
    ratio: scaled / original, unitDot: Math.cos(phase), scaledUnitDot: Math.cos(scaledPhase),
  };
}
