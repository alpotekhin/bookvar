import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { llama31Experiment, ropeLabDefaults, rotationExperiment } from '../../site/src/lib/rope-lab';
import { loadManifest } from '../adapter/manifest.js';

describe('interactive RoPE lab calculations', () => {
  it('distinguishes relative offset signs and preserves the score under common shifts', () => {
    const original = rotationExperiment(2, 5, 0);
    expect(original.delta).toBe(3);
    expect(original.absoluteDot).toBeCloseTo(2, 13);
    expect(original.relativeDot).toBeCloseTo(2, 13);
    expect(Math.hypot(...original.rotatedQuery)).toBeCloseTo(Math.sqrt(5), 13);
    expect(Math.hypot(...original.rotatedKey)).toBeCloseTo(5, 13);
    for (const shift of [1, 4, 12]) {
      const moved = rotationExperiment(2, 5, shift);
      expect(moved.query).toBe(2 + shift);
      expect(moved.key).toBe(5 + shift);
      expect(moved.absoluteDot).toBeCloseTo(2, 13);
      expect(moved.relativeDot).toBeCloseTo(2, 13);
    }
    expect(rotationExperiment(2, 5, 1).rotatedKey[0]).not.toBeCloseTo(original.rotatedKey[0], 6);
    expect(rotationExperiment(5, 2, 0).absoluteDot).toBeCloseTo(-2, 13);
    expect(rotationExperiment(5, 2, 0).relativeDot).toBeCloseTo(-2, 13);
    expect(rotationExperiment(5, 5, 0).absoluteDot).toBeCloseTo(11, 13);
  });

  it('matches the wavelength branches and the three worked pairs', () => {
    const counts = { unchanged: 0, blend: 0, slow: 0 };
    for (let j = 0; j < 64; j++) {
      const raw = 1 / Math.pow(500000, j / 64);
      const wavelength = 2 * Math.PI / raw;
      const result = llama31Experiment(j, 1000);
      let expected = raw;
      if (wavelength > 8192) {
        expected = raw / 8;
      } else if (wavelength >= 2048) {
        const mix = (8192 / wavelength - 1) / 3;
        expected = raw / 8 + mix * (raw - raw / 8);
      }
      expect(result.scaled).toBeCloseTo(expected, 14);
      counts[result.band]++;
    }
    expect(counts).toEqual({ unchanged: 29, blend: 6, slow: 29 });
    expect(llama31Experiment(0, 1000).ratio).toBe(1);
    expect(llama31Experiment(40, 1000).ratio).toBe(.125);
    const example = llama31Experiment(ropeLabDefaults.pair, ropeLabDefaults.distance);
    expect(example.ratio).toBeCloseTo(.37112227951784466, 13);
    expect(example.unitDot).toBeCloseTo(.1559436947653746, 13);
    expect(example.scaledUnitDot).toBeCloseTo(.8654010374940259, 13);
  });

  it('changes phase but not the frequency ratio when the distance changes', () => {
    const samePosition = llama31Experiment(32, 0);
    const far = llama31Experiment(32, 131071);
    expect(samePosition.ratio).toBe(far.ratio);
    expect(samePosition.phase).toBe(0);
    expect(samePosition.scaledPhase).toBe(0);
    expect(samePosition.scaledUnitDot).toBe(1);
    expect(far.scaledPhase).toBeGreaterThan(1);
    for (const [pair, distance] of [[-1,0],[64,0],[0,-1],[0,131072],[0,.5],[0,NaN]]) {
      expect(() => llama31Experiment(pair, distance)).toThrow(RangeError);
    }
  });
});

describe('RoPE lab reading paths', () => {
  const root = resolve(import.meta.dirname, '../..');
  const read = (file: string) => readFileSync(resolve(root, file), 'utf8');
  const source = '06 Практика/02a Проверить повороты RoPE.md';
  const sourceEn = 'en/06 Practice/02a Explore RoPE rotations.md';
  it('registers both authored editions and links the lab from the practice index and chapter', () => {
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    const page = manifest.sections.find(section => section.id === 'practice')!
      .pages.find(page => page.route === 'practice/rope');
    expect(page).toMatchObject({ source, sourceEn });
    for (const file of ['06 Практика/_index.md', '00 Учебник/07 Анатомия современной LLM/04 RoPE.md']) {
      expect(read(file)).toContain(source.slice(0, -3));
    }
    expect(read(sourceEn)).toContain(createHash('sha256').update(read(source)).digest('hex'));
    for (const file of [source, sourceEn]) {
      expect(read(file)).toContain('00 Учебник/07 Анатомия современной LLM/04 RoPE#rope-llama31-scaling');
    }
  });
});
