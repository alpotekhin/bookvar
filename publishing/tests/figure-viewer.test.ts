import { describe, expect, it } from 'vitest';
import { clampFigureZoom, figureContainWidth, figureDisplayWidth, shouldOpenFigure } from '../../site/src/components/figure-viewer';

describe('source figure viewer', () => {
  it('fits the complete figure in both dimensions at the initial zoom', () => {
    for (const [w, h, availableW, availableH, vector] of [
      [1245, 650, 1195, 535, false], // The source GEMM panel formerly lost its bottom rows.
      [1000, 4000, 343, 600, false],
      [4000, 1000, 343, 600, false],
      [300, 150, 1200, 500, true],
      [300, 600, 1200, 500, true],
    ] as const) {
      const width = figureContainWidth(w, h, availableW, availableH, vector);
      expect(width).toBeLessThanOrEqual(availableW);
      expect(width * h / w).toBeLessThanOrEqual(availableH + 1e-9);
    }
    expect(figureDisplayWidth(1245, 1195, 1) * 650 / 1245).toBeGreaterThan(535);
    expect(figureContainWidth(479, 200, 1200, 535)).toBe(479);
    expect(figureContainWidth(300, 600, 1200, 500, true)).toBe(250);
  });
  it('rejects invalid containment dimensions rather than emitting NaN styles', () => {
    for (const bad of [0, -1, NaN, Infinity]) {
      for (let index = 0; index < 4; index++) {
        const size: [number, number, number, number] = [1000, 500, 343, 600];
        size[index] = bad;
        expect(() => figureContainWidth(...size)).toThrow();
      }
    }
  });
  it('fits the available width without automatically upscaling a small source', () => {
    expect(figureDisplayWidth(2560, 1200, 1)).toBe(1200);
    expect(figureDisplayWidth(479, 1200, 1)).toBe(479);
    expect(figureDisplayWidth(2560, 343, 2)).toBe(686);
    expect(figureDisplayWidth(2560, 343, 4)).toBe(1372);
    expect(figureDisplayWidth(3600, 343, 8)).toBe(2744);
  });
  it('bounds zoom and rejects invalid geometry', () => {
    expect(figureDisplayWidth(300, 1200, 1, true)).toBe(1200);
    expect(figureDisplayWidth(300, 343, 1, true)).toBe(343);
    expect(figureDisplayWidth(300, 343, 2, true)).toBe(686);
    expect(figureDisplayWidth(300, 1200, 1, false)).toBe(300);
    expect(clampFigureZoom(-1)).toBe(1);
    expect(clampFigureZoom(10)).toBe(8);
    expect(clampFigureZoom(5)).toBe(5);
    expect(clampFigureZoom(8.5)).toBe(8);
    expect(clampFigureZoom(1.5)).toBe(1.5);
    for (const value of [NaN, Infinity, -Infinity]) expect(() => clampFigureZoom(value)).toThrow();
    for (const value of [0, -1, NaN, Infinity]) {
      expect(() => figureDisplayWidth(value, 343, 1)).toThrow();
      expect(() => figureDisplayWidth(1000, value, 1)).toThrow();
    }
  });
  it('intercepts only an unmodified primary click pointing to the displayed image', () => {
    const click = { button: 0, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false };
    const url = 'https://example.org/bookvar/assets/source.png';
    expect(shouldOpenFigure(click, url, url)).toBe(true);
    for (const key of ['altKey', 'ctrlKey', 'metaKey', 'shiftKey'] as const)
      expect(shouldOpenFigure({ ...click, [key]: true }, url, url)).toBe(false);
    expect(shouldOpenFigure({ ...click, button: 1 }, url, url)).toBe(false);
    expect(shouldOpenFigure(click, 'https://example.org/source.pdf', url)).toBe(false);
    expect(shouldOpenFigure(click, 'javascript:alert(1)', 'javascript:alert(1)')).toBe(false);
    expect(shouldOpenFigure(click, 'not an absolute URL', url)).toBe(false);
  });
});
