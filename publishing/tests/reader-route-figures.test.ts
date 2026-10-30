import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
describe('original source panels for reader-route review', () => {
  it('keeps the Qwen block as the exact JPEG embedded in the course PDF', () => {
    const temp = mkdtempSync(join(tmpdir(), 'bookvar-qwen-proof-'));
    try {
      execFileSync('pdfimages', ['-f', '10', '-l', '10', '-j',
        resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/lecture_04.pdf'),
        join(temp, 'slide')]);
      expect(readFileSync(resolve(root, '00 Учебник/Assets/Figures/curated/stanford-cs336-2026/architectures/qwen-next-block-original.jpg')))
        .toEqual(readFileSync(join(temp, 'slide-000.jpg')));
      const page = readFileSync(resolve(root, '02 Атлас моделей/Семейства/Qwen.md'), 'utf8');
      expect(page).toContain('qwen-next-block-original.jpg');
      expect(page).toContain('qwen-next-hybrid-architecture.jpg');
      expect(page).toContain('<summary>Полный слайд CS336');
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });
});
