import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { reductionStart, reductionEnd, withoutLocalReductionAddition } from './helpers/local-reduction-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const originals = '05 Источники/Originals/reduction-primitives-2026-10-27';
const read = (path: string) => readFile(join(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

describe('local reduction tree integration', () => {
  it('preserves accepted S8 outside the addition, source metadata and one mobile math reflow', async () => {
    const text = await read(chapter);
    expect(text.split(reductionStart)).toHaveLength(2);
    expect(text.split(reductionEnd)).toHaveLength(2);
    expect(sha(withoutLocalReductionAddition(text)))
      .toBe('1aaeef7e9387d9840ac2361926df475179f23150db0c25771aea2bad0f06b33e');
  });

  it('retains the exact original CUDA helper and its complete public notice', async () => {
    const source = await read(`${originals}/reduction_kernel-5443602.cu`);
    const start = source.indexOf('template <class T> __device__ __forceinline__ T warpReduceSum(');
    const original = source.slice(start, source.indexOf('\n}', start) + 2);
    const addition = (await read(chapter)).split(reductionStart)[1].split(reductionEnd)[0];
    const code = addition.split('```cpp\n')[1].split('\n```')[0];
    expect(code).toBe(original);
    expect(await read('site/public/licenses/nvidia-cuda-samples-5443602-BSD-3-Clause.txt'))
      .toBe(await read(`${originals}/NVIDIA-cuda-samples-LICENSE`));
  });

  it('keeps the complete original animation bytes and its attribution', async () => {
    const asset = '00 Учебник/Assets/Figures/curated/reduction-primitives-2026-10-27';
    const original = await readFile(join(root, originals, 'binomial-tree.gif'));
    expect(sha(original)).toBe('ca079d087b8e9a7d4f61cf91d3d64323eb1ebeb5c8992b4c0d8c3f790de1ee8a');
    expect(await readFile(join(root, asset, 'binomial-tree.gif'))).toEqual(original);
    const text = await read(chapter);
    for (const label of ['Flogr', 'CC BY-SA 4.0', 'four frames', 'not elapsed GPU time',
      'not a complete kernel or an executed CUDA test']) expect(text).toContain(label);
  });
});
