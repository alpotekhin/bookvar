import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { withoutLocalReductionAddition } from './helpers/local-reduction-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const originals = '05 Источники/Originals/harvard-performance-2026-10-26';
const start = '<!-- BEGIN REVIEW: CUDA Graphs 2026-10-26 -->';
const end = '<!-- END REVIEW: CUDA Graphs 2026-10-26 -->';
const read = (path: string) => readFile(join(root, path), 'utf8');
const digest = (text: string | Buffer) => createHash('sha256').update(text).digest('hex');
const sourceBlock = (text: string, label: string) => text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();

describe('CUDA Graphs integration', () => {
  it('preserves accepted S8 apart from the bounded addition, metadata and math alignment', async () => {
    let text = withoutLocalReductionAddition(await read(chapter));
    const compileStart = '<!-- BEGIN REVIEW: torch.compile 2026-10-26 -->';
    const compileEnd = '<!-- END REVIEW: torch.compile 2026-10-26 -->';
    const compileAddition = text.split(compileStart)[1].split(compileEnd)[0];
    text = text.replace(compileStart + compileAddition + compileEnd + '\n\n', '');
    for (const path of ['compile/programming_model.dynamo_core_concepts.md',
      'compile/programming_model.recompilation.md', 'torch.compiler_dynamo_overview.md'])
      text = text.replace(`  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/${path}\n`, '');
    const addition = text.split(start)[1].split(end)[0];
    let baseline = text.replace(start + addition + end + '\n\n', '');
    baseline = baseline.replace('last_updated: 2026-10-26', 'last_updated: 2026-10-23')
      .replace('last_verified: 2026-10-26', 'last_verified: 2026-10-23');
    baseline = baseline.replace(
      String.raw`\begin{gathered}
a(x)=\sqrt{\frac{2}{\pi}}(x+0.044715x^3),\\
\operatorname{GELU}_{\tanh}(x)=\frac{x}{2}\left(1+\tanh a(x)\right).
\end{gathered}`,
      String.raw`\begin{aligned}
a(x)&=\sqrt{\frac{2}{\pi}}(x+0.044715x^3),\\
\operatorname{GELU}_{\tanh}(x)&=\frac{x}{2}\left(1+\tanh a(x)\right).
\end{aligned}`);
    for (const url of [
      'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md',
      'https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/',
      'https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd'
    ]) baseline = baseline.replace(`  - ${url}\n`, '');
    expect(digest(baseline)).toBe('e8d7a99ec414e09e491cddd5cb09a0b364b66ebaa9155b64ca0c9309996f2fb5');
    expect(baseline).toBe(await read('docs/audit/snapshots/2026-10-26-cuda-graphs-before.md'));
  });

  it('keeps original PyTorch overview and API code, changing only documentation markup', async () => {
    const source = await read(`${originals}/pytorch-v2.14.1-cuda.md`);
    let original = source.split('## CUDA Graphs\n')[1].split('\nSee\n')[0].trim();
    original = original.replaceAll('{class}`', '`')
      .replace('{ref}`constraints<capture-constraints>`',
        '[constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints)')
      .replace('```{warning}\n\n    This API is in beta and may change in future releases.\n```',
        '> [!warning] Original API notice\n> This API is in beta and may change in future releases.');
    expect(sourceBlock(await read(chapter), 'PyTorch CUDA Graphs overview')).toBe(original);
  });

  it('keeps the full selected graph-memory explanation in its original language', async () => {
    const source = await read(`${originals}/pytorch-v2.14.1-cuda.md`);
    const original = source.split('### Graph memory management\n')[1]
      .split('#### Sharing memory across captures')[0].trim()
      .replace('{class}`~torch.cuda.CUDAGraph`', '`torch.cuda.CUDAGraph`');
    expect(sourceBlock(await read(chapter), 'PyTorch graph memory management')).toBe(original);
  });

  it('retains the unmodified pinned image bytes and full source licenses', async () => {
    const directory = '00 Учебник/Assets/Figures/curated/cuda-graphs-review-2026-10';
    for (const [file, hash, blob] of [
      ['launch-vs-replay.png', '0067710c0299743c23eccc02fab976fd88cb9742c15363c65462b411ced064a2', '5cc2bade9a43b2cc2bda14b1d8441b8214bc92a6'],
      ['mask-rcnn-before.png', '8bc59c2ab6d28f386eebccf636b51f850dd78bc9dc787fa9d9538338ed5d81c5', '8c77a5aea0bc908b257f443620981d73a65bb533'],
      ['mask-rcnn-after.png', 'ddcad283c86d64541f5d0abe7dc86e524d153addb003d5673d8559a3ed7f5086', '4a854558e0b8e0aa3af2ba0e73891091b266d198']
    ]) {
      const bytes = await readFile(join(root, directory, file));
      expect(digest(bytes)).toBe(hash);
      expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')).toBe(blob);
    }
    expect(digest(await read(`${directory}/LICENSE-website-BSD-3-Clause.txt`)))
      .toBe('9e8b5a3ee55cbf30f0ce2b43b25daff2461f1251862d40e012f5c45916603dd2');
    expect(await read('site/public/licenses/pytorch-v2.14.1-LICENSE.txt'))
      .toBe(await read(`${originals}/pytorch-v2.14.1-LICENSE`));
  });

  it('states the backend, memory, model and benchmark boundaries explicitly', async () => {
    const text = (await read(chapter)).split(start)[1].split(end)[0];
    for (const phrase of ['not kernel\nfusion', 'not a CUDA graph\nimplementation',
      'not measurements of CUDA', 'not yet supported by the Inductor backend',
      'another graph\'s output', 'compare their labelled\ndurations',
      'not presented as a GPU run performed for this chapter']) expect(text).toContain(phrase);
  });
});
