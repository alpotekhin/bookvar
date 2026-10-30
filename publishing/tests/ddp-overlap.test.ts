import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { withoutCollectiveDestinations } from './helpers/collective-destinations-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md';
const read = (path: string) => readFile(join(root, path), 'utf8');
const digest = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const block = (text: string, label: string) => text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();

describe('DDP overlap source integration', () => {
  it('preserves the accepted DDP explanation outside the bounded addition and metadata', async () => {
    let text = withoutCollectiveDestinations(await read(chapter));
    const start = '<!-- BEGIN REVIEW: DDP overlap 2026-10-26 -->';
    const end = '<!-- END REVIEW: DDP overlap 2026-10-26 -->';
    const addition = text.split(start)[1].split(end)[0];
    text = text.replace(start + addition + end + '\n\n', '')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-10-23')
      .replace('last_verified: 2026-10-26', 'last_verified: 2026-10-23')
      .replace('source_language: mixed\n', '');
    for (const url of [
      'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md',
      'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/distributed.md',
      'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/ddp.md',
      'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_dynamo/backends/distributed.py',
      'https://huggingface.co/spaces/nanotron/ultrascale-playbook',
      'https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd'
    ]) text = text.replace(`  - ${url}\n`, '');
    expect(digest(text)).toBe('631003b2e908f1dff7e083b99cea2d96ac15bc75dc4383612825ed1865f948a1');
  });

  it('retains the whole selected CUDA-stream explanation and all three source snippets', async () => {
    const source = await read('05 Источники/Originals/harvard-performance-2026-10-26/pytorch-v2.14.1-cuda.md');
    const original = source.split('### CUDA streams\n\n')[1]
      .split('\n(bwd-cuda-stream-semantics)=')[0].trim()
      .replace(/\{meth\}`~?([^`]+)`/g, '`$1`');
    expect(block(await read(chapter), 'PyTorch side-stream allocation lifetime')).toBe(original);
  });

  it('retains the complete selected bucket and DDPOptimizer paragraphs', async () => {
    const hf = await read('05 Источники/Originals/distributed-overview-2026-10-23/ultra_blog.md');
    const bucket = hf.split('#### **Second optimization:** Bucketing gradients\n\n')[1]
      .split('\n![dp_overlap3.svg]')[0].trim();
    expect(block(await read(chapter), 'Ultra-Scale gradient bucketing')).toBe(bucket);
    const ddp = await read('05 Источники/Originals/ddp-overlap-2026-10-26/pytorch-ddp-design.md');
    const optimizer = ddp.split('### TorchDynamo DDPOptimizer\n\n')[1]
      .split('\nSee [this blog post]')[0].trim();
    expect(block(await read(chapter), 'PyTorch DDPOptimizer overlap')).toBe(optimizer);
  });

  it('keeps the original complete SVG and full source license', async () => {
    const dir = '00 Учебник/Assets/Figures/curated/ddp-overlap-review-2026-10';
    const bytes = await readFile(join(root, dir, 'gradient-buckets.svg'));
    expect(digest(bytes)).toBe('d132f3fe62be322dfb534b51684a7233e49be22e95a106546b5c5b13b8c0da63');
    expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex'))
      .toBe('fecc6625e9dd50a4373a620080f7887a04807e15');
    expect(await read(`${dir}/LICENSE-Apache-2.0.txt`)).toBe(
      await read('05 Источники/Originals/distributed-overview-2026-10-23/Apache-2.0.txt'));
  });
});
