import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { withoutLocalReductionAddition } from './helpers/local-reduction-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const sourceDir = '05 Источники/Originals/harvard-performance-2026-10-26';
const read = (path: string) => readFile(join(root, path), 'utf8');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const start = '<!-- BEGIN REVIEW: torch.compile 2026-10-26 -->';
const end = '<!-- END REVIEW: torch.compile 2026-10-26 -->';

describe('Dynamo explanation integration', () => {
  it('preserves the accepted CUDA Graphs revision outside the bounded compiler addition', async () => {
    let text = withoutLocalReductionAddition(await read(chapter));
    const addition = text.split(start)[1].split(end)[0];
    text = text.replace(start + addition + end + '\n\n', '');
    for (const path of ['compile/programming_model.dynamo_core_concepts.md',
      'compile/programming_model.recompilation.md', 'torch.compiler_dynamo_overview.md'])
      text = text.replace(`  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/${path}\n`, '');
    expect(sha(text)).toBe('d565a5d4a7320d41236fa6041869e55b9c553783916fffab9246da691de9b493');
  });

  it('retains complete selected source paragraphs without rewriting or translation', async () => {
    const source = await read(`${sourceDir}/pytorch-dynamo-core-concepts.md`);
    const text = await read(chapter);
    const selected = [
      ['PyTorch Dynamo tracing', source.split('## Dynamo Tracing\n')[1].split('\n![')[0]],
      ['PyTorch Dynamo guards', source.split('## Guards\n\n')[1].split('\nBecause guards')[0]],
      ['PyTorch Dynamo graph breaks', source.split('## Graph Breaks\n')[1].split('\nHowever, it is possible')[0]]
    ];
    for (const [label, original] of selected) {
      const excerpt = text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
        .split(`<!-- END SOURCE: ${label} -->`)[0].trim();
      expect(excerpt).toBe(original.trim());
    }
  });

  it('keeps the complete original pipeline image and full PyTorch license', async () => {
    const folder = '00 Учебник/Assets/Figures/curated/torch-compile-review-2026-10';
    const bytes = await readFile(join(root, folder, 'torch-dynamo-pipeline.png'));
    expect(sha(bytes)).toBe('cd6ac0e1646b94c21487e78f22d25b200c12e8cefe430701521dda5e901d9790');
    expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex'))
      .toBe('351689d80dc925ba5d1fbdc53c5f0ade693855e9');
    expect(await read(`${folder}/LICENSE-PyTorch.txt`)).toBe(await read(`${sourceDir}/pytorch-v2.14.1-LICENSE`));
  });
});
