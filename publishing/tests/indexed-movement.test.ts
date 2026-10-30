import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { indexedStart, indexedEnd, withoutIndexedMovementAddition } from './helpers/indexed-movement-baseline.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const originals = '05 Источники/Originals/indexed-movement-2026-10-27';
const read = (path: string) => readFile(join(root, path), 'utf8');
const digest = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const block = (text: string, label: string) => text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();
const sphinx = (text: string) => text.replaceAll(':attr:', '').replaceAll('``', '`');
const whitespace = (text: string) => text.replace(/\s+/g, ' ').trim();

describe('indexed movement integration', () => {
  it('preserves the complete accepted S8 outside the bounded addition and four primary sources', async () => {
    const text = await read(chapter);
    expect(text.split(indexedStart)).toHaveLength(2);
    expect(text.split(indexedEnd)).toHaveLength(2);
    expect(digest(withoutIndexedMovementAddition(text)))
      .toBe('b51b579453cf677bc17eb37dbbd29dd6386a75727005ca30505a8c4c156e4faf');
  });

  it('retains the original selected English gather explanation with only markup conversion', async () => {
    const source = await read(`${originals}/pytorch-_torch_docs.py`);
    const original = source.split('Gathers values along an axis specified by `dim`.')[1]
      .split('\nArgs:')[0];
    const converted = sphinx('Gathers values along an axis specified by `dim`.' + original)
      .replace('specified by::\n\n', 'specified by:\n\n```text\n')
      .replace('    out[', 'out[').replaceAll('\n    out[', '\nout[')
      .replace('\n\n`input`', '\n```\n\n`input`');
    expect(whitespace(block(await read(chapter), 'PyTorch gather coordinates')))
      .toBe(whitespace(converted));
  });

  it('retains Fey reduction prose and the PyTorch assignment warning without rewriting', async () => {
    const text = await read(chapter);
    const source = await read(`${originals}/scatter.py`);
    const paragraph = source.slice(source.indexOf('    Reduces all values'), source.indexOf('\n\n    Formally'))
      .replace(/^    /gm, '');
    expect(whitespace(block(text, 'PyTorch Scatter reduction')))
      .toBe(whitespace(sphinx(paragraph)));
    const tensor = await read(`${originals}/pytorch-_tensor_docs.py`);
    const warning = tensor.slice(tensor.indexOf('    When indices are not unique,'))
      .split('\n\n')[0].replace(/^    /gm, '');
    expect(whitespace(block(text, 'PyTorch repeated scatter assignment').replace(/^> /gm, '')))
      .toBe(whitespace(sphinx(warning)));
  });

  it('preserves native figure bytes and the complete public MIT notice', async () => {
    const asset = '00 Учебник/Assets/Figures/curated/indexed-movement-2026-10-27';
    const source = await readFile(join(root, originals, 'scatter-add.svg'));
    expect(digest(source)).toBe('6f3791b14d81bd96aab6f239f0eb985536996eb5d4941d687638e66bfac3a46f');
    expect(await readFile(join(root, asset, 'scatter-add.svg'))).toEqual(source);
    expect(await read('site/public/licenses/pytorch-scatter-f514c10-MIT.txt'))
      .toBe(await read(`${originals}/scatter-LICENSE`));
    expect(await read('site/src/styles/custom.css')).toContain("/indexed-movement-2026-10-27/scatter-add.svg");
  });

  it('keeps worked computations and portability/measurement boundaries in the chapter', async () => {
    const text = (await read(chapter)).split(indexedStart)[1].split(indexedEnd)[0];
    for (const phrase of ['[8, 7, 5, 4, 0]', '[2., 0., 4., 4.]', '[10., 0., 60., 40.]',
      'GatherElements-13', 'Gather-13', 'do not broadcast', 'a GPU schedule',
      'do not\ndefine an invertible map', 'not measured HBM traffic', 'torch.distributed.gather'])
      expect(text).toContain(phrase);
    expect(text.match(/^```python$/gm)).toHaveLength(4);
  });
});
