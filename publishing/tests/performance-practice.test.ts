import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '06 Практика/13 Оптимизировать один Transformer step.md';
const archive = '05 Источники/Originals/performance-practice-2026-10-26';
const read = (path: string) => readFile(join(root, path), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
const block = (text: string, label: string) => text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();
const dedent = (text: string) => text.replace(/^ {4}/gm, '').trim();

describe('Performance practice source integration', () => {
  it('preserves the previous EDLS and TinyTorch routes outside the bounded extension', async () => {
    let text = await read(chapter);
    const start = '<!-- BEGIN REVIEW: performance evidence 2026-10-26 -->';
    const end = '<!-- END REVIEW: performance evidence 2026-10-26 -->';
    const addition = text.split(start)[1].split(end)[0];
    text = text.replace(start + addition + end + '\n\n', '')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-09-15')
      .replace('last_verified: 2026-10-26\n', '').replace('source_language: mixed\n', '');
    expect(sha(text)).toBe('444962666a6abf9f73459c32ce666a74da182003d90840cab1cf3aa217aa59d0');
  });

  it('retains the original complete Harvard sequence and feasibility expression', async () => {
    const text = await read(chapter);
    const source = await read(`${archive}/harvard-lab09.py`);
    const sequence = source.split('The shared sequence is:\n\n')[1].split('\n            """')[0].trim();
    expect(block(text, 'Harvard performance lab sequence')).toBe(sequence);
    const feasibility = '```\nfeasible =' + source.split('```\nfeasible =')[1].split('\n```')[0] + '\n```';
    expect(block(text, 'Harvard performance feasibility')).toBe(feasibility);
  });

  it('retains complete selected SciPy paragraphs, adapting only docstring indentation', async () => {
    const source = await read(`${archive}/scipy-stats-py.py`);
    const doc = source.split('def ttest_ind(a, b,')[1].split('    """')[1];
    const intro = dedent(doc.split('\n    Parameters')[0]);
    const sign = '    The statistic is calculated as ' + doc.split('    The statistic is calculated as ')[1]
      .split('\n\n    References')[0];
    const text = await read(chapter);
    expect(block(text, 'SciPy independent means')).toBe(intro);
    expect(block(text, 'SciPy statistic sign')).toBe(dedent(sign));
  });

  it('pins source snapshots and makes the full SciPy license available', async () => {
    const hashes = {
      'harvard-lab09.py': '610454e785c393c7375fea944d6ebac0796d570f29d055285b02c31b3cea36eb',
      'harvard-deck09.tex': 'f264acdbf7c487a55e9389ec2cb64972025780d279af0139d69906637f5b168e',
      'scipy-stats-py.py': 'bfd5ab202427ec19ad39a7d985b8fe3a5b283718ae1adec0bd8fc42f605e5988',
      'scipy-LICENSE.txt': '221e59f5e910fd7f94e44f0dac77436a11338c285c6346232e4a850a50da0e94',
      'nist-sample-size.html': '53f8dfb61d33842e71e79e60ab325be0d782df614325ae79d776c65606ab0c60'
    };
    for (const [name, expected] of Object.entries(hashes))
      expect(sha(await readFile(join(root, archive, name))), name).toBe(expected);
    expect(await read('site/public/licenses/scipy-1.18.0-LICENSE.txt')).toBe(await read(`${archive}/scipy-LICENSE.txt`));
    const text = await read(chapter);
    expect(text).toContain('Those numbers\nare not measurements');
    expect(text).toContain('hold: no feasible option');
  });
});
