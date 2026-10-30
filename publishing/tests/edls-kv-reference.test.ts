import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '06 Практика/14 Собрать минимальный inference engine.md';
const archive = '05 Источники/Originals/edls-inference-2026-10-26';
const read = (p: string) => readFile(join(root, p), 'utf8');
const sha = (text: string | Buffer) => createHash('sha256').update(text).digest('hex');

describe('EDLS neural KV reference', () => {
  it('preserves the entire prior Harvard/EDLS practice outside the new neural section', async () => {
    const text = beforeTextCheckpointRuntime(await read(chapter))
      .replace(/<!-- BEGIN REVIEW: EDLS neural admission 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural admission 2026-10-26 -->\n\n/, '')
      .replace(/<!-- BEGIN REVIEW: EDLS metric receipts 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS metric receipts 2026-10-26 -->\n\n/, '')
      .replace(/<!-- BEGIN REVIEW: EDLS neural KV 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural KV 2026-10-26 -->\n\n/, '');
    expect(sha(text)).toBe('a6f5cfb5a68307f7a17c653a68a5830867fb3a5d3e03bd829fed112c71aae30e');
  });

  it('exposes the complete runnable module without a shortened replacement', async () => {
    const text = await read(chapter);
    const code = text.split('<!-- BEGIN SOLUTION: EDLS KV module -->')[1]
      .split('<!-- END SOLUTION: EDLS KV module -->')[0].trim();
    const module = await read('06 Практика/Code/edls_kv_reference.py');
    expect(await read('site/public/examples/edls_kv_reference.py')).toBe(module);
    expect(code).toBe('```python\n' + module.trimEnd() + '\n```');
    expect(text).toContain('downloadable Python file');
    expect(text).toContain('not meaningful\nlanguage responses');
  });

  it('retains the original manual-generation contract in its original language', async () => {
    const notebook = JSON.parse(await read(`${archive}/homework_week8.ipynb`));
    const paragraphs = notebook.cells.filter((cell: {cell_type: string}) => cell.cell_type === 'markdown')
      .flatMap((cell: {source: string | string[]}) =>
        (Array.isArray(cell.source) ? cell.source.join('') : cell.source).split('\n\n'));
    const original = paragraphs.find((p: string) => p.startsWith('Since we manually divide'));
    const text = await read(chapter);
    const quotation = text.split('<!-- BEGIN SOURCE: EDLS manual greedy contract -->')[1]
      .split('<!-- END SOURCE: EDLS manual greedy contract -->')[0].trim().replace(/^> /gm, '');
    expect(quotation).toBe(original);
  });

  it('pins both complete notebooks, engine interface, config and complete MIT license', async () => {
    for (const [file, hash] of Object.entries({
      'seminar.ipynb': 'da9e0318007379db227c448da621d0174f940c8517f30706fe9ef0a018b7f77b',
      'homework_week8.ipynb': 'bb8e0f4b929c0b3db6e4479b6c6a6e0d4a6b023e4b8a92f203af629793f0a2ee',
      'engine.py': 'd4e1837a38429b88945276cb89feda2d6157dc822eadbbb86dca737134482424',
      'config.py': 'ced1ea6742265f205e19f52a7818f09831b467c5eb40e2d32d40b01c841ef30b',
      'LICENSE-MIT.txt': '01908feead9275d5043e082e9fa37e415a2972dca45fd62b8489e1fd2a058a13'
    })) expect(sha(await readFile(join(root, archive, file)))).toBe(hash);
    expect(await read('site/public/licenses/edls-e632aa89-MIT.txt')).toBe(await read(`${archive}/LICENSE-MIT.txt`));
  });
});
