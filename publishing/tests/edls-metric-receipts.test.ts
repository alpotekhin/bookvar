import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const read = (p: string) => readFile(join(root, p), 'utf8');
const chapter = '06 Практика/14 Собрать минимальный inference engine.md';
const archive = '05 Источники/Originals/edls-inference-2026-10-26';
const sha = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');

describe('EDLS request metric receipts', () => {
  it('preserves the entire previously accepted neural and Harvard practice', async () => {
    const prior = beforeTextCheckpointRuntime(await read(chapter))
      .replace(/<!-- BEGIN REVIEW: EDLS neural admission 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural admission 2026-10-26 -->\n\n/, '')
      .replace(/<!-- BEGIN REVIEW: EDLS metric receipts 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS metric receipts 2026-10-26 -->\n\n/, '');
    expect(sha(prior)).toBe('c47997e5b9601a3e8ef31f6a9d669286ff11d46399604d56a7e368c25ea9848a');
  });

  it('executes the published example and independent metric/boundary checks', () => {
    const output = execFileSync('python3', ['publishing/tools/check_edls_metric_receipts.py'], {
      cwd: root, encoding: 'utf8', env: {...process.env, LC_ALL: 'C', LANG: 'C'},
    });
    expect(output).toContain('PASS actual receipt fence');
    expect(output).toContain('200 affine-clock cases');
  });

  it('retains the original three-policy assignment in its original language', async () => {
    const notebook = JSON.parse(await read(`${archive}/homework_week8.ipynb`));
    const text = notebook.cells.filter((c: {cell_type: string}) => c.cell_type === 'markdown')
      .map((c: {source: string | string[]}) => Array.isArray(c.source) ? c.source.join('') : c.source).join('\n\n');
    const quote = (await read(chapter)).split('<!-- BEGIN SOURCE: EDLS three policies -->')[1]
      .split('<!-- END SOURCE: EDLS three policies -->')[0].trim().replace(/^> /gm, '');
    expect(text).toContain(quote);
    expect(quote).toContain('at least **THREE**');
  });

  it('pins complete scheduler, metric manager, API, launch and benchmark sources', async () => {
    for (const [file, hash] of Object.entries({
      'scheduler_manager.py': 'e21dda90555d43fcebf9191bc1187543e1b2291b5232011656a64bc9e4a6b6da',
      'metric_manager.py': 'c91b4c88ab95122d229b1e9ff09ea3456008f6ad3e8731348c42450976006379',
      'api_server.py': '7297c0e4890b4c66b822540f847e2a091b596f6fbddd160a9169da4f285f1015',
      'launch.py': '0fc503fc73986a8bc59b59c846aa2061a36e2bc61f38e7f8b37bc796de8ea827',
      'test_benchmark.py': '5634426b506c09df7d36c1bf95f50081045741828c3f4662170891c9ec2233f6',
    })) expect(sha(await readFile(join(root, archive, file)))).toBe(hash);
  });
});
