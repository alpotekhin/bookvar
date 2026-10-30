import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const path = '06 Практика/14 Собрать минимальный inference engine.md';
const archive = '05 Источники/Originals/inference-practice-2026-10-26';
const read = (p: string) => readFile(join(root, p), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
const block = (text: string, name: string) => text.split(`<!-- BEGIN SOURCE: ${name} -->`)[1]
  .split(`<!-- END SOURCE: ${name} -->`)[0].trim();

describe('Inference practice source integration', () => {
  it('preserves the EDLS assignment and its existing metric requirements', async () => {
    let text = beforeTextCheckpointRuntime(await read(path));
    text = text.replace(/<!-- BEGIN REVIEW: EDLS neural admission 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural admission 2026-10-26 -->\n\n/, '');
    text = text.replace(/<!-- BEGIN REVIEW: EDLS metric receipts 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS metric receipts 2026-10-26 -->\n\n/, '');
    text = text.replace(/<!-- BEGIN REVIEW: EDLS neural KV 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural KV 2026-10-26 -->\n\n/, '');
    text = text.replace(/\n\n<!-- BEGIN REVIEW: inference practice 2026-10-26 -->[\s\S]*?<!-- END REVIEW: inference practice 2026-10-26 -->\n/, '\n')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-09-15')
      .replace('last_verified: 2026-10-26\n', '').replace('source_language: mixed\n', '');
    expect(sha(text)).toBe('5315e35516a200b68cc8d50c1fad635d33dfce549c039b882563f4be1283c165');
  });

  it('retains the complete selected English source explanation and math blocks', async () => {
    const source = await read(`${archive}/harvard-lab10.py`);
    const text = await read(path);
    const state = 'For transformer serving,' + source.split('For transformer serving,')[1].split('\n"""')[0];
    expect(block(text, 'Harvard inference state explanation')).toBe(state);
    const cost = '```\nC_total(t)' + source.split('```\nC_total(t)')[1].split('\n```')[0] + '\n```';
    expect(block(text, 'Harvard inference cost equations')).toBe(cost);
    const feasibility = '```\nfeasible =' + source.split('```\nfeasible =')[1].split('\n""",')[0];
    expect(block(text, 'Harvard inference feasibility model')).toBe(feasibility);
  });

  it('pins the complete lab, deck, repository license and retrieved HF explanation', async () => {
    for (const [file, hash] of Object.entries({
      'harvard-lab10.py': '82b6f9a57c635a3026319e976dca90ce34bc2605bd71d9856af8271de9e4e239',
      'harvard-deck10.tex': 'c4bd3440691c4e5e73ad5629364a1b6e11e05c096036f2d06d92e86637f88653',
      'harvard-LICENSE.md': 'b6f4689d6fccf48d82d0871a3b994a122427fcb0b278971f6ba008be44e2c926',
      'hf-continuous-batching.md': '2a0f647d8a09056d5775611237c5d9b1242b113f9179f8405566760213b7a605'
    })) expect(sha(await readFile(join(root, archive, file)))).toBe(hash);
    expect(await read('site/public/licenses/harvard-45ecc8d8-LICENSE.md'))
      .toBe(await read(`${archive}/harvard-LICENSE.md`));
  });

  it('reuses the full licensed original HF figures, without new generated diagrams', async () => {
    const text = await read(path);
    for (const [file, hash] of Object.entries({
      'hf-dynamic-batching.png': '1721080f5c46eed465bcf3d6e19f276037c7fea144acd3ec7b386a76f60ee641',
      'hf-continuous-batching.png': 'c57b1deeacea8c2e591687d1eb105722fd352492d86cdfd74c05f166fc96068d'
    })) {
      const p = '00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/' + file;
      expect(text).toContain('![[02 Areas/ML & DL/' + p + ']]');
      expect(sha(await readFile(join(root, p)))).toBe(hash);
    }
    expect(text).not.toContain('```mermaid');
    expect(text).toContain('not GPU\nutilization, MFU or MBU');
  });
});
