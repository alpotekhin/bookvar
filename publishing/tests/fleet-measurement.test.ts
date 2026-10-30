import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md';
const archive = '05 Источники/Originals/fleet-measurement-2026-10-26';
const read = (path: string) => readFile(join(root, path), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
const excerpt = (text: string, label: string) => text.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();

describe('Fleet measurement source integration', () => {
  it('preserves the complete accepted chapter outside the bounded addition and metadata', async () => {
    let text = await read(chapter);
    const start = '<!-- BEGIN REVIEW: fleet measurement 2026-10-26 -->';
    const end = '<!-- END REVIEW: fleet measurement 2026-10-26 -->';
    const addition = text.split(start)[1].split(end)[0];
    text = text.replace(start + addition + end + '\n\n', '')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-10-23')
      .replace('last_verified: 2026-10-26', 'last_verified: 2026-10-23')
      .replace('source_language: mixed', 'source_language: ru');
    expect(sha(text)).toBe('1ae6298b758d8d9ec1b970930abc9ac3750f15a77c2c32d309351a4889f2154a');
  });

  it('retains the complete selected Ultra-Scale and NVIDIA explanations without translation', async () => {
    const text = await read(chapter);
    const hf = await read('05 Источники/Originals/distributed-overview-2026-10-23/ultra_blog.md');
    const original = 'When you’re measuring' + hf.split('When you’re measuring')[1]
      .split('\n\nMost training frameworks')[0];
    expect(excerpt(text, 'Ultra-Scale HFU and MFU')).toBe(original.trim());
    const source = await read(`${archive}/megatron-readme.md`);
    const paragraphs = source.split('# Training Speed and Scalability\n\n')[1]
      .split('\n# Setup')[0].split(/\n\n!\[[^\n]+\n\n?/).map(s => s.trim()).filter(Boolean);
    expect(paragraphs).toHaveLength(3);
    for (const [i, label] of ['Megatron scaling setup', 'Megatron weak scaling', 'Megatron strong scaling'].entries())
      expect(excerpt(text, label)).toBe(paragraphs[i]);
  });

  it('retains complete selected goodput entries and distinguishes overlap and recovery accounting', async () => {
    const source = await read(`${archive}/goodput-readme.md`);
    const text = await read(chapter);
    const definition = source.split('#### Retrieve Badput Breakdown\n\n')[1]
      .split('\n\nFollowing Badput')[0].trim();
    expect(excerpt(text, 'Goodput badput definition')).toBe(definition);
    for (const [label, first, next] of [
      ['Goodput data loading', ' - Data Loading Time (DATA_LOADING_SYNC)', ' - Checkpointing Time'],
      ['Goodput wasted progress', ' - Wasted Progress due to Disruption', ' - Infrastructure Recovery Time']
    ]) {
      const original = first + source.split(first)[1].split(next)[0];
      expect(excerpt(text, label)).toBe(original.trim());
    }
    expect(text).toContain('Assign each interval once');
    expect(text).toContain('not a claim that the\nlibrary calculates MFU');
  });

  it('pins source bytes, preserves native figures and ships matching licenses', async () => {
    const refs = JSON.parse(await read(`${archive}/references.json`));
    for (const item of [...refs.files, ...refs.figures])
      expect(sha(await readFile(join(root, item.path))), item.path).toBe(item.sha256);
    for (const item of refs.figures) {
      const bytes = await readFile(join(root, item.path));
      expect([bytes.readUInt32BE(16), bytes.readUInt32BE(20)]).toEqual(item.dimensions);
    }
    const license = await read(`${archive}/megatron-LICENSE`);
    expect(license).toContain('The following applies to all files unless otherwise noted');
    expect(await read('00 Учебник/Assets/Figures/curated/fleet-measurement-2026-10/LICENSE-Megatron.txt')).toBe(license);
    expect(await read('site/public/licenses/megatron-73a28a1-LICENSE.txt')).toBe(license);
    expect(await read('site/public/licenses/goodput-fa782faa-Apache-2.0.txt')).toBe(await read(`${archive}/goodput-LICENSE`));
  });
});
