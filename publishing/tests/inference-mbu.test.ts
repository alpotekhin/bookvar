import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeMathReflow } from '../tools/math-reflow-baseline.mjs';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const chapter = '00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md';
const archive = '05 Источники/Originals/mbu-2026-10-26';
const assets = '00 Учебник/Assets/Figures/curated/mbu-review-2026-10';
const read = (path: string) => readFile(join(root, path), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');

describe('Inference MBU explanation', () => {
  it('preserves the full accepted chapter outside the bounded addition', async () => {
    let text = beforeMathReflow(await read(chapter), chapter);
    text = text.replace(/<!-- BEGIN REVIEW: MBU 2026-10-26 -->[\s\S]*?<!-- END REVIEW: MBU 2026-10-26 -->\n\n/, '')
      .replace('last_updated: 2026-10-26', 'last_updated: 2026-10-14')
      .replace('last_verified: 2026-10-26', 'last_verified: 2026-10-14')
      .replace('source_language: mixed\n', '');
    expect(sha(text)).toBe('6c459a9f5153b9eaaa35e437411bb14f4301b2e49d251c6abbe278c0c4257fd5');
  });

  it('retains the entire original Appendix A, changing only heading depth and figure includes', async () => {
    const source = await read(`${archive}/jax-inference.md`);
    const original = ('### Appendix A:' + source.split('### Appendix A:')[1].split('### Appendix B:')[0]).trim();
    const text = await read(chapter);
    const actual = text.split('<!-- BEGIN SOURCE: JAX inference Appendix A -->')[1]
      .split('<!-- END SOURCE: JAX inference Appendix A -->')[0].trim();
    const expected = original.replace('### Appendix A:', '#### Appendix A:')
      .replace(/{% include figure\.liquid path="assets\/img\/batch-scaling-(latency|throughput)\.png" class="img-fluid img-small" %}/g,
        '![[02 Areas/ML & DL/' + assets + '/jax-batch-scaling-$1.png]]');
    expect(actual).toBe(expected);
  });

  it('pins sources, complete original PNGs and all license copies', async () => {
    const refs = JSON.parse(await read(`${archive}/references.json`));
    for (const source of refs.sources)
      expect(sha(await readFile(join(root, archive, source.file))), source.file).toBe(source.sha256);
    for (const figure of refs.figures) {
      const png = await readFile(join(root, assets, figure.file));
      expect(sha(png)).toBe(figure.sha256);
      expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual(figure.dimensions);
    }
    const license = await read(`${archive}/jax-LICENSE-MIT.txt`);
    expect(await read(`${assets}/LICENSE-MIT.txt`)).toBe(license);
    expect(await read('site/public/licenses/jax-scaling-book-MIT.txt')).toBe(license);
  });

  it('preserves accepted post-training chapters before separately reviewed additions', async () => {
    const files = {
      '04 Policy gradient и PPO для LLM.md': '29efb9a8c255c1b04abdcedc5c696f659a20ef45010572fe59093429698fcae5',
      '05 DPO.md': 'fd7a08d5813a5427b74a943d8a57f00978877b8edce13ac1027ccc04cf172c06',
      '06 RLVR и verifiers.md': '7b8b6d68960196ce4dd0c59fe047f974decd60b187f95b3acda55076b73efbba',
      '07 GRPO и DeepSeek-R1.md': '7173f3bbdb587d32f8d2f04c5663164c132563b6966ce7543ccc613f4e007e1a'
    };
    for (const [file, expected] of Object.entries(files)) {
      const source = '00 Учебник/12 Post-training и Alignment/' + file;
      const text = beforeTopicConsolidation(await read(source), source, root,
        '2026-10-29-constitutional-ai-additions.json');
      expect(sha(text)).toBe(expected);
    }
  });
});
