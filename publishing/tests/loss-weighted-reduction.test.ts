import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-30-loss-weighted-reduction-additions.json'));
const row = journal.files[0];
const current = read(row.source);
const reductionRevision = beforeTopicConsolidation(current, row.source, root,
  '2026-10-30-loss-weighted-reduction-additions.json');
const prior = beforeTopicConsolidation(current, row.source, root,
  '2026-10-30-probability-native-additions.json');

describe('weighted cross-entropy reductions', () => {
  it('restores the complete prior chapter without changing its historical hash', () => {
    expect(row.before_sha256).toBe('801ed9e1eb605d4ffe25cfbf8347734fea8963c05bc0c044c0d41071ad152531');
    expect(sha(reductionRevision)).toBe(row.after_sha256);
    expect(sha(prior)).toBe(row.before_sha256);
    expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
      '2026-10-30-probability-native-additions.json')).toThrow();
  });

  it('keeps the weighted reduction example and all eight current source images', () => {
    const fences = (s: string) => [...s.matchAll(/^```python\n([\s\S]*?)^```\s*$/gm)].map(m => m[1]);
    const blocks = fences(current);
    expect(fences(prior)).toHaveLength(5);
    expect(blocks).toHaveLength(6);
    expect([...blocks.slice(0, 4), blocks[5]]).toEqual(fences(prior));
    const images = (s: string) => [...s.matchAll(/!\[\[([^\]]+)\]\]/g)].map(m => m[1]);
    expect(images(reductionRevision)).toHaveLength(9);
    expect(images(reductionRevision)).toEqual(images(prior));
    expect(images(current)).toHaveLength(8);
    expect(images(current).filter(image => !image.includes('google-softmax-multiclass.png')))
      .toEqual(images(prior).filter(image => !/cs231n-(softmax-cross-entropy|linear-decision-boundaries)\.png/.test(image)));
    for (const mechanism of ['### Class weights and the meaning of mean',
      '0.27154744216738813', '0.5430948843347763',
      'scaled_probability_mean', 'S\\,p_j-r_j',
      'no label smoothing and no ignored targets',
      'does not invoke the PyTorch API'])
      expect(current, mechanism).toContain(mechanism);
    expect(current).toContain('https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html');
  });
});
