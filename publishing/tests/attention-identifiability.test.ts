import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
it('preserves the accepted self-attention chapter outside the numerical identifiability example',()=>{
  const chapter=readFileSync(resolve(root,'00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md'),'utf8');
  const restored=beforeTopicConsolidation(chapter,'00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md',root).replace(/<!-- attention-identifiability 2026-10-26 -->\n[\s\S]*?<!-- \/attention-identifiability -->\n\n/,'')
    .replace('last_updated: 2026-10-26','last_updated: 2026-10-22');
  expect(createHash('sha256').update(restored).digest('hex'))
    .toBe('d402a9a6810e8076173faf2e0b0e9043dbffa3673d7edb794e16e9c29f7ffd60');
  expect(chapter.match(/^```python$/gm)).toHaveLength(4);
  expect(chapter).toContain('Both rows are strictly positive softmax distributions from finite Q/K');
  expect(chapter).toContain('For fixed inputs and\nparameters the head remains deterministic.');
});
