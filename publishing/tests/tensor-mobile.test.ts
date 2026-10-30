import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect,it} from 'vitest';
import {withoutTensorStorageAddition} from './helpers/tensor-storage-baseline.js';

const root=resolve(import.meta.dirname,'../..');
it('wraps the rectangular QK example without rewriting the accepted tensor chapter',()=>{
  const chapter=readFileSync(resolve(root,'00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md'),'utf8');
  const before="QK^\\top=\n\\begin{bmatrix}1&2&0\\\\0&1&3\\end{bmatrix}\n\\begin{bmatrix}1&0\\\\0&2\\\\1&1\\end{bmatrix}\n=\\begin{bmatrix}1&4\\\\3&5\\end{bmatrix}.";
  const after="\\begin{aligned}\nQK^\\top&=\n\\begin{bmatrix}1&2&0\\\\0&1&3\\end{bmatrix}\n\\begin{bmatrix}1&0\\\\0&2\\\\1&1\\end{bmatrix}\\\\[4pt]\n&=\\begin{bmatrix}1&4\\\\3&5\\end{bmatrix}.\n\\end{aligned}";
  expect(chapter).toContain(after);
  const restored=withoutTensorStorageAddition(chapter).replace(after,before).replace('last_updated: 2026-10-27','last_updated: 2026-10-23');
  expect(createHash('sha256').update(restored).digest('hex'))
    .toBe('06ffb472ba85aa805651996971e02a7774895c85787762560e134301228753a8');
  expect(chapter.match(/^```python$/gm)).toHaveLength(2);
});
