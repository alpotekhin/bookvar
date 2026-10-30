import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const digest=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
const path='00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md';
const text=read(path).toString();
const pattern=/<!-- BEGIN REVIEW: Micro arena 2026-10-27 -->\n([\s\S]*?)<!-- END REVIEW: Micro arena 2026-10-27 -->\n\n/;
const addition=text.match(pattern)!;
describe('micro-runtime arena walkthrough',()=>{
  it('preserves the entire accepted Edge/FL chapter and its original anchor placement',()=>{
    const baseline=beforeReaderRouteNavigation(text, path).replace(pattern,'').replace('last_updated: 2026-10-27','last_updated: 2026-10-26')
      .replace('last_verified: 2026-10-27','last_verified: 2026-10-26');
    expect(digest(baseline)).toBe('9038a3d22ef6b746b7a99720542763f94b8117641c84543499433305a001c7e3');
    expect(text.match(/^```python/gm)).toHaveLength(7);
    expect(text.match(/!\[\[/g)).toHaveLength(7);
    expect(text).toContain('<a id="ограничить-изменяемое-состояние"></a>\n\n## Почему замороженные');
  });
  it('retains both complete source C++ fragments and distinguishes code from executed bookkeeping',()=>{
    const manual=read('05 Источники/Originals/micro-runtime-2026-10-27/memory_management.md').toString();
    const original=[...manual.matchAll(/```c\+\+\n([\s\S]*?)\n```/g)].map(m=>m[1]);
    const printed=[...addition[1].matchAll(/```cpp\n([\s\S]*?)\n```/g)].map(m=>m[1]);
    expect(printed).toEqual(original);
    expect(printed).toHaveLength(2);
    for(const term of ['check_arena_plan','28 KiB','112 KiB','9568','2048',
      'No C++ build or board','not a recording from a microcontroller',
      'not the RFC\'s proposed external-buffer API']) expect(addition[1]).toContain(term);
  });
  it('binds the source manifest, original image and actual source license',()=>{
    const base='05 Источники/Originals/micro-runtime-2026-10-27/';
    const rows=JSON.parse(read(base+'source-manifest.json').toString());
    expect(rows).toHaveLength(7);
    for(const row of rows) {
      const data=read(base+row.file);
      expect(digest(data)).toBe(row.sha256);
      expect(createHash('sha1').update(`blob ${data.length}\0`).update(data).digest('hex')).toBe(row.git_blob);
    }
    expect(read(base+'tflm-LICENSE.txt')).toEqual(read('site/public/licenses/tflm-Apache-2.0.txt'));
    expect(read('00 Учебник/Assets/Figures/curated/micro-runtime-2026-10-27/model-buffer-and-arena.png'))
      .toEqual(read(base+'preallocated_tensors_bg_1.png'));
  });
});
