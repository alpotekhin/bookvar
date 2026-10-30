import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const chapter=read('00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md');
const pattern=/<!-- BEGIN REVIEW: Module hooks 2026-10-27 -->\n([\s\S]*?)<!-- END REVIEW: Module hooks 2026-10-27 -->\n\n/;
const addition=chapter.match(pattern)!;
describe('complete module-hook observation example',()=>{
  it('preserves the complete accepted MLP outside the independent addition',()=>{
    expect(createHash('sha256').update(chapter.replace(pattern,'')).digest('hex'))
      .toBe('5e7aa6cd8ffc7af8ff995943582f13cbcc59c784ae2399801c16b0ee4f7547e0');
    expect(chapter.match(/^```python/gm)).toHaveLength(11);
    expect(chapter.match(/!\[\[/g)).toHaveLength(4);
  });
  it('keeps the entire Harvard hook code and its original comment language',()=>{
    const source=read('05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd');
    const original=source.split('::: {#lst-module_hooks ')[1].split('```{.python}\n')[1].split('```')[0].trim();
    const blocks=[...addition[1].matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1].trim());
    expect(blocks).toHaveLength(4);
    expect(blocks[0]).toBe(original);
    expect(addition[1]).toContain('CC BY-NC-SA 4.0');
  });
  it('distinguishes observed output/input derivatives, parameter gradients and firing conditions',()=>{
    for(const term of ['grad_output[0]','grad_input[0]','layer.weight.grad','13.5',
      'None','inspect_linear','no_grad_records','bypass_records','in-place',
      'register_full_backward_hook','positional']) expect(addition[1]).toContain(term);
    expect(addition[1]).toContain('not a low-overhead profiler or a compiler/GPU benchmark');
  });
});
