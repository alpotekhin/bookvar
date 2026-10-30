import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const digest=(value:Buffer|string)=>createHash('sha256').update(value).digest('hex');
const text=read('00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md').toString();
const pattern=/<!-- BEGIN REVIEW: JAX transforms 2026-10-27 -->\n([\s\S]*?)<!-- END REVIEW: JAX transforms 2026-10-27 -->\n\n/;
const addition=text.match(pattern)!;
const source=read('05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').toString();

describe('individual gradients through original JAX transforms',()=>{
  it('preserves every byte of the accepted AD chapter outside one addition',()=>{
    expect(digest(text.replace(pattern,'')))
      .toBe('f4a436da9642cc59dfec871c9e2134119871e81e7993b0cc2dcd68af68d31b82');
    expect(text.match(/^```python/gm)).toHaveLength(20);
    expect(text.match(/!\[\[/g)).toHaveLength(7);
  });
  it('reproduces the complete source example, keeping code, comment language and attribution',()=>{
    const original=source.split('{#lst-jax-transformations')[1].split('```{.python}\n')[1].split('```')[0].trim();
    const blocks=Array.from(addition[1].matchAll(/^```python\n([\s\S]*?)^```/gm),m=>m[1].trim());
    expect(blocks).toHaveLength(5);
    expect(blocks[0]).toBe(original);
    expect(addition[1]).toContain('CC BY-NC-SA 4.0');
    for(const term of ['in_axes=(None,0,0)','pytree','(2,2)','1.1925',
      'column_targets','wrong_gradient','torch.func.vmap','not a speedup or a multi-GPU run'])
      expect(addition[1]).toContain(term);
  });
  it('pins the read primary guide and license to the tested JAX release',()=>{
    expect(digest(read('05 Источники/Originals/frameworks-2026-10-27/jax-dc4af684-transformations.md')))
      .toBe('7a3eb4f9f726316658ad37744a6271d803d91184a8cc673ca1aebe582f1fd5b8');
    expect(digest(read('05 Источники/Originals/frameworks-2026-10-27/jax-dc4af684-LICENSE.txt')))
      .toBe('cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30');
    expect(addition[1]).toContain('JAX/JAXlib 0.12.0.1 and PyTorch 2.14.1');
  });
});
