import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const chapter=read('00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md')
  .replace(/<!-- BEGIN REVIEW: Module hooks 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: Module hooks 2026-10-27 -->\n\n/,'');
const source=read('05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd');
const sha=(text:string)=>createHash('sha256').update(text).digest('hex');

describe('MLP module-state source and preservation',()=>{
  it('retains the entire previously accepted explanation outside the bounded addition',()=>{
    let previous=chapter.replace(/<!-- BEGIN REVIEW: Module state 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: Module state 2026-10-27 -->\n\n/,'')
      .replace('last_updated: 2026-10-27','last_updated: 2026-10-21');
    const layout=JSON.parse(read('publishing/tools/mlp-review-layout.json'));
    expect(layout.reflows).toHaveLength(2);
    const stripLayout=(text:string)=>text.replace(/\\(?:begin|end)\{aligned\}|\\qquad|\\\\|&|\s/g,'');
    for(const reflow of layout.reflows){
      expect(stripLayout(reflow.after)).toBe(stripLayout(reflow.before));
      expect(previous.split(reflow.after)).toHaveLength(2);
      previous=previous.replace(reflow.after,reflow.before);
    }
    expect(sha(previous)).toBe('2fd8e90374a5802bd2c3997d36ee2d945e9aa706dc2e2810c023b39a4a695e05');
  });
  it('reproduces the complete Harvard registration example, not a shortened replacement',()=>{
    const original=source.match(/\{#lst-parameter_registration[\s\S]*?```\{\.python\}\n([\s\S]*?)```/)![1];
    const blocks=[...chapter.matchAll(/```python\n([\s\S]*?)```/g)].map(match=>match[1]);
    expect(blocks).toHaveLength(7);
    expect(blocks[2]).toBe(original);
    expect(sha(source)).toBe('2cdf087ebca6106368840273f9ef6c447aba277de197223bc3c24a7022010059');
  });
  it('retains the pinned PyTorch documentation and its actual license',()=>{
    expect(sha(read('05 Источники/Originals/frameworks-2026-10-27/pytorch-v2.14.1-modules.md')))
      .toBe('f2d85df6e5fa8c46dd4a75aff7ce096dea8db0ba57c7471b773659c06136c5a8');
    const license=read('05 Источники/Originals/frameworks-2026-10-27/pytorch-5c488690-LICENSE.txt');
    expect(license).toBe(read('site/public/licenses/pytorch-v2.14.1-LICENSE.txt'));
    expect(chapter).toContain('/licenses/pytorch-v2.14.1-LICENSE.txt');
  });
});
