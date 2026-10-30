import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {tracingStart,tracingEnd,withoutTracingControlAddition} from './helpers/tracing-control-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
const chapter=read('00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md');
const sourceDir='05 Источники/Originals/frameworks-2026-10-27/';
const section=chapter.split(tracingStart)[1].split(tracingEnd)[0];
const blocks=[...section.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]);

describe('Tracing control-flow source integration',()=>{
  it('preserves the full accepted S8 outside its new bounded explanation',()=>{
    expect(hash(withoutTracingControlAddition(chapter)))
      .toBe('61b8e6b36d4484c9d3ba2d920942701af321638e77556b7215fde7dbaa39922b');
  });
  it('retains the full Harvard failure example, adding only its documented import',()=>{
    const source=read(sourceDir+'chapter-2bd97c5.qmd');
    const example=source.match(/\{#lst-tracing-silent-failure[\s\S]*?```\{\.python\}\n([\s\S]*?)```/)![1];
    expect(blocks).toHaveLength(7);
    expect(blocks[0].replace(/^import torch\n\n/,'')).toBe(example);
  });
  it('retains both original PyTorch control-flow classes and the pinned API evidence',()=>{
    const source=read(sourceDir+'pytorch-5c488690-export-model.md');
    const selected=source.split('#### Dynamic Data-Dependent Control Flow')[1];
    const example=selected.match(/```python\n([\s\S]*?)```/)![1];
    expect(blocks[5]).toBe(example);
    expect(hash(source)).toBe('5ab3a99b76f426c4f0391d9d381e4437daca204b274751ef1eeb7079affac23b');
    expect(hash(read(sourceDir+'pytorch-5c488690-trace.py')))
      .toBe('033128a7919976d29b460124755efde0e15c1501e701bccc40646f0e97539429');
    expect(read(sourceDir+'pytorch-5c488690-torchscript-status.md')).toContain('TorchScript is deprecated');
    expect(section).toContain('/licenses/pytorch-v2.14.1-LICENSE.txt');
  });
});
