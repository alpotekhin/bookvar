import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {jaxRuntimeStart,jaxRuntimeEnd,withoutJaxRuntimeAddition} from './helpers/jax-runtime-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const text=(path:string)=>read(path).toString();
const sha=(value:Buffer|string)=>createHash('sha256').update(value).digest('hex');
const chapter='00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md';
const originals='05 Источники/Originals/jax-runtime-2026-10-27';
const current=text(chapter);
const section=current.split(jaxRuntimeStart)[1].split(jaxRuntimeEnd)[0];
const fragment=(label:string)=>section.split(`<!-- BEGIN SOURCE: ${label} -->`)[1]
  .split(`<!-- END SOURCE: ${label} -->`)[0].trim();
const select=(file:string,start:string,end:string)=>{
  const source=text(`${originals}/${file}`);
  const begin=source.indexOf(start);
  expect(begin).toBeGreaterThan(-1);
  const finish=source.indexOf(end,begin+start.length);
  expect(finish).toBeGreaterThan(begin);
  return source.slice(begin,finish).trim().replaceAll('```{code-cell}','```python');
};

describe('JAX runtime mechanisms in the existing compiler chapter',()=>{
  it('preserves the entire previously accepted S8 byte for byte',()=>{
    expect(sha(withoutJaxRuntimeAddition(current)))
      .toBe('4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23');
    expect(current.split(jaxRuntimeStart)).toHaveLength(2);
    expect(current.split(jaxRuntimeEnd)).toHaveLength(2);
    expect([...section.matchAll(/```python\n/g)]).toHaveLength(22);
  });

  it('keeps the complete selected original counter explanations and programs',()=>{
    const original=select('state.md',"Let's start with a simple stateful program:",
      '## Threading state through pure functions')
      .replace('{ref}`jax-101-tracing`','[Transformations](https://docs.jax.dev/en/latest/101/transformations.html)');
    expect(fragment('JAX untraceable counter')).toBe(original);
    expect(fragment('JAX threaded counter')).toBe(select('state.md',
      "The fix is to make the state part of the function's interface:",
      'What we did to the counter works'));
    expect(fragment('JAX ref update')).toBe(select('state.md',
      'Threading values is the standard approach,', 'For a ref called `x_ref`')
      .replace('{func}`jax.new_ref <jax.ref.new_ref>`',
        '[`jax.new_ref`](https://docs.jax.dev/en/latest/_autosummary/jax.ref.new_ref.html)'));
  });

  it('keeps the original rank, cache and scan explanations without the unsupported timing generalization',()=>{
    expect(fragment('JAX captured rank branch')).toBe(select('jit.md',
      'A jaxpr records the function', 'Impure functions are dangerous'));
    expect(fragment('JAX cache keys')).toBe(select('jit.md',
      'Suppose we define `f = jax.jit(g)`.', 'That last point has a practical consequence:'));
    expect(fragment('JAX scan example')).toBe(select('control-flow.md',
      'The most commonly used of the four', 'Compared to unrolling a Python loop,')
      .replace('{func}`jax.lax.scan`',
        '[`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)'));
    expect(section).toContain('does not guarantee\nconstant total compilation time');
  });

  it('keeps the exact source collective program, full image and license bytes',()=>{
    expect(fragment('JAX psum program')).toBe(select('shard-map.md',
      "```{code-cell}\nmesh1d =", 'In the case where there\'s a single axis name'));
    const asset='00 Учебник/Assets/Figures/curated/jax-runtime-2026-10-27';
    expect(read(`${asset}/psum.png`)).toEqual(read(`${originals}/psum.png`));
    expect(sha(read(`${asset}/psum.png`)))
      .toBe('ab23d21957bed8ff35f25b068861d691d28668b94170c630a2ceb750fb1d3914');
    expect(read('site/public/licenses/jax-dc4af684-Apache-2.0.txt'))
      .toEqual(read(`${originals}/LICENSE.txt`));
    expect(read('site/public/licenses/jax-dc4af684-AUTHORS.txt'))
      .toEqual(read(`${originals}/AUTHORS`));
  });

  it('binds all eight source objects to independently recorded upstream Git blobs',()=>{
    const expected:Record<string,string>={
      'AUTHORS':'7281acf41a9a829eeaadc1c1437eab62a763a4a8',
      'LICENSE.txt':'d645695673349e3947e8e5ae42332d0ac3164cd7',
      'state.md':'0a8f055adb736ba8fec6107c5ec71101d240aa18',
      'jit.md':'f4c26bf6a3e1e133ea52d499628830a3375d128a',
      'control-flow.md':'92b9985a32415d31fdbca74aeffbd38e66591a2f',
      'placement.md':'4dd844c2b34ae43f7cd38b11df261c155f963076',
      'shard-map.md':'8ec4c765748d24b1d78630bfe335b0389d3c84d2',
      'psum.png':'3d8bc67469477015dbc274daf52592246747c56f',
    };
    const manifest=JSON.parse(text(`${originals}/source-manifest.json`));
    expect(manifest.entries).toHaveLength(8);
    for(const entry of manifest.entries){
      const bytes=read(`${originals}/${entry.file}`);
      expect(sha(bytes)).toBe(entry.sha256);
      const blob=createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
      expect(blob).toBe(expected[entry.file]);
      expect(entry.git_blob).toBe(blob);
    }
  });

  it('retains the state, specialization, shape, placement and measurement boundaries',()=>{
    for(const phrase of ['[1, 1, 1]','[1, 2, 3]', 'trace_signatures',
      'TracerBoolConversionError','not a measured reduction',
      'ambient mesh', 'single_mesh', '[22, 20, 12, 17]', '[9, 22, 21, 19]',
      'four logical CPU devices', 'not four GPUs', '.block_until_ready()'])
      expect(section).toContain(phrase);
    expect(text(`${originals}/README.md`)).toContain('archived, not claimed read');
  });
});
