import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';

const root=resolve(import.meta.dirname,'../..');
const chapter='00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md';
const text=readFileSync(resolve(root,chapter),'utf8');
const source=readFileSync(resolve(root,'05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd'),'utf8');
const addition=text.match(/<!-- BEGIN REVIEW: training step 2026-10-27 -->\n([\s\S]*?)<!-- END REVIEW: training step 2026-10-27 -->\n\n/)!;

describe('one complete source-faithful training step',()=>{
  it('preserves every byte of the accepted profiling chapter outside the bounded addition and dates',()=>{
    const baseline=text.replace(addition[0],'')
      .replace('last_updated: 2026-10-27','last_updated: 2026-10-23')
      .replace('last_verified: 2026-10-27','last_verified: 2026-10-23');
    expect(createHash('sha256').update(baseline).digest('hex'))
      .toBe('e930009037625adff0485b39cac3d91b30340abfe665a16049f439df785deac8');
    expect(text.match(/!\[\[/g)).toHaveLength(5);
    expect(text.match(/^```python/gm)).toHaveLength(11);
  });

  it('retains the complete original source, with its limitation exposed rather than silently fixed',()=>{
    const original=source.split('{#lst-training-step-anatomy')[1].split('```{.python}\n')[1].split('```')[0].trim();
    const fences=Array.from(addition[1].matchAll(/^```python\n([\s\S]*?)^```/gm),r=>r[1].trim());
    expect(fences).toHaveLength(6);
    expect(fences[0]).toBe(original);
    expect(addition[1]).toContain('not a standalone runnable program');
    expect(addition[1]).toContain('the next example supplies a corrected, executable program');
    expect(addition[1]).toContain('CC BY-NC-SA 4.0');
    expect(fences[1]).toContain('torch.randint(0, C, (B,))');
    expect(fences[1]).toContain('optimizer.zero_grad(set_to_none=True)');
  });

  it('connects both requested-gradient cases to measured CPU event shapes without asserting CUDA measurements',()=>{
    for(const term of ['g_logits /= B','g_a = g_h * (a > 0)',
      'parameter.detach(), before - 0.01 * gradient','event.cpu_parent',
      'len(fixed_input_products) == 5','len(trainable_input_products) == 6',
      '13_008_896','13_172_736','12_845_056','not five CUDA kernel launches',
      'record_shapes','Saved forward values and','This event list is from eager',
      '2.644112 -> 2.476242']) expect(addition[1]).toContain(term);
  });

  it('retains the complete source network diagram with pinned provenance and a dimension-qualified caption',()=>{
    const read=(path:string)=>readFileSync(resolve(root,path));
    const digest=(path:string)=>createHash('sha256').update(read(path)).digest('hex');
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/training-step-2026-10-27/assets.json').toString());
    expect(rows).toHaveLength(1);
    const row=rows[0];
    expect(digest(row.asset)).toBe('6d7bfc74b8315a0d7ab35d4cb8aeaae38bc439cc56be1c561b1b9a98b208fac9');
    expect(read(row.asset)).toEqual(read(row.source_asset));
    expect(digest(row.source_page_snapshot)).toBe(row.source_page_sha256);
    expect(digest(row.license_evidence)).toBe(row.license_evidence_sha256);
    expect(row.dimensions_px).toEqual([511,350]);
    expect(row.source_image_url).toContain('/8ec57ccfc32379ae4b37c45de949af359673f92f/');
    expect(row.license).toBe('MIT');
    expect(parse(read('05 Источники/asset-registry.yml').toString()).assets.find((r:{id:string})=>r.id===row.id)).toEqual(row);
    expect(addition[1]).toContain('Three inputs feed four hidden units');
    expect(addition[1]).toContain('widths 784–256–10');
  });
});
