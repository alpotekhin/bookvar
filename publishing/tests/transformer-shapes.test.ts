import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/transformer-shapes-figures.json'),'utf8'));
describe('Transformer shapes and resource explanations',()=>{
  it('contains executable shape, aliasing, inventory and backward-work cases',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(6);
    for(const fact of ['heads.untyped_storage()','parameter_layout','22696448','17576448',
      'state_dict','G.T @ X.detach()','9265217536','27795652608','19.067306',
      'заморожены','не ещё одна реализация LM'])expect(body).toContain(fact);
  });
  it('works through unequal-token accumulation, mixed-format roofline and global MFU',()=>{
    for(const fact of ['7.5','5.333333333333333','136.1','226.3','27.80%','3.47%',
      '20480','BatchNorm','2ND+DF+2NF','не число текстов','не получены бенчмарком'])
      expect(body).toContain(fact);
    expect(body).toContain('1/4+(3/4)(29/3)=7.5');
  });
  it('replaces all cropped page fragments without deleting old source assets',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(14);
    expect(new Set(embeds)).toEqual(new Set(specs.map((s:any)=>s.asset)));
    expect(new Set(assets.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset))).toEqual(new Set(embeds));
    expect(body).not.toContain('/a1-sequence-');
    for(const spec of specs){
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.asset))).digest('hex')).toBe(spec.sha256);
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.source_asset))).digest('hex')).toBe(spec.source_asset_sha256);
    }
    const old=assets.filter((a:any)=>a.asset.includes('/foundations/a1-sequence-'));
    expect(old).toHaveLength(10);
    for(const row of old){
      expect(row.used_in).toEqual([]);
      expect(existsSync(resolve(root,row.asset))).toBe(true);
    }
  });
  it('preserves course anchors and keeps practice separate from the book sequence',()=>{
    for(const id of ['notation-and-shapes','parameter-accounting','compute-accounting','memory-accounting',
      'arithmetic-intensity','a1-construction','evidence-boundary','exercise-transition'])
      expect(body.split('id="'+id+'"')).toHaveLength(2);
    expect(fm.source_unit_id).toHaveLength(21);
    expect(fm.source_unit_id).toContain('lecture-02-flop-units');
    expect(fm.previous).toContain('01 Dense FFN');
    expect(fm.next).toContain('01 Данные и pre-training');
    expect(body).toContain('**Практика:**');
  });
});
