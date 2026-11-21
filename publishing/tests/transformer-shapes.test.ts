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
    expect(body.match(/~~~python\n/g)).toHaveLength(7);
    for(const fact of ['heads.untyped_storage()','parameter_layout','22696448','17576448',
      'state_dict','G.T @ X.detach()','9265217536','27795652608','19.067306',
      'заморожены','регистрирует параметры нужных форм'])expect(body).toContain(fact);
  });
  it('connects a shifted target to its vocabulary row and distinguishes GPT-2 XL from the A1 block',()=>{
    for(const fact of ['targets = sequence[:, :-1], sequence[:, 1:]','token_logits.gather',
      'input_grad + output_grad','input_grad[2]','0.2654','0.0351','2.3378',
      '1 640 452 800','1 557 611 200','9.2%','61.7%','h[:, -1, :]'])
      expect(body).toContain(fact);
    expect(body).toContain('https://aclanthology.org/E17-2025.pdf#page=2');
  });
  it('works through unequal-token accumulation, mixed-format roofline and global MFU',()=>{
    for(const fact of ['7.5','5.333333333333333','136.1','226.3','27.80%','3.47%',
      '20480','BatchNorm','2ND+DF+2NF','не число текстов','не получены бенчмарком'])
      expect(body).toContain(fact);
    expect(body).toContain('1/4+(3/4)(29/3)=7.5');
  });
  it('retains full course pages and uses the registered licensed originals',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^|\]]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(15);
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
    const compute=assets.find((a:any)=>a.asset.endsWith('/foundations/compute-memory.png'));
    expect(compute).toMatchObject({publication_status:'retired',used_in:[],former_used_in:[file]});
    expect(existsSync(resolve(root,compute.asset))).toBe(true);
    expect(body).toContain(compute.source_url);
    expect(assets.find((a:any)=>a.asset.endsWith('/tpu-load-weights.gif'))?.license).toBe('CC BY 4.0');
    expect(assets.find((a:any)=>a.asset.endsWith('/gpt2-output-projection-2.png'))?.license).toBe('CC BY-NC-SA 4.0');
  });
  it('preserves course anchors and keeps practice separate from the book sequence',()=>{
    for(const id of ['notation-and-shapes','parameter-accounting','compute-accounting','memory-accounting',
      'arithmetic-intensity','a1-construction','evidence-boundary','exercise-transition',
      'logits-to-next-token-loss','gpt2-sized-not-gpt2'])
      expect(body.split('id="'+id+'"')).toHaveLength(2);
    expect(fm.source_unit_id).toHaveLength(21);
    expect(fm.source_unit_id).toContain('lecture-02-flop-units');
    expect(fm.previous).toContain('01 Dense FFN');
    expect(fm.next).toContain('01 MHA, MQA и GQA');
    expect(body).not.toContain('**Практика:**');
    expect(body).not.toContain('06 Практика/');
    for(const s of ['id="token-ids-through-one-block"','(0.0689,0.9311)',
      '(0.0992,3.9225)','(0.0358,1.4138,-0.0358)','2 560 000 логитов'])
      expect(body).toContain(s);
  });
});
