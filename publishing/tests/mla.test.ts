import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/mla-figures.json'),'utf8'));
const originals=JSON.parse(readFileSync(resolve(root,'publishing/tools/deepseek-v2-mla-originals.json'),'utf8'));
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('MLA chapter explanation and scope',()=>{
  it('includes the numerical obstruction, normalized cache and complete two-mode example',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(4);
    for(const s of ['class TinyMLA','same_latents','latent_output','q_gain','kv_gain',
      'math.sqrt(4 + 2)','[B,T,H,S]','11110','11111','torch.inference_mode()',
      '80 чисел','20 чисел','301 989 888','369 098 752','13 254 000 640'])expect(body).toContain(s);
    expect(body).not.toContain('## Контрольные выводы');
    expect(body).toContain('не воспроизводит скорость');
    expect(body).toContain('веса FP8');
    expect(body).toContain('8 H800');
    for(const s of ['420 млрд','1,33 трлн','34{,}6/860{,}2','4,0 процентного пункта',
      'Таблица не показывает, какой','ключ на этой упрощённой схеме не показан'])
      expect(body).toContain(s);
  });
  it('continues the original two-latent example through the positional term and value reconstruction',()=>{
    const withoutRoPE=body.indexOf('## Один запрос к двум латентам');
    const worked=body.indexOf('id="mla-decoupled-worked-example"');
    const code=body.indexOf('id="mla-layer-code"');
    expect(withoutRoPE).toBeGreaterThan(0);
    expect(worked).toBeGreaterThan(withoutRoPE);
    expect(code).toBeGreaterThan(worked);
    for(const fact of [String.raw`\sqrt{3/5}`, '0.3155', '0.6845', '0.6310,2.7381',
      'больший вес получает вторая позиция', '8 вместо 14 чисел'])
      expect(body.slice(worked,code)).toContain(fact);
  });
  it('keeps five originals, preserves historical assets and does not reintroduce the disputed operation plot',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(5);
    const historical=specs.filter((s:any)=>s.rights_status==='paper-figure-license-unverified');
    expect(historical).toHaveLength(3);
    for(const {asset} of historical){
      expect(assets.find((a:any)=>a.asset===asset)).toMatchObject({
        publication_status:'retired', used_in:[], rights_status:'paper-figure-license-unverified',
      });
      expect(existsSync(resolve(root,asset))).toBe(true);
    }
    expect(new Set(embeds)).toEqual(new Set([
      ...specs.filter((s:any)=>!historical.includes(s)), ...originals,
    ].map((s:any)=>s.asset)));
    expect(new Set(assets.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset))).toEqual(new Set(embeds));
    const previous=assets.find((a:any)=>a.asset==='00 Учебник/Assets/Figures/deepseek-v2-figure2-hq.png');
    expect(previous.used_in).toEqual([]);
    expect(existsSync(resolve(root,previous.asset))).toBe(true);
    expect(body).not.toContain('mla-weight-absorption-orders.png');
    expect(readFileSync(resolve(root,'site/public/licenses/deepseek-v2-ec98ee3c-MIT.txt'),'utf8'))
      .toContain('Copyright (c) 2023 DeepSeek');
  });
  it('preserves stable anchors, course units and previous/next reading order',()=>{
    expect(fm.source_unit_id).toHaveLength(3);
    expect(fm.previous).toContain('01 MHA, MQA и GQA');
    expect(fm.next).toContain('03 Длинный контекст');
    expect(body.match(/id="deepseek-mla-cache-compression"/g)).toHaveLength(1);
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Далее:**');
  });
});
