import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/07 Анатомия современной LLM/04 RoPE.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/rope-figures.json'),'utf8'));
const rows=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('RoPE explanation and examples',()=>{
  it('keeps the causal-mask qualification and independent sign/pairing examples',()=>{
    for(const text of ['Причинная маска уже задаёт порядок','при фиксированных $q,k$',
      'ответ $4-6=-2$','R_{\\rm split}=P R_{\\rm adjacent}P^T',
      'переставлять V и выходную проекцию','невращаемые координаты'])
      expect(body.toLowerCase()).toContain(text.toLowerCase());
    expect(body.match(/~~~python\n/g)).toHaveLength(4);
    expect(body).toContain('import torch');
    expect(body).toContain('positions.shape not in ((1, T), (B, T))');
  });
  it('explains cached attention through actual values and the changed-grid counterexample',()=>{
    for(const text of ['3.701805','0.117410','1.890151',
      'k_positions[None, :] <= q_positions[:, None]',
      'torch.testing.assert_close(cached, full)','stale_kv',
      '0.00232628','Кэш всё ещё экономит','конфигурация RoPE'])
      expect(body).toContain(text);
    expect(body).not.toContain('уничтожает\\nодно из преимуществ');
  });
  it('separates PI equivalence, specific scaling implementations and their limits',()=>{
    for(const text of ['2047.75','63496.04','0.00625','1.29648','1000 шагов',
      '400 шагов','200 шагов','линейная функция $r_j$',
      'не измерение attention logits','за $2^{24}$'])
      expect(body).toContain(text);
    expect(body).toContain('не нужно представлять как две разные');
    expect(fm.previous).toContain('03 Pre-norm');
    expect(fm.next).toContain('01 Dense FFN');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Далее:**');
  });
  it('uses exactly five verifiable original figures and retains attribution',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(5);
    expect(new Set(embeds)).toEqual(new Set(specs.map((s:any)=>s.asset)));
    expect(new Set(rows.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset)))
      .toEqual(new Set(embeds));
    for(const spec of specs){
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.asset))).digest('hex')).toBe(spec.sha256);
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.source_asset))).digest('hex')).toBe(spec.source_asset_sha256);
      const row=rows.find((r:any)=>r.asset===spec.asset);
      expect(row.metadata_status).toBe('verified');
      expect(['CC BY 4.0','CC BY-NC-ND 4.0']).toContain(row.license);
    }
  });
});
