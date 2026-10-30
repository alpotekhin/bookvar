import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('dense FFN explanation and source preservation',()=>{
  it('retains seven figures/animations and adds the original Geva composition plot',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(8);
    expect(embeds.filter(p=>p.endsWith('.gif'))).toHaveLength(2);
    for(const path of embeds){
      expect(existsSync(resolve(root,path))).toBe(true);
      expect(assets.find((a:any)=>a.asset===path)?.used_in).toContain(file);
    }
    expect(body).toContain('Mixtral of Experts, Figure 1');
    expect(body).not.toContain('Отдельная лицензия изображения');
  });
  it('includes full forward, sharding and backward instead of replacing manual derivations',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(3);
    for(const s of ['def swiglu(','def swiglu_backward(','np.logaddexp','fused_projection',
      'split_y','flat_x.T @ flat_du','3,3,4','4000 префиксах','примерно ниже 68%',
      'log-perplexity','1.633','135 266 304','4 328 521 728','172 MiB','8320'])
      expect(body).toContain(s);
    expect(body).toContain('нет реального обмена');
  });
  it('preserves stable anchors and adjacent navigation',()=>{
    for(const id of ['gating-значение-и-пропуск-в-разных-проекциях','точный-parameter-accounting'])
      expect(body.match(new RegExp('id="'+id+'"','g'))).toHaveLength(1);
    expect(fm.previous).toContain('04 RoPE');
    expect(fm.next).toContain('05 Transformer с нуля');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Дальше:**');
  });
});
