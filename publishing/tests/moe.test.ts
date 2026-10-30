import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const rows=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('MoE explanation and source figures',()=>{
  it('matches retained MegaBlocks and LatentMoE images to their archived source bytes',()=>{
    for(const name of ['megablocks-expert-matmuls.png','standard-moe.png','latent-moe.png']){
      const row=rows.find((r:any)=>r.asset.endsWith('/'+name));
      expect(row.source_asset).toContain('05 Источники/Originals/moe-2026-10-23/');
      expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
    }
  });
  it('retains meaningful older figures and replaces small plot copies with complete originals',()=>{
    const images=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(images).toHaveLength(19);
    expect(images.filter(p=>p.includes('/moe-2026-10-23/'))).toHaveLength(8);
    for(const path of images){
      expect(existsSync(resolve(root,path))).toBe(true);
      expect(rows.find((r:any)=>r.asset===path)?.used_in).toContain(file);
    }
    expect(body).toContain('Из Training loss исключён сам вспомогательный штраф');
    expect(body).toContain('CC BY 4.0');
    expect(body).toContain('Apache-2.0');
    for(const removed of ['moe-topk-gating-equations.png','switch-load-balance-loss.png',
      'deepseek-v3-bias-routing.png','deepseek-v1-moe-slide.png','deepseek-v2-moe-slide.png'])
      expect(images.some(p=>p.endsWith(removed))).toBe(false);
    expect(existsSync(resolve(root,'site/public/licenses/olmoe-357454f4-Apache-2.0.txt'))).toBe(true);
  });
  it('connects four executable blocks to worked routing, dropping, losses and node restriction',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(4);
    for(const s of ['def dispatch_combine(','def capacity_mask(','def router_losses(',
      'def node_limited_route(','0.99024','-1.40625','32$ байта','np.add.at',
      '0.54/1.14','1.55 и 1.05','route_scale','GPU и сетевые пересылки он'])expect(body).toContain(s);
    expect(body).not.toContain('Здесь видно тонкое различие');
    expect(body).not.toContain('худший\nзначение');
  });
  it('preserves named anchors and neighboring chapter links',()=>{
    for(const id of ['moe-routing-mechanisms','moe-load-balancing','moe-systems-capacity',
      'moe-training-failures','moe-upcycling','deepseek-moe-evolution','moe-recap'])
      expect(body.match(new RegExp('id="'+id+'"','g'))).toHaveLength(1);
    expect(fm.previous).toContain('03 Длинный контекст');
    expect(fm.next).toContain('03 Mamba');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Дальше:**');
  });
});
