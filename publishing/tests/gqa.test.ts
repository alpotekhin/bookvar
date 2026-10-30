import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/gqa-figures.json'),'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(resolve(root,p))).digest('hex');
describe('GQA explanation and original visual evidence',()=>{
  it('contains the grouped/cache and non-lossless conversion examples',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(3);
    for(const text of ['def grouped_attention','[0,0,1,1]','[0,1,0,1]',
      'query_pos = old','3, 4 и 5','2.5379','3.4621','37.5%',
      'n_kv_heads // model_parallel_size','128 MiB','31.88','torch.inference_mode()'])
      expect(body).toContain(text);
  });
  it('preserves all nine sourced images and archives superseded course copies',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(9);
    expect(new Set(embeds)).toEqual(new Set(specs.map((s:any)=>s.asset)));
    expect(new Set(assets.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset))).toEqual(new Set(embeds));
    for(const spec of specs){
      expect(sha(spec.asset)).toBe(spec.sha256);
      expect(sha(spec.source_asset)).toBe(spec.source_asset_sha256);
      if(spec.source_page){
        const row=assets.find((a:any)=>a.asset===spec.asset);
        expect(row.license).toBe('CC BY 4.0');
        expect(sha(row.license_evidence)).toBe(row.license_evidence_sha256);
        expect(row.crop_box_px).toEqual(spec.crop_box_px);
        expect(row.render_page_size_px).toEqual([2481,3508]);
        expect(row.render_dpi).toBe(300);
        expect(row.modifications).toContain('English caption');
      }
    }
    for(const name of ['gqa-quality-latency-table.png','gqa-groups-speed.png']){
      const old=assets.find((a:any)=>a.asset.endsWith('/source-first-35-36/'+name));
      expect(old.used_in).toEqual([]);
      expect(existsSync(resolve(root,old.asset))).toBe(true);
    }
  });
  it('keeps adaptation/benchmark scope and linear reader navigation explicit',()=>{
    for(const text of ['time per sample per TPUv4 chip','ROUGE-1','не весь набор задач',
      'не задержку одиночного запроса','трём запускам','не доказательство неизбежной'])
      expect(body).toContain(text);
    expect(fm.source_unit_id).toHaveLength(2);
    expect(fm.previous).toContain('_index');
    expect(fm.next).toContain('02 MLA и сжатие KV-cache');
    expect(body).toContain('К модулям:');
    expect(body).toContain('**Далее:**');
  });
});
