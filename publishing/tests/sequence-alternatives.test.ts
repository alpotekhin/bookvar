import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const registry=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}).assets;
describe('sequence models explanation',()=>{
  it('keeps six executable examples and explicit operator boundaries',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(6);
    for(const s of ['def diagonal_recurrence(','def selective_ssm(','def ssd_chunked(',
      'def normalized_linear(','def gated_delta(','def ttt_linear(',
      '9.20625','1.68704','6.125','17.5','6.28125','7077888',
      'lambda_t=0.5+O','первый выход всё равно использует только первую поправку',
      'одного последнего $h$ для'])expect(body).toContain(s);
    expect(body).not.toContain('Для учебной траектории важно различать');
    expect(body).not.toContain('blob/main/mamba_ssm/');
  });
  it('preserves 32 source illustrations and complete new figures',()=>{
    const images=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(images).toHaveLength(32);
    expect(images.filter(p=>p.includes('/sequence-alternatives-2026-10-23/'))).toHaveLength(2);
    for(const path of images){
      expect(existsSync(resolve(root,path))).toBe(true);
      expect(registry.find((r:any)=>r.asset===path)?.used_in).toContain(file);
    }
    for(const name of ['linear-attention-associativity-slide.png','linear-attention-recurrent-slide.png',
      'mamba2-state-update-slide.png','gated-deltanet-state-update-slide.png','mamba2-block-figure.png']){
      expect(images.some(p=>p.endsWith(name))).toBe(false);
      const row=registry.find((r:any)=>r.asset.endsWith('/'+name));
      expect(row.used_in).not.toContain(file);
      expect(existsSync(resolve(root,row.asset))).toBe(true);
    }
    const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/sequence-figures.json'),'utf8'));
    for(const spec of specs){
      const row=registry.find((r:any)=>r.asset===spec.asset);
      expect(row.source_page).toBe(spec.page);
      expect(row.crop_box_px).toEqual(spec.crop_pixels);
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.asset))).digest('hex')).toBe(spec.sha256);
    }
  });
  it('keeps durable source anchors and explicit adjacent navigation',()=>{
    for(const id of ['attention-alternatives-overview','linear-attention-recurrent-state',
      'hybrid-sequence-models','честное-сравнение-памяти','stateful-serving-меняется-не-только-kernel'])
      expect(body.match(new RegExp('id="'+id+'"','g'))).toHaveLength(1);
    expect(fm.previous).toContain('02 Mixture of Experts');
    expect(fm.next).toContain('_index');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('К модулям:');
  });
});
