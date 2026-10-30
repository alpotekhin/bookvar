import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md';
const body=read(chapter).toString();
const figures=JSON.parse(read('publishing/tools/gpt3-figures.json').toString());
const assets=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
describe('GPT-3 chapter contracts',()=>{
  it('retains complete examples and separates scoring, training and evaluation protocols',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(2);
    for(const fact of ['completion_logprob','prompt_length - 1','0.36, 0.64','768 MiB','9 GiB',
      '2022','2058','0.424','Books2','0.43','72.5%','22.0%','80.2','80.4','2209','86.9%',
      '2048','2944','86.0%','10 000','2080','57','Table 3.10','Table 3.11','Table 3.12',
      '49–54%','48–57%','SacreBLEU','tokens-through-stack-frame-385.png']) {
      expect(body).toContain(fact);
    }
    expect(body).not.toContain('```mermaid');
    expect(body).not.toContain('Научный вопрос состоял');
  });
  it('registers all five licensed figures and keeps retired paper copies out of the chapter',()=>{
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set(figures.map((s:any)=>s.asset)));
    expect(figures).toHaveLength(5);
    for(const f of figures){
      const rows=assets.filter((r:any)=>r.asset===f.asset);
      expect(rows).toHaveLength(1);
      for(const [key,value] of Object.entries(f)){
        if(['name','width','height','frames'].includes(key))continue;
        expect(rows[0][key]).toEqual(value);
      }
      expect(hash(read(f.asset))).toBe(f.sha256);
      expect(hash(read(f.source_asset))).toBe(f.source_asset_sha256);
      if(f.derivation==='source-gif-copy')expect(read(f.asset).equals(read(f.source_asset))).toBe(true);
    }
    expect(figures.filter((f:any)=>f.rights_status==='source-license-verified')).toHaveLength(5);
    expect(figures.filter((f:any)=>f.rights_status==='paper-figure-license-unverified')).toHaveLength(0);
    for(const name of ['zero-one-few-shot-vs-finetuning.png','model-sizes-table-2-1.png'])
      expect(body).not.toContain(name);
    expect(new Set(assets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset)))
      .toEqual(new Set(embedded));
  });
  it('keeps GPT-2 to GPT-3 to T5 navigation and legacy fragment aliases',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('05 GPT-2');
    expect(fm.next).toContain('07 T5');
    const footer=body.slice(body.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous.slice(0,-2)+'|');
    expect(footer).toContain(fm.next.slice(0,-2)+'|');
    for(const id of ['архитектура-знакомый-блок-в-новом-масштабе',
      'ключевые-результаты-без-упрощения','наследие-и-следующий-переход'])
      expect(body).toContain('id="'+id+'"');
  });
});
