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
  it('registers the source figures without restoring retired paper copies',()=>{
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
    const english='en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md';
    const mask=assets.find((row:any)=>row.id==='curated-long-context-2c0f0d17c4d9');
    const demonstrations=assets.find((row:any)=>row.id==='curated-gpt3-min-demonstrations-0c9388f07b5f');
    const filtering=assets.find((row:any)=>row.id==='curated-gpt3-hellaswag-adversarial-filtering-9c9eebd504eb');
    const copa=assets.find((row:any)=>row.id==='curated-gpt3-holtzman-copa-65312423131c');
    const pet=assets.filter((row:any)=>row.id?.startsWith('curated-gpt3-schick-pet-figure'));
    expect(new Set(embedded)).toEqual(new Set([...figures.map((s:any)=>s.asset),mask.asset,demonstrations.asset,filtering.asset,copa.asset,...pet.map((r:any)=>r.asset)]));
    expect(copa).toMatchObject({used_in:[chapter,english],license:'CC BY 4.0',
      source_image_url:'https://arxiv.org/pdf/2104.08315v9#page=8'});
    expect(hash(read(copa.asset))).toBe(copa.sha256);
    expect(hash(read(copa.source_asset))).toBe(copa.source_asset_sha256);
    expect(read(english).toString()).toContain(copa.asset+'|');
    expect(pet.map((r:any)=>r.source_page)).toEqual([1,2]);
    for (const r of pet) {
      expect(r.used_in).toEqual([chapter,english]);
      expect(r.license).toBe('CC BY 4.0');
      expect(r.source_image_url).toBe('https://aclanthology.org/2021.naacl-main.185.pdf#page='+r.source_page);
      expect(hash(read(r.asset))).toBe(r.sha256);
      expect(hash(read(r.source_asset))).toBe(r.source_asset_sha256);
      const png=read(r.asset);
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual(r.dimensions_px);
      expect(read(english).toString()).toContain(r.asset+'|');
    }
    expect(mask.used_in).toEqual(expect.arrayContaining([chapter,english]));
    expect(mask.license).toBe('CC BY 4.0');
    expect(mask.source_image_url).toBe('https://arxiv.org/html/2310.06825v1/swa.png');
    expect(hash(read(mask.asset))).toBe(mask.source_asset_sha256);
    expect(read(english).toString()).toContain(mask.asset+'|');
    expect(read(english).toString()).toContain(demonstrations.asset+'|');
    expect(demonstrations.used_in).toEqual(expect.arrayContaining([chapter,english]));
    expect(demonstrations.license).toBe('CC BY 4.0');
    expect(hash(read(demonstrations.asset))).toBe(demonstrations.sha256);
    expect(hash(read(demonstrations.source_asset))).toBe(demonstrations.source_asset_sha256);
    expect(filtering.used_in).toEqual([chapter,english]);
    expect(filtering.license).toBe('CC BY 4.0');
    expect(filtering.figure_url).toBe('https://aclanthology.org/P19-1472.pdf#page=2');
    expect(hash(read(filtering.asset))).toBe(filtering.sha256);
    expect(hash(read(filtering.source_asset))).toBe(filtering.source_asset_sha256);
    expect(read(english).toString()).toContain(filtering.asset+'|');
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
  it('uses the same model-scoring comparison and keeps the reported splits distinct',()=>{
    const english=read('en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md').toString();
    for(const [locale,story,hella] of [
      [body,'83,2% | 84,7% | 87,7%','78,9% | 78,1% | 79,3%'],
      [english,'83.2% | 84.7% | 87.7%','78.9% | 78.1% | 79.3%']
    ]){
      expect(locale).toContain('id="gpt-llama-completion-scoring"');
      expect(locale).toContain('| StoryCloze, test | '+story+' ($K=70$)');
      expect(locale).toContain('| HellaSwag, dev | '+hella+' ($K=20$)');
      expect(locale).toContain('#gpt-llama-comparison|');
    }
    const logprobs=[-3.6,-4.8];
    expect(logprobs[0]/2).toBeLessThan(logprobs[1]/4);
    expect(logprobs[0]/12).toBeGreaterThan(logprobs[1]/14);
  });
  it('keeps GPT-2 to GPT-3 to LLaMA navigation and legacy fragment aliases',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('05 GPT-2');
    expect(fm.next).toContain('01 LLaMA');
    const footer=body.slice(body.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous.slice(0,-2)+'|');
    expect(footer).toContain(fm.next.slice(0,-2)+'|');
    for(const id of ['архитектура-знакомый-блок-в-новом-масштабе',
      'ключевые-результаты-без-упрощения','наследие-и-следующий-переход'])
      expect(body).toContain('id="'+id+'"');
  });
});
