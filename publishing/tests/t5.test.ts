import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md';
const body=read(chapter).toString();
const conceptPath='Concepts/Architectures/T5.md';
const concept=read(conceptPath).toString();
const paper=read('Papers/T5.md').toString();
const figures=JSON.parse(read('publishing/tools/t5-figures.json').toString());
const assets=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
describe('T5 chapter contracts',()=>{
  it('retains complete preprocessing and training examples with precise recipe boundaries',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(2);
    for(const fact of ['replace_runs','32099','9961','222 903 552','568','114',
      'decoder_input_ids != 0','labels.ne(-100)','11 целевых','random_roll=False',
      'независимыми','1.05 трлн','0.6109','18 MiB','GEGLU','C4 без downstream mixture'])
      expect(body).toContain(fact);
    expect(body).not.toContain('```mermaid');
    expect(body).not.toContain('Схема сопровождает материалы');
    expect(body).toContain('[14, 9, 13, 1, -100, -100, -100]');
    expect(body).not.toContain('[14, 9, 12, 1');
    for(const fact of ['EncoderDecoderCache.is_updated','общий текстовый префикс',
      'сбрасывают и self-KV','Сам график показывает только training loss'])expect(body).toContain(fact);
  });
  it('embeds six original licensed JMLR figures and removes obsolete usages',()=>{
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set(figures.map((s:any)=>s.asset)));
    expect(figures).toHaveLength(6);
    for(const f of figures){
      const rows=assets.filter((r:any)=>r.asset===f.asset);
      expect(rows).toHaveLength(1);
      expect(rows[0].used_in).toContain(chapter);
      expect(rows[0].license).toBe('CC BY 4.0');
      for(const [key,value] of Object.entries(f)){
        if(['name','width','height'].includes(key))continue;
        expect(rows[0][key]).toEqual(value);
      }
      expect(hash(read(f.asset))).toBe(f.sha256);
      expect(hash(read(f.source_asset))).toBe(f.source_asset_sha256);
      expect(hash(read(f.license_evidence))).toBe(f.license_evidence_sha256);
      const png=read(f.asset);
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([f.width,f.height]);
    }
    expect(new Set(assets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset)))
      .toEqual(new Set(embedded));
    for(const id of ['curated-t5-60a4f5c68c9e','curated-t5-961e963bd519'])
      expect(assets.find((r:any)=>r.id===id).used_in).toEqual([]);
    const conceptFigures=[...concept.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(conceptFigures).toEqual([figures[0].asset,figures[3].asset,figures[5].asset]);
    expect(new Set(assets.filter((r:any)=>r.used_in?.includes(conceptPath)).map((r:any)=>r.asset)))
      .toEqual(new Set(conceptFigures));
  });
  it('corrects legacy source-note and concept facts instead of retaining fabricated tables',()=>{
    for(const text of [concept,paper]){
      for(const fact of ['SQuAD 1.1','91.26','96.22','88.9','32.1','71.36','59.29'])
        expect(text).toContain(fact);
      for(const falseClaim of ['SQuAD 2.0 F1','Input re-processed','строгое, монотонное',
        '91.3 | QA'])expect(text).not.toContain(falseClaim);
      expect(text).toContain('07 T5 — text-to-text Transformer');
    }
    expect(paper).toContain('2019-10-23');
    expect(concept).toContain('74.70 | 61.14');
    expect(concept).toContain('2P / M');
    expect(concept).toContain('двунаправленное внимание');
  });
  it('keeps previous/next links and old fragment aliases',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('06 GPT-3');
    expect(fm.next).toContain('01 LLaMA');
    const footer=body.slice(body.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous.slice(0,-2)+'|');
    expect(footer).toContain(fm.next.slice(0,-2)+'|');
    for(const id of ['почему-авторы-выбрали-именно-эту-цель',
      'c4-масштаб-начинается-с-отбора-данных',
      'перенос-задача-становится-данными-а-не-новой-головой',
      'как-читать-ключевые-эксперименты'])expect(body).toContain('id="'+id+'"');
  });
});
