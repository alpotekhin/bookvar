import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md';
const body=read(chapter).toString();
const figures=JSON.parse(read('publishing/tools/gpt2-figures.json').toString());
const assets=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
describe('GPT-2 chapter contracts',()=>{
  it('keeps standalone examples, exact source excerpt and corrected evaluation protocols',()=>{
    expect(body.match(/\x60\x60\x60python\n/g)).toHaveLength(2);
    const excerpt=body.match(/\x60\x60\x60python title="Original implementation — Modified MIT"\n(.*?)\n\x60\x60\x60/s)![1];
    expect(read('05 Источники/Originals/gpt2-2026-10-22/model.py').toString()).toContain(excerpt);
    expect(read('site/public/licenses/openai-gpt2-Modified-MIT.txt')).toEqual(read('05 Источники/Originals/gpt2-2026-10-22/LICENSE'));
    for(const fact of ['124 439 808','52.66%','63.24%','stop-word filter','two-shot',
      'class TinyGPT2','self.final_norm','same logits','0.5–1 F1','4.1% exact match',
      'P+G-1','семь позиций вместо четырёх','индекс 7 вместо 4'])
      expect(body).toContain(fact);
    expect(body).not.toContain('\x60\x60\x60mermaid');
  });
  it('registers precisely the six embedded originals and their source/license evidence',()=>{
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set(figures.map((s:any)=>s.asset)));
    expect(figures).toHaveLength(6);
    for(const f of figures){
      const matches=assets.filter((r:any)=>r.asset===f.asset);
      expect(matches).toHaveLength(1);
      const row=matches[0];
      expect(row.used_in).toContain(chapter);
      for(const [key,value] of Object.entries(f)){
        if(['name','width','height','frames'].includes(key))continue;
        expect(row[key]).toEqual(value);
      }
      expect(hash(read(f.asset))).toBe(f.sha256);
      expect(read(f.asset)).toEqual(read(f.source_asset));
      expect(hash(read(f.source_page_snapshot))).toBe(f.source_page_sha256);
    }
    expect(new Set(assets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset)))
      .toEqual(new Set(embedded));
  });
  it('keeps GPT-1 to GPT-2 to GPT-3 navigation',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('04 GPT-1');
    expect(fm.next).toContain('06 GPT-3');
    const footer=body.slice(body.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous.slice(0,-2)+'|');
    expect(footer).toContain(fm.next.slice(0,-2)+'|');
  });
});
