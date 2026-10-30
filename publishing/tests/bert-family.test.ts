import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p: string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md';
const body=read(chapter).toString();
const figures=JSON.parse(read('publishing/tools/bert-family-figures.json').toString());
const registry=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const sha=(b: Buffer)=>createHash('sha256').update(b).digest('hex');
describe('BERT family chapter contracts',()=>{
  it('keeps standalone examples separate from the original MIT excerpt',()=>{
    expect(body.match(/\x60\x60\x60python\n/g)).toHaveLength(8);
    expect(body).toContain('\x60\x60\x60python title="Original implementation — MIT"');
    for(const fact of ['TinyBertPretraining','src_key_padding_mask=~valid',
      'self.word.weight','reduction="sum"','nsp_target','forward =','reverse =',
      'Enhanced Mask Decoder','emd_context_layer','не полная реализация DeBERTa'])
      expect(body).toContain(fact);
    expect(body).not.toContain('\x60\x60\x60mermaid');
    const source=read('05 Источники/Originals/bert-family-2026-10-22/masked_language_model.py').toString();
    const excerpt=body.match(/\x60\x60\x60python title="Original implementation — MIT"\n(.*?)\n\x60\x60\x60/s)![1];
    const norm=(s: string)=>s.split('\n').map(l=>l.trimEnd()).join('\n');
    expect(norm(source)).toContain(norm(excerpt));
    expect(read('site/public/licenses/microsoft-deberta-MIT.txt')).toEqual(
      read('05 Источники/Originals/bert-family-2026-10-22/deberta-LICENSE'));
  });
  it('keeps the seven reviewed originals and adds the two original SpanBERT figures',()=>{
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]!.split('|')[0]);
    const objectiveFigures=JSON.parse(read('00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/assets.json').toString());
    expect(new Set(embedded)).toEqual(new Set([...figures,...objectiveFigures].map((f:any)=>f.asset)));
    expect(figures).toHaveLength(7);
    for(const f of figures){
      const rows=registry.filter((r:any)=>r.asset===f.asset);
      expect(rows).toHaveLength(1);
      const row=rows[0];
      expect(row.used_in).toContain(chapter);
      for(const [key,value] of Object.entries(f)){
        if(['name','width','height'].includes(key)) continue;
        expect(row[key]).toEqual(value);
      }
      for(const [p,h] of [[f.asset,f.sha256],[f.source_asset,f.source_asset_sha256],
        [f.source_page_snapshot,f.source_page_sha256]])
        expect(sha(read(p))).toBe(h);
      const png=read(f.asset);
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([f.width,f.height]);
    }
    const features=body.indexOf('bert-feature-extraction-contextualized-embeddings.png');
    expect(features).toBeGreaterThan(body.indexOf('### Фиксированные признаки'));
    expect(body).not.toContain('bert-input-output.png');
    expect(body).not.toContain('bert-tasks.png');
  });
  it('preserves the reading route and removes stale figure uses',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('01 Три архитектурных паттерна');
    expect(fm.next).toContain('04 GPT-1');
    const footer=body.slice(body.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous);
    expect(footer).toContain(fm.next);
    for(const id of ['curated-bert-d54ae8e6b6eb','curated-bert-5f2b6bb2468e',
      'curated-roberta-dynamic-masking-0000000000ca','curated-deberta-equation4-0000000000cb']){
      expect(registry.find((r:any)=>r.id===id).used_in).not.toContain(chapter);
    }
  });
});
