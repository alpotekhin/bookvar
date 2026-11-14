import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md';
const body=read(chapter).toString();
const assets=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const figures=JSON.parse(read('publishing/tools/gpt2-figures.json').toString())
  .filter((figure:{asset:string})=>assets.find((row:any)=>row.asset===figure.asset)?.used_in?.includes(chapter));
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
  it('registers the original Alammar, PMLR, CoQA, translation, and LAMBADA figures',()=>{
    const xiong='00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png';
    const coqa='00 Учебник/Assets/Figures/curated/gpt-coqa-2026-11-11/coqa-figure4-conversation-flow.png';
    const translation='00 Учебник/Assets/Figures/curated/gpt-coqa-2026-11-11/liu22-figure3-translation-prefix.png';
    const lambada=assets.find((r:any)=>r.id==='curated-gpt2-lambada-in-context-6fc74d6f6c42');
    const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^|\]]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set([...figures.map((s:any)=>s.asset),xiong,coqa,translation,lambada.asset]));
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
    const en='en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language.md';
    const rows=assets.filter((r:any)=>r.asset===xiong);
    expect(rows).toHaveLength(1);
    const row=rows[0];
    expect(row.used_in).toEqual(expect.arrayContaining([chapter,en]));
    expect(read(en).toString()).toContain(xiong+'|');
    expect(row.derivation).toBe('pdf-page-render-crop');
    expect(row.license).toBe('CC BY 4.0');
    expect(hash(read(xiong))).toBe('fb882ba9d546192eafc0ee81f51244043faa986a1641138e20b50e998e0efdaf');
    expect(hash(read(row.source_asset))).toBe(row.source_asset_sha256);
    expect(hash(read(row.license_evidence))).toBe(row.license_evidence_sha256);
    const coqaRows=assets.filter((r:any)=>r.asset===coqa);
    expect(coqaRows).toHaveLength(1);
    const coqaDigest='b606973444a2a603b531303badc254705ca922b7ac3e838bce21e35fac9feba4';
    expect(coqaRows[0]).toMatchObject({
      used_in: [chapter,en],
      figure_url: 'https://aclanthology.org/Q19-1016.pdf#page=8',
      source_page: 8,
      license: 'CC BY 4.0',
      license_evidence_url: 'https://transacl.org/ojs/index.php/tacl/article/view/1572',
      derivation: 'pdf-page-render-crop',
      sha256: coqaDigest,
    });
    expect(hash(read(coqa))).toBe(coqaDigest);
    expect(hash(read(coqaRows[0].source_asset))).toBe(coqaRows[0].source_asset_sha256);
    expect(read(en).toString()).toContain(coqa+'|');
    const entry=assets.find((r:any)=>r.asset===translation);
    expect(entry).toMatchObject({source_page:12,source_printed_page:111,
      crop_box_px:[391,770,1445,1327],license:'CC BY 4.0',used_in:[chapter,en],
      source_image_url:'https://aclanthology.org/2022.deelio-1.10.pdf#page=12'});
    const png=read(translation);
    expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([1054,557]);
    expect(hash(png)).toBe(entry.sha256);
    expect(hash(read(entry.source_asset))).toBe(entry.source_asset_sha256);
    expect(read(en).toString()).toContain(translation+'|');
    expect(lambada).toMatchObject({used_in:[chapter,en],license:'CC BY 4.0',
      figure_url:'https://aclanthology.org/P16-1144.pdf#page=6'});
    expect(hash(read(lambada.asset))).toBe(lambada.sha256);
    expect(hash(read(lambada.source_asset))).toBe(lambada.source_asset_sha256);
    expect(read(en).toString()).toContain(lambada.asset+'|');
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
