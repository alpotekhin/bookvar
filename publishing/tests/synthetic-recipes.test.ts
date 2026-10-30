import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(b:string|Buffer)=>createHash('sha256').update(b).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md';
const packet='05 Источники/Originals/synthetic-recipes-2026-10-29';

describe('source-faithful synthetic recipe additions',()=>{
  it('restores the exact preceding owners and registry without accepting unknown edits',()=>{
    const journal=JSON.parse(read('docs/audit/2026-10-29-synthetic-recipes-additions.json'));
    expect(journal.files).toHaveLength(4);
    for(const row of journal.files){
      const text=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-synthetic-recipes-additions.json');
      expect(hash(text)).toBe(row.after_sha256);
      const previous=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-semi-extensions-additions.json');
      expect(hash(previous)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(text+'\nunknown edit\n',row.source,root,
        '2026-10-29-semi-extensions-additions.json')).toThrow();
    }
  });
  it('keeps the existing full explanation and develops the three historical recipes',()=>{
    const text=read(chapter);
    expect(text.match(/^!\[\[/gm)).toHaveLength(12);
    expect(text.match(/^(```|~~~)python$/gm)).toHaveLength(2);
    for(const value of ['CodeTextbook','CodeExercises','Instruction reversal',
      '50.6%','4T sampled tokens','20K human-annotated records']){
      expect(text).toContain(value);
    }
    for(const id of ['phi-synthetic-recipes','nemotron4-synthetic-alignment'])
      expect(text).toContain('id="'+id+'"');
    expect(text).toContain('not an isolated');
    expect(text).toContain('checkpoint index');
    expect(text).not.toContain('Quality} \\propto');
    for(const [family,id] of [['Phi','phi-synthetic-recipes'],['Nemotron','nemotron4-synthetic-alignment']])
      expect(read('02 Атлас моделей/Семейства/'+family+'.md')).toContain('#'+id+'|');
  });
  it('pins the selected articles and all three unchanged licensed original PNGs',()=>{
    const manifest=JSON.parse(read(packet+'/manifest.json'));
    expect(manifest.sources).toHaveLength(6);
    for(const row of manifest.sources){
      const bytes=readFileSync(resolve(root,packet,row.file));
      expect(hash(bytes)).toBe(row.sha256);
      expect(bytes.length).toBe(row.bytes);
      expect(row.readScope).toBeTruthy();
    }
    const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/synthetic-recipes-2026-10-29/assets.json'));
    expect(assets).toHaveLength(3);
    for(const row of assets){
      const image=readFileSync(resolve(root,row.asset));
      expect(hash(image)).toBe(row.sha256);
      expect(image.equals(readFileSync(resolve(root,row.source_asset)))).toBe(true);
      expect([image.readUInt32BE(16),image.readUInt32BE(20)]).toEqual(row.dimensions_px);
      expect(row.license).toBe('CC BY 4.0');
      expect(read(row.license_evidence)).toContain('CC BY 4.0');
      expect(row.derivation).toBe('source-image-copy');
      expect(row.modifications).toContain('No crop');
    }
  });
});
