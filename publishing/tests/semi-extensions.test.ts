import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md';
const packet='05 Источники/Originals/semi-supervised-extensions-2026-10-29';

describe('remaining semi-supervised mechanisms',()=>{
  it('preserves the exact preceding chapter and registry and rejects unknown edits',()=>{
    const journal=JSON.parse(read('docs/audit/2026-10-29-semi-extensions-additions.json'));
    expect(journal.files).toHaveLength(2);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-semi-extensions-additions.json');
      expect(hash(current)).toBe(row.after_sha256);
      const before=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-consistency-training-additions.json');
      expect(hash(before)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(current+'\nunknown edit\n',row.source,root,
        '2026-10-29-consistency-training-additions.json')).toThrow();
    }
  });
  it('keeps full mechanisms, runnable examples and original source figures in one owner',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(14);
    expect(text.match(/^!\[\[/gm)).toHaveLength(8);
    expect(text.match(/^\$\$$/gm)).toHaveLength(44);
    for(const anchor of ['noisy-student','mixtext','dai-le-sequence-pretraining','ssl-structural-assumptions'])
      expect(text).toContain('id="'+anchor+'"');
    for(const value of ['without dropout or','value is still $H(q_i)$',
      '81 million unique','gradient recording','49/58','squared-norm hinge',
      '40 human labels','same recurrent weights','truncated backpropagation']){
      if(value==='gradient recording') expect(text).toContain('autograd recording');
      else expect(text).toContain(value);
    }
    expect(text).toContain('not a universal taxonomy');
    expect(text).toContain('IMDb prose and table disagree');
  });
  it('pins selected originals and copies only source-licensed figures into published assets',()=>{
    const manifest=JSON.parse(read(packet+'/manifest.json'));
    expect(manifest.sources).toHaveLength(13);
    for(const row of manifest.sources){
      const bytes=readFileSync(resolve(root,packet,row.file));
      expect(hash(bytes)).toBe(row.sha256);
      expect(bytes.length).toBe(row.bytes);
      expect(row.readScope).toBeTruthy();
    }
    const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/assets.json'));
    expect(assets).toHaveLength(2);
    for(const row of assets){
      expect(row.source_url).toBe('https://aclanthology.org/2020.acl-main.194/');
      expect(row.license).toBe('CC BY 4.0');
      expect(row.render_dpi).toBe(240);
      expect(hash(readFileSync(resolve(root,row.asset)))).toBe(row.sha256);
      expect(row.modifications).toContain('full caption');
      expect(row.modifications).toContain('No redrawing');
    }
    expect(manifest.figurePolicy).toContain('external links');
  });
});
