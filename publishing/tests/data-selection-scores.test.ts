import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const chapter='00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md';
const route='textbook/pretraining/data-selection-scores';

describe('labelled data selection publication and source boundaries',()=>{
  it('inserts a single thematic chapter into the existing reader route',()=>{
    const nav=parse(read('publishing/navigation.yml'));
    const raw=read('publishing/navigation.yml');
    expect(raw.match(new RegExp(`route: ${route}$`,'gm'))).toHaveLength(2);
    expect(raw).toContain(`source: ${chapter}\n        route: ${route}`);
    expect(nav).toBeDefined();
    expect(read('00 Учебник/_index.md')).toContain('41b Отбор обучающих примеров');
    expect(read('00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md'))
      .toContain('**Дальше:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction');
    expect(read('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md'))
      .toContain('**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация');
  });

  it('keeps five previous and three CL programs, four original figures and attribution',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(8);
    expect(text.match(/^!\[\[/gm)).toHaveLength(4);
    expect(text).toContain('source_language: en');
    expect(text).toContain('2107.07075v2');
    expect(text).toContain('not relabelled here as a standard');
    expect(text).toContain('not a crop');
    expect(text).toContain('not a guarantee that any');
  });

  it('preserves source bytes and records licenses separately from paper versions',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/data-selection-2026-10-28/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(2);
    for(const row of rows){
      expect(registry.find((x:any)=>x.id===row.id)).toEqual(row);
      expect(row.license).toBe('Apache-2.0');
      expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect(createHash('sha256').update(readFileSync(resolve(root,row.asset))).digest('hex')).toBe(row.sha256);
      expect(readFileSync(resolve(root,row.public_license_copy))).toEqual(readFileSync(resolve(root,row.license_evidence)));
    }
  });
});
