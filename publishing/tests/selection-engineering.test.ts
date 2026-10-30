import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe, expect, it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const chapter='00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества.md';
const edge='41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных — стоимость, шарды и проверка качества';

describe('selection engineering reader',()=>{
  it('historical explanation preservation: inserts a navigable chapter without changing accepted neighbouring explanations',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/pretraining\/selection-engineering-evaluation$/gm)).toHaveLength(2);
    expect(nav).toContain('source: '+chapter);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain('41i Отбор данных');
    const previous=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей.md');
    expect(previous).toContain(edge);
    expect(hash(previous.replace(edge,'42 Next-token prediction|Целевая функция предобучения')))
      .toBe('a70bb55df66be92c41dfec18850b0b1b3169103136149c78972a8ef385755d63');
    const next=readBeforeReaderRouteNavigation('00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md');
    expect(next).toContain(edge);
    expect(hash(next.replace(edge,'41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding')))
      .toBe('36663ad6d1759134b00f1aebf4d8e9d7538513109e1749b64f5bb3a37fcd13af');
  });

  it('preserves worked accounting, distributed selection, echoing and evaluation explanations',()=>{
    const text=read(chapter);
    expect(text).toContain('source_language: en');
    expect(text.match(/^```python$/gm)).toHaveLength(9);
    expect(text.match(/^!\[\[/gm)).toHaveLength(5);
    for(const anchor of ['selection-method-choice','selection-time-budget','selection-lifecycle-cost',
      'selection-storage-access','distributed-selection','data-echoing',
      'selection-quality-measurement','selection-cross-layer-effects'])
      expect(text).toContain('id="'+anchor+'"');
    for(const qualification of ['not a hardware benchmark','fixed per-record scores',
      'held-out data','not the marginal value','does not halve',
      'not a promised','never repays','strictly less than 60'])
      expect(text.replace(/\s+/g,' ')).toContain(qualification);
    expect(text).not.toMatch(/```(?:mermaid|svg)/);
  });

  it('adds method feasibility without rewriting the accepted accounting explanation',()=>{
    const text=read(chapter);
    const accepted=beforeReaderRouteNavigation(text, chapter)
      .replace(/<a id="selection-artifact-flow"><\/a>\n[\s\S]*?(?=Removing half the corpus at fixed epochs can reduce)/,'')
      .replace(/<a id="selection-icr-proxy"><\/a>\n[\s\S]*?(?=### Coverage can disagree with average quality)/,'')
      .replace(/<a id="selection-method-choice"><\/a>\n[\s\S]*?(?=<a id="selection-time-budget">)/,'');
    expect(hash(accepted)).toBe('b0e6e4621d64b1583c7522a3bb5cdc51b6f4ddaed461c7ebe1df0ad188e2feaf');
    for(const preserved of ['Access to oracle, unlabeled pool, retraining infrastructure',
      'Some labeled data, unlabeled data from same distribution',
      'Generative model or simulator, domain gap mitigation'])
      expect(text).toContain(preserved);
    for(const caveat of ['fails the rare-group requirement','does not establish privacy',
      'not observed advantages','Increasing a probability distribution\'s entropy alone'])
      expect(text.replace(/\s+/g,' ')).toContain(caveat);
    const guide=text.split('<a id="selection-time-budget">')[0];
    expect(guide).toContain('08 Reasoning distillation|distillation chapter');
    expect(guide).toContain('53a Синтетические изображения и перенос из симуляции');
  });

  it('binds complete original web figures to their actual distinct license',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/selection-engineering-2026-10-28/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(4);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.used_in).toEqual([chapter]);
      expect(row.license).toBe('CC BY-NC-ND 4.0');
      const original=readFileSync(resolve(root,row.source_asset));
      const asset=readFileSync(resolve(root,row.asset));
      expect(asset.equals(original)).toBe(true);
      expect(hash(asset)).toBe(row.sha256);
      expect(hash(readFileSync(resolve(root,row.source_page_snapshot)))).toBe(row.source_page_sha256);
      expect(read(row.license_evidence)).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    }
    expect(read('site/public/licenses/harvard-data-selection-2bd97c5-LICENSE.txt'))
      .toBe(read('05 Источники/Originals/nn-architectures-2026-10-26/LICENSE-2bd97c5.md'));
  });

  it('records a decision for every pinned native file without counting covers as new reader figures',()=>{
    const packet='05 Источники/Originals/data-selection-reconciliation-2026-10-28/';
    const inventory=JSON.parse(read(packet+'native-figure-decisions.json'));
    const listings=['svg','png','webp'].flatMap(kind=>JSON.parse(read(packet+kind+'-api.json')));
    expect(inventory.files).toHaveLength(14);
    expect(inventory.files.map((r:any)=>r.source_path).sort())
      .toEqual(listings.map((r:any)=>r.path).sort());
    expect(inventory.total_bytes).toBe(6684684);
    for(const row of inventory.files){
      const bytes=readFileSync(resolve(root,row.local));
      expect(bytes.length).toBe(row.bytes);
      expect(hash(bytes)).toBe(row.sha256);
      const blob=createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
      expect(blob).toBe(row.git_blob_sha);
      expect(listings.find((r:any)=>r.path===row.source_path)?.sha).toBe(blob);
      expect(['link-only','omit-new-reader-placement']).toContain(row.decision);
      expect(row.rationale.length).toBeGreaterThan(50);
    }
    expect(inventory.files.find((r:any)=>r.name==='running_out_of_data.png').decision).toBe('link-only');
  });
});
