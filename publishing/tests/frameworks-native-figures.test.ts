import {createHash} from 'node:crypto';
import {readFileSync,readdirSync} from 'node:fs';
import {resolve,basename} from 'node:path';
import {describe,expect,it} from 'vitest';
import {parse} from 'yaml';
import {withoutTensorStorageAddition} from './helpers/tensor-storage-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
const report=JSON.parse(read('docs/audit/2026-10-27-frameworks-native-figures.json').toString());
const chapter='00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md';

describe('Frameworks figure inventory and tensor-storage integration',()=>{
  it('preserves and accounts for all 21 actual chapter objects, in order',()=>{
    expect(hash(read(report.source))).toBe(report.source_sha256);
    expect(hash(read(report.source_code))).toBe(report.source_code_sha256);
    const source=read(report.source).toString();
    const images=Array.from(source.matchAll(/!\[\[(Assets\/Sources\/Harvard ML Systems\/vol1\/frameworks\/[^\]|]+)(?:\|[^\]]*)?\]\]/g),m=>m[1]);
    expect(images).toHaveLength(21);
    expect(report.figures.map((r:{asset:string})=>r.asset)).toEqual(images);
    expect(readdirSync(resolve(root,'Assets/Sources/Harvard ML Systems/vol1/frameworks')).sort())
      .toEqual(images.map(path=>basename(path)).sort());
    const sourceLines=source.split('\n');
    const qmd=read(report.source_code).toString().split('\n');
    for(const [i,row] of report.figures.entries()) {
      expect(row.number).toBe(i+1);
      expect(hash(read(row.asset))).toBe(row.sha256);
      expect(sourceLines[row.source_line-1]).toContain(row.asset);
      expect(qmd[row.qmd_line-1]).toBe(row.source_caption_or_alt);
    }
  });

  it('does not confuse published figures, native files, aliases or held candidates',()=>{
    const inventory=JSON.parse(read('05 Источники/Originals/frameworks-2026-10-27/native-directory.json').toString());
    expect(inventory.tree_truncated).toBe(false);
    expect(inventory.files).toHaveLength(46);
    expect(report.native_directory.files.map(({decision,finding,published_figure,...r}:any)=>r)).toEqual(inventory.files);
    expect(new Set(inventory.files.map((r:{sha256:string})=>r.sha256)).size).toBe(43);
    expect(new Set([...inventory.files,...report.figures].map(r=>r.sha256)).size).toBe(56);
    expect(report.native_directory.files.filter((r:any)=>r.published_figure)).toHaveLength(8);
    expect(report.figures.filter((r:any)=>r.decision.startsWith('candidate')).map((r:any)=>r.number)).toEqual([]);
    expect(report.figures.filter((r:any)=>r.teaching_asset).map((r:any)=>r.number)).toEqual([13,17]);
    expect(report.native_directory.files.filter((r:any)=>r.decision.startsWith('candidate'))).toHaveLength(0);
    const notSelected=report.native_directory.files.filter((r:any)=>r.decision==='reference-only-not-selected');
    expect(notSelected).toHaveLength(2);
    for(const row of notSelected) {
      expect(row.published_figure).toBeNull();
      expect(row.finding).toContain('redistribution permission remains unverified');
      expect(row.finding).toContain('Do not add a teaching copy');
      expect(row.finding).toContain('frameworks-native-figures.md#fl-and-transfer-disposition');
    }
    expect(notSelected.some((r:any)=>r.finding.includes('Full 1118-line Edge/FL'))).toBe(true);
    expect(notSelected.some((r:any)=>r.finding.includes('Full 893-line BERT'))).toBe(true);
    expect(report.figures[16].finding).toContain('32 logical coordinates');
  });

  it('adds the worked storage explanation without rewriting the accepted 749-line chapter',()=>{
    const text=read(chapter).toString();
    expect(hash(withoutTensorStorageAddition(text)))
      .toBe('f8b39f3482e80bc07ff4cafcdddd287f7fe89cb31d00c1f9d0db0d125dd6a536');
    expect(text.match(/^```python$/gm)).toHaveLength(2);
    for(const mechanism of ['layout_f.strides == (4, 8)', 'np.shares_memory(layout_c, layout_t)',
      'layout_slice.nbytes == 16', 'span == layout_c.nbytes == 24',
      'layout_copy[0, 0] == layout_f[0, 0] == 1']) expect(text).toContain(mechanism);
    expect(text).toContain('not the result of transposing');
    expect(text).toContain('already reports **byte** strides');
  });

  it('keeps the published matrix summary compact and points to the detailed evidence',()=>{
    const matrix=read('05 Источники/Source maps/Harvard ML Systems — complete transfer matrix.md').toString();
    for(const [id,evidence] of [['H-V1-07','2026-10-27-frameworks-section-map.md'],
      ['H-F1-07','2026-10-27-frameworks-native-figures.json']]) {
      const row=matrix.split('\n').find(line=>line.startsWith(`| ${id} |`));
      expect(row,id).toBeDefined();
      const cells=row!.split('|').slice(1,-1).map(cell=>cell.trim());
      expect(cells).toHaveLength(8);
      // The full audit belongs in its report, not a multi-screen status cell.
      expect(cells[7].length,id).toBeLessThan(300);
      expect(cells[7]).toContain(evidence);
    }
  });

  it('retains the complete original SVG, source revision and adjacent license evidence',()=>{
    const [entry]=JSON.parse(read('00 Учебник/Assets/Figures/curated/frameworks-native-2026-10-27/assets.json').toString());
    expect(parse(read('05 Источники/asset-registry.yml').toString()).assets
      .find((r:{id:string})=>r.id===entry.id)).toEqual(entry);
    expect(read(entry.asset)).toEqual(read(entry.source_asset));
    for(const [path,sha] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]]) expect(hash(read(path))).toBe(sha);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.crop_box_px).toBeUndefined();
  });
});
