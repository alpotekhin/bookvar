import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {existsSync,readFileSync,mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeConfidentLearningAddition} from '../tools/confident-learning-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const journal=JSON.parse(read('docs/audit/2026-10-29-confident-learning-addition.json'));
const text=read(journal.source);
const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/confident-learning-2026-10-29/assets.json'));

describe('Confident Learning: bounded integration, not whole-course acceptance',()=>{
  it('binds the reviewed addition and preserves every previous explanation byte',()=>{
    expect(sha(text)).toBe(journal.after_sha256);
    const before=beforeConfidentLearningAddition(text,journal.source);
    expect(sha(before)).toBe(journal.before_sha256);
    expect(existsSync(resolve(root,journal.review))).toBe(true);
    expect(before.match(/^```python$/gm)).toHaveLength(5);
    expect(text.match(/^```python$/gm)).toHaveLength(8);
    expect(before.match(/^!\[\[/gm)).toHaveLength(2);
    expect(text.match(/^!\[\[/gm)).toHaveLength(4);
    expect(()=>beforeConfidentLearningAddition(text.replace('Four records were','Five records were'),journal.source))
      .toThrow('Unreviewed change');
    expect(sha(beforeConfidentLearningAddition(text+'\nUnreviewed outside edit.\n',journal.source)))
      .not.toBe(journal.before_sha256);
  });

  it('records the paper/software, model/label and literal/estimated-count boundaries',()=>{
    for(const fragment of ['out-of-sample','class-conditional','human inspection',
      'Fractional entries','restricted argmax','global argmax','diagonal entry to at least one',
      'not a calibrated probability','rare C example','not a CIFAR/ImageNet'])
      expect(text.toLowerCase()).toContain(fragment.toLowerCase());
    expect(text).toContain('source_language: en');
    expect(text).toContain('750625747de1b26d8530954f51f0530bd0b51d3c');
    expect(text).toContain('JAIR License Version 1.0');
    expect(text).not.toContain('mermaid');
  });

  it('reproduces both complete original panels from the pinned PDF',()=>{
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    const temp=mkdtempSync(join(tmpdir(),'bookvar-cl-figures-'));
    try {
      for(const [i,row] of assets.entries()){
        expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
        expect(sha(readFileSync(resolve(root,row.asset)))).toBe(row.sha256);
        expect(sha(readFileSync(resolve(root,row.source_asset)))).toBe(row.source_asset_sha256);
        expect(sha(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
        expect(readFileSync(resolve(root,row.public_license_copy)))
          .toEqual(readFileSync(resolve(root,row.license_evidence)));
        const [x,y,x2,y2]=row.crop_box_px;
        const out=join(temp,`figure${i}`);
        execFileSync('pdftoppm',['-f',String(row.source_page),'-l',String(row.source_page),
          '-r',String(row.render_dpi),'-x',String(x),'-y',String(y),'-W',String(x2-x),
          '-H',String(y2-y),'-singlefile','-png',resolve(root,row.source_asset),out]);
        expect(readFileSync(out+'.png')).toEqual(readFileSync(resolve(root,row.asset)));
      }
    } finally { rmSync(temp,{recursive:true,force:true}); }
  });
});
