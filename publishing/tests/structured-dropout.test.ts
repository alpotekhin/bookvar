import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p:string) => readFileSync(resolve(root,p));
const hash = (s:Buffer|string) => createHash('sha256').update(s).digest('hex');
const chapter = '00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md';
const ru = '00 Учебник/05 Attention и Transformer/03 Полный Transformer.md';
const en = 'en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md';
const part = (p:string, name:string) => read(p).toString()
  .split('<!-- BEGIN REVIEW: '+name+' 2026-10-29 -->\n')[1]
  .split('<!-- END REVIEW: '+name+' 2026-10-29 -->')[0];

describe('structured dropout and its Transformer owner', () => {
  it('keeps complete examples and aligns the same explanation in both locales', () => {
    expect(read(chapter).toString().match(/^\x60{3}python$/gm)).toHaveLength(11);
    expect(read(ru).toString().match(/^\x60{3}python$/gm)).toHaveLength(6);
    expect(part(ru,'Transformer dropout')).toBe(part(en,'Transformer dropout'));
    const code = (p:string) => [...read(p).toString().matchAll(/\x60{3}python\n(.*?)\x60{3}/gs)].map(x=>x[1]);
    expect(code(ru)).toEqual(code(en));
    expect(read(en).toString()).toContain('canonical_revision: "'+hash(read(ru))+'"');
    expect(read(en).toString()).toContain('correct token \x60sleeps\x60, loss');
    for(const anchor of ['dropconnect','stochastic-depth','transformer-dropout'])
      for(const p of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(read(p).toString()).toContain('#'+anchor+'|');
  });

  it('credits a complete original figure with evidence from the actual licensed PDF', () => {
    const [asset] = JSON.parse(read('00 Учебник/Assets/Figures/curated/structured-dropout-2026-10-29/assets.json').toString());
    const rows = parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(rows.filter((r:any)=>r.asset===asset.asset)).toEqual([asset]);
    expect(asset.dimensions_px).toEqual([1770,795]);
    expect(asset.crop_box_px).toEqual([340,890,2110,1685]);
    for(const [p,h] of [[asset.asset,asset.sha256],
      [asset.source_asset,asset.source_asset_sha256]]) expect(hash(read(p))).toBe(h);
    const page1 = execFileSync('pdftotext',['-f','1','-l','1',
      resolve(root,asset.license_evidence),'-'],{encoding:'utf8'});
    expect(page1).toContain('License: CC-BY 4.0');
    expect(read(chapter).toString()).toContain('original caption says');
    expect(read('site/public/licenses/timm-beabdc3a-Apache-2.0.txt'))
      .toEqual(read('05 Источники/Originals/structured-dropout-2026-10-29/timm-LICENSE'));
  });

  it('restores every preceding file exactly and preserves old examples and figures', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-structured-dropout-additions.json').toString());
    expect(journal.files).toHaveLength(6);
    for(const row of journal.files) {
      const live = read(row.source).toString();
      const current = beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-structured-dropout-additions.json');
      let old = current;
      expect(hash(current)).toBe(row.after_sha256);
      for(const c of [...row.changes].reverse()) {
        expect(old.split(c.new)).toHaveLength(2);
        old = old.replace(c.new,()=>c.old);
      }
      expect(hash(old)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-adaptive-clipping-additions.json')).toBe(old);
      if([chapter,ru,en].includes(row.source)) {
        const block = row.source===chapter?'structured dropout':'Transformer dropout';
        const without = current.replace(new RegExp('<!-- BEGIN REVIEW: '+block+
          ' 2026-10-29 -->[\\s\\S]*?<!-- END REVIEW: '+block+' 2026-10-29 -->'),'');
        expect([...without.matchAll(/\x60{3}python\n(.*?)\x60{3}/gs)].map(x=>x[1]))
          .toEqual([...old.matchAll(/\x60{3}python\n(.*?)\x60{3}/gs)].map(x=>x[1]));
        const oldImages = [...old.matchAll(/!\[\[[^\]]+\]\]/g)].map(x=>x[0]);
        for(const image of oldImages) expect(current).toContain(image);
      }
    }
  });
});
