import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const map=JSON.parse(read('docs/audit/2026-10-28-training-section-map.json').toString());
const figures=JSON.parse(read('docs/audit/2026-10-28-training-figures.json').toString());

describe('Training source reconciliation evidence boundaries',()=>{
  it('partitions all actual H2/H3 source sections and binds live destination revisions',()=>{
    const source=read(map.source.path);
    expect(hash(source)).toBe(map.source.sha256);
    const lines=source.toString().trimEnd().split('\n');
    expect(lines).toHaveLength(6801);
    const headings:{line:number,title:string}[]=[];
    let fence=false;
    lines.forEach((line,i)=>{
      if(/^```/.test(line))fence=!fence;
      if(!fence&&/^#{2,3} /.test(line))headings.push({line:i+1,title:line.replace(/^#+ /,'').split(' {')[0]});
    });
    expect(headings).toHaveLength(32);
    expect(map.sections).toHaveLength(headings.length);
    map.sections.forEach((s:any,i:number)=>{
      expect(s.heading_line).toBe(headings[i].line);
      expect(s.heading).toBe(headings[i].title);
      expect(s.start).toBe(i===0?1:headings[i].line);
      expect(s.end).toBe(headings[i+1]?headings[i+1].line-1:lines.length);
      for(const id of s.targets)expect(map.targets[id]).toBeDefined();
      for(const id of s.findings)expect(map.source_findings[id]).toBeDefined();
    });
    for(const t of Object.values(map.targets) as any[]){
      expect(hash(beforeReaderRouteNavigation(read(t.path).toString(), t.path))).toBe(t.sha256);
      expect(read(t.review).length).toBeGreaterThan(100);
    }
    // Structural coverage is not explanatory acceptance. The energy comparison
    // is resolved by a fully read existing chapter, not a new definition stub.
    expect(map.status).toBe('source-chapter-thematic-review-complete-with-findings');
    expect(Object.keys(map.open_items)).toEqual([]);
    expect(Object.keys(map.targets)).toHaveLength(19);
    expect(map.resolved_items['energy-emissions'].destination).toBe('responsible');
    expect(map.sections.find((s:any)=>s.heading==='When to scale: The physical ceiling').status)
      .toBe('mapped-with-source-findings');
  });

  it('distinguishes pinned native paths from archived rendered media and checks every byte object',()=>{
    const inventory=JSON.parse(read(figures.native_inventory).toString());
    expect(inventory.tree_truncated).toBe(false);
    expect(inventory.files).toHaveLength(33);
    expect(figures.native).toHaveLength(33);
    expect(figures.rendered).toHaveLength(27);
    for(const [i,x] of inventory.files.entries()){
      const b=read('05 Источники/Originals/training-2026-10-28/native/'+x.relative);
      expect(b.length).toBe(x.bytes);
      expect(hash(b)).toBe(x.sha256);
      expect(createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`),b])).digest('hex'))
        .toBe(x.git_blob_sha1);
      expect(figures.native[i].sha256).toBe(x.sha256);
      expect(figures.native[i].decision.trim()).not.toBe('');
      expect(figures.native[i].status).toMatch(/^(reference-only|source-duplicate|prefer-primary|decorative)/);
    }
    for(const x of figures.rendered){
      expect(hash(read(x.path))).toBe(x.sha256);
      expect(x.source_url).toMatch(/^https:\/\/mlsysbook\.ai\/vol1\/training\//);
      expect(x.decision.trim()).not.toBe('');
      expect(x.status).toMatch(/^(reference-only|prefer-primary|decorative)/);
    }
    expect(new Set(figures.native.map((x:any)=>x.sha256)).size).toBe(32);
    expect(new Set([...figures.native,...figures.rendered].map((x:any)=>x.sha256)).size).toBe(50);
    expect(hash(read(figures.primary_replacement.asset))).toBe(figures.primary_replacement.sha256);
    expect(figures.primary_replacement.license).toBe('CC-BY-4.0');
  });
});
