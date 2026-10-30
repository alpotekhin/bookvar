import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const map=JSON.parse(read('docs/audit/2026-10-28-data-selection-section-map.json'));

describe('Data Selection source map, not a full-unit completion claim',()=>{
  it('accounts for all 4741 pinned lines in contiguous checked spans',()=>{
    const source=read(map.source.path),lines=source.split('\n');
    expect(hash(source)).toBe(map.source.sha256);
    expect(map.source.line_count).toBe(4741);
    expect(map.sections).toHaveLength(31);
    let next=1;
    for(const row of map.sections){
      expect(row.start).toBe(next);
      expect(row.end).toBeGreaterThanOrEqual(row.start);
      expect(hash(lines.slice(row.start-1,row.end).join('\n'))).toBe(row.source_span_sha256);
      for(const id of row.targets) expect(map.targets[id]).toBeDefined();
      expect(row.notes.length).toBeGreaterThan(50);
      next=row.end+1;
    }
    expect(next).toBe(4742);
  });

  it('binds current reader bytes to real scoped audits without inventing new acceptance',()=>{
    expect(Object.keys(map.targets)).toHaveLength(19);
    for(const row of Object.values(map.targets) as any[]){
      expect(hash(beforeReaderRouteNavigation(read(row.path), row.path))).toBe(row.sha256);
      expect(existsSync(resolve(root,row.review))).toBe(true);
      if(row.additional_review) expect(existsSync(resolve(root,row.additional_review))).toBe(true);
    }
    expect(map.scope).toContain('not proof of pedagogical completeness');
  });

  it('closes all three recorded reader gaps while preserving figure dispositions',()=>{
    expect(map.status).toBe('section-map-complete-recorded-reader-gaps-closed');
    expect(Object.keys(map.open_items)).toEqual([]);
    expect(Object.keys(map.closed_items)).toEqual(['label-errors','image-dedup','cold-embedding-ids']);
    for(const [id,target] of [['label-errors','scores'],['image-dedup','dedup'],['cold-embedding-ids','recsys']]){
      const closed=map.closed_items[id],destination=map.targets[target];
      expect(closed.status).toBe('reader-explanation-reviewed');
      expect(hash(read(destination.path))).toBe(closed.chapter_sha256);
      expect(destination.current_sha256).toBe(closed.chapter_sha256);
      expect(existsSync(resolve(root,closed.review))).toBe(true);
      expect(read(destination.path)).toContain(`id="${closed.anchor}"`);
      expect(closed.required.length).toBeGreaterThan(100);
    }
    const partial=map.sections.filter((r:any)=>r.status==='partial');
    expect(partial).toEqual([]);
    const figures=JSON.parse(read(map.source.generated_figures));
    expect(figures.figures).toHaveLength(14);
    expect(figures.figures.filter((r:any)=>r.acceptance==='pending-reader-qa')).toEqual([]);
    expect(map.source.native_figures).not.toBe(map.source.generated_figures);
  });
});
