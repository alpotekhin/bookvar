import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {existsSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const digest=(value:Buffer)=>createHash('sha256').update(value).digest('hex');
const map=JSON.parse(read('docs/audit/2026-10-27-frameworks-section-map.json').toString());
const source=read(map.source.path);

// These are consistency checks for the editorial record, not a semantic
// completion test for the source chapter or all destination explanations.
describe('Frameworks source section record',()=>{
  it('partitions every source line and records every actual level-two/three heading',()=>{
    expect(digest(source)).toBe(map.source.sha256);
    const lines=source.toString().trimEnd().split('\n');
    expect(lines).toHaveLength(map.source.lines);
    let fenced=false;
    const headings:Array<{line:number;title:string}>=[];
    lines.forEach((line,index)=>{
      if(/^```/.test(line)) fenced=!fenced;
      if(!fenced&&/^#{2,3} /.test(line))
        headings.push({line:index+1,title:line.replace(/^#{2,3} /,'').replace(/\s*\{.*$/,'')});
    });
    expect(headings).toHaveLength(37);
    expect(map.sections.map((section:any)=>({line:section.heading_line,title:section.heading})))
      .toEqual(headings);
    expect(map.sections[0].start).toBe(1);
    for(let i=0;i<map.sections.length;i++) {
      const section=map.sections[i];
      expect(section.end).toBe((map.sections[i+1]?.start??map.source.lines+1)-1);
      expect(section.heading_line).toBeGreaterThanOrEqual(section.start);
      expect(section.heading_line).toBeLessThanOrEqual(section.end);
      expect(section.decision.length).toBeGreaterThan(50);
    }
  });

  it('binds all named destinations to current bytes and existing review evidence',()=>{
    for(const target of Object.values(map.targets) as any[]) {
      expect(digest(Buffer.from(beforeReaderRouteNavigation(read(target.path).toString(), target.path))),target.path).toBe(target.sha256);
      expect(existsSync(resolve(root,target.review)),target.review).toBe(true);
    }
    for(const section of map.sections) {
      expect(section.targets.length).toBeGreaterThan(0);
      for(const target of section.targets) expect(map.targets[target]).toBeDefined();
      for(const item of section.open??[]) expect(map.open_items[item]).toBeDefined();
      for(const item of section.resolved??[]) expect(map.resolved_items[item]).toBeDefined();
      for(const finding of section.findings??[]) expect(map.source_findings[finding]).toBeDefined();
    }
  });

  it('keeps missing comparisons distinct from reviewed mechanisms and source findings',()=>{
    expect(map.status).toBe('source-chapter-thematic-review-complete-with-findings');
    const jax=map.sections.find((row:any)=>row.heading==='JAX: The functional transformation engine');
    expect(jax.status).toBe('mapped-with-source-findings');
    expect(jax.resolved).toContain('jax-runtime');
    expect(map.open_items['jax-runtime']).toBeUndefined();
    expect(existsSync(resolve(root,map.resolved_items['jax-runtime'].review))).toBe(true);
    expect(map.resolved_items['jax-runtime'].explanation).toContain('Four logical CPU devices');
    expect(map.open_items['placement-figure']).toBeUndefined();
    expect(existsSync(resolve(root,map.resolved_items['placement-figure'].review))).toBe(true);
    expect(map.targets.distributed).toBeDefined();
    expect(Object.keys(map.open_items)).toEqual([]);
    expect(existsSync(resolve(root,map.resolved_items['edge-transfer-figures'].review))).toBe(true);
    expect(map.resolved_items['edge-transfer-figures'].explanation).toContain('reference-only');
    expect(map.resolved_items['edge-transfer-figures'].explanation).toContain('permission remains unverified');
    expect(map.open_items.platform).toBeUndefined();
    expect(existsSync(resolve(root,map.resolved_items.platform.review))).toBe(true);
    for(const heading of ['Framework Platform Analysis','TensorFlow: The graph-first production machine',
      'Runtime constraint map','Deployment Targets','Framework selection criteria']){
      const section=map.sections.find((row:any)=>row.heading===heading);
      expect(section.status).toBe('mapped-with-source-findings');
      expect(section.resolved).toContain('platform');
      expect(section.open??[]).not.toContain('platform');
    }
    expect(map.resolved_items.hooks.explanation).toContain('grad_output');
    expect(existsSync(resolve(root,map.resolved_items.hooks.review))).toBe(true);
    expect(map.source_findings.module).toContain('load_state_dict');
    expect(map.source_findings.training).toContain('not observed CUDA');
    expect(map.scope).toContain('English-translation');
  });
});
