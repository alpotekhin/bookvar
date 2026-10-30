import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {checkSectionMap} from '../tools/check_nn_section_map.mjs';

const root=resolve(import.meta.dirname,'../..');
const map=JSON.parse(readFileSync(resolve(root,'docs/audit/2026-10-27-nn-section-map.json'),'utf8'));
describe('Network Architectures editorial decision integrity',()=>{
  it('binds every reviewed source range and destination to the current artifacts',()=>{
    const result=checkSectionMap(map,root);
    expect(result.errors).toEqual([]);
    expect(result.rows).toBe(37);
    expect(result.destinations).toBe(17);
    expect(result.source_lines).toBe(5256);
    expect(result.scope).toContain('not an automated semantic-completeness test');
  });
  it('detects missing source coverage instead of silently accepting fewer rows',()=>{
    const changed=structuredClone(map);
    changed.rows[1].start+=1;
    expect(checkSectionMap(changed,root).errors).toContain('Gap/overlap at principles');
  });
  it('requires re-review when a destination changes',()=>{
    const changed=structuredClone(map);
    changed.targets.practice.sha256='0'.repeat(64);
    expect(checkSectionMap(changed,root).errors).toContain('Stale reviewed destination: practice');
  });
});
