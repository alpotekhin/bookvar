import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,it,expect} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const hash=(path:string)=>createHash('sha256').update(readFileSync(resolve(root,path))).digest('hex');
describe('Harvard operations arithmetic integrated into chapter and practice',()=>{
  it('executes the original pure functions and every new printed Python block',()=>{
    const report=JSON.parse(execFileSync('python3',['publishing/tools/check_ops_scale_examples.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(report.accepted_2240_chapter_lines_preserved).toBe(true);
    expect(report.accepted_645_practice_lines_preserved).toBe(true);
    expect(report.source_helpers_executed).toHaveLength(8);
    expect(report.independent_accounting_cases).toBe(200);
    expect(report.new_python_blocks_executed).toBe(5);
    expect(report.verbatim_source_excerpts).toBe(3);
    expect(report.availability.ok).toBe(false);
    expect(report.source_accepts_canary_past_deadline.actual).toBe('balanced');
    expect(report.incidents.typed_runbook.lost_work).toBe(41600);
    expect(report.policies.balanced.feasible).toBe(true);
    expect(report.policies.conservative.cost_ok).toBe(false);
    expect(report.source_track_scenarios.oura_ring.actual).toBe('aggressive_exposed');
    expect(report.source_track_scenarios.robotaxi.required_hours).toBe(50);
    expect(report.scope).toContain('no Marimo UI');
  });

  it('binds the public CPU result to the current prose, sources and checker',()=>{
    const report=JSON.parse(readFileSync(resolve(root,
      'site/public/examples/ops-scale-source-checks-2026-10-26.json'),'utf8'));
    expect(report.chapter_sha256).toBe(hash('00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'));
    expect(report.practice_sha256).toBe(hash('06 Практика/17 Развернуть наблюдаемый ML сервис.md'));
    expect(report.checker_sha256).toBe(hash('publishing/tools/check_ops_scale_examples.py'));
    for (const [path,expected] of Object.entries(report.source_hashes)) expect(hash(path)).toBe(expected);
  });
});
