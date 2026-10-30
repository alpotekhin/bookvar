import {execFileSync} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe,it,expect} from 'vitest';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
describe('MIG source figures and worked progress example',()=>{
  it('preserves accepted content and verifies arithmetic, profiles and archive members',()=>{
    const report=JSON.parse(execFileSync('python3',
      ['publishing/tools/check_mig_visual_examples.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(report.accepted_2080_lines_preserved).toBe(true);
    expect(report.new_python_blocks_executed).toBe(1);
    expect(report.relative_progress_cases).toBe(300);
    expect(report.source_profile_rows).toBe(5);
    expect(report.original_figures).toBe(2);
    expect(report.verbatim_source_excerpts).toBe(1);
    expect(report.scope).toContain('no MIG driver, GPU or scheduler experiment');
  });
});
