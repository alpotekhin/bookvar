import {execFileSync} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe,it,expect} from 'vitest';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');

describe('Platform resource policies and isolation',()=>{
  it('preserves the accepted chapter and executes the printed quota example',()=>{
    const report=JSON.parse(execFileSync('python3',
      ['publishing/tools/check_platform_resource_examples.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(report.accepted_1312_lines_preserved).toBe(true);
    expect(report.new_python_blocks_executed).toBe(1);
    expect(report.enumerated_quota_cases).toBe(500);
    expect(report.verified_source_excerpts).toBe(8);
    expect(report.original_figures).toBe(3);
    expect(report.scope).toContain('no scheduler, GPU or live Kubernetes validation');
  });
});
