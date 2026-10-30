import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe,it,expect} from 'vitest';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');

describe('Feature-store source integration',()=>{
  it('preserves accepted prose, verifies five excerpts and executes partition examples',()=>{
    const report=JSON.parse(execFileSync('python3',
      ['publishing/tools/check_feature_store_examples.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LANG:'C',LC_ALL:'C'}}));
    expect(report.accepted_chapter_preserved).toBe(true);
    expect(report.executed_new_python_blocks).toBe(2);
    expect(report.retained_bytes).toBe(87_600_000_000_000_000);
    expect(report.partition_cases).toBe(300);
    expect(report.source_excerpts).toBe(5);
    expect(report.unchanged_source_figures).toBe(3);
    // SQL execution is a separate explicit DuckDB check, not implied by this test.
    expect(report.sql_status).toContain('not-run');
  });
  it('retains the Feast availability caveat and both distinct source licenses',async()=>{
    const chapter=await readFile(resolve(root,
      '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'),'utf8');
    expect(chapter).toContain('filter_by_created_timestamp=False');
    expect(chapter).toContain('filter_by_created_timestamp=True');
    expect(chapter).toContain('Feature views without a creation-timestamp column are');
    expect(chapter).toContain('CC BY-NC-SA 4.0');
    expect(chapter).toContain('Apache-2.0');
    expect(chapter).not.toContain('feature-store-architecture.svg');
  });
});
