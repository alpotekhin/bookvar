import {execFileSync} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe,it,expect} from 'vitest';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
describe('FinOps teaching examples',()=>{
  it('preserves the accepted chapter, executes examples and checks source identity',()=>{
    const report=JSON.parse(execFileSync('python3',
      ['publishing/tools/check_finops_examples.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(report.accepted_1656_lines_preserved).toBe(true);
    expect(report.new_python_blocks_executed).toBe(4);
    expect(report.generated_accounting_cases).toBe(900);
    expect(report.verbatim_source_excerpts).toBe(2);
    expect(report.source_figure_cells_verified).toBe(1);
    expect(report.scope).toContain('no real prices, invoices or savings measured');
  });
});
