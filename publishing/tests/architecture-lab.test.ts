import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {loadManifest} from '../adapter/manifest.js';

const root=resolve(import.meta.dirname,'../..');
const page='06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md';
const hash=(path:string)=>createHash('sha256').update(readFileSync(resolve(root,path))).digest('hex');
describe('Harvard architecture selection walkthrough',()=>{
  it('runs the original helper, four literal source scenarios and printed examples',()=>{
    const report=JSON.parse(execFileSync('python3',['publishing/tools/check_architecture_lab.py'],
      {cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(report.source_candidates).toBe(12);
    expect(report.independent_sweeps).toBe(300);
    expect(report.executed_printed_blocks).toBe(4);
    expect(report.verbatim_source_excerpts).toBe(1);
    expect(report.local_source_helpers).toHaveLength(8);
    expect(report.source_plot_source_sha256).toBe('7e097ec3195471135814668c6922200e881d85900343569f113c8ae5e40f5461');
    expect(report.tracks.iphone.topology.best_label).toBe('Efficient mobile transformer');
    expect(report.tracks.iphone.review.headroom_pick.architecture_id).toBe('mobile_cnn');
    expect(report.source_slide_arithmetic_findings.five_images_in_25ms_per_second).toBe(200);
    expect(report.scope).toContain('no Marimo');
    expect(report.wildlife_budget_checks.partial_memory_bytes).toBe(65045056);
    expect(report.wildlife_budget_checks.conditional_model_time_s).toBe('3/40');
    expect(report.wildlife_budget_checks.daily_energy_100_j).toBe('34617/40');
    expect(report.wildlife_budget_checks.capacity_180_days_wh).toBe('34617/800');
    expect(report.wildlife_budget_checks.hardware_measurements).toBe(false);
  });

  it('binds the public result to the current page, checker and source files',()=>{
    const report=JSON.parse(readFileSync(resolve(root,
      'site/public/examples/architecture-lab-source-checks-2026-10-26.json'),'utf8'));
    expect(report.page_sha256).toBe(hash(page));
    expect(report.checker_sha256).toBe(hash('publishing/tools/check_architecture_lab.py'));
    for(const [file,expected] of Object.entries(report.source_hashes))
      expect(hash('05 Источники/Originals/nn-architectures-2026-10-26/'+file)).toBe(expected);
  });

  it('places the worked case in reader navigation and both practice indexes',()=>{
    const pages=loadManifest(resolve(root,'publishing/navigation.yml')).sections
      .find(s=>s.id==='practice')!.pages;
    expect(pages.find(p=>p.source===page)?.route).toBe('practice/18a-architecture-selection');
    for(const file of ['06 Практика/_index.md','06 Практика/18 Harvard ML Systems — интерактивные design labs.md'])
      expect(readFileSync(resolve(root,file),'utf8')).toContain(page.replace(/\.md$/,''));
  });
});
