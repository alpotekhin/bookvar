import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const digest=(value:string|Buffer)=>createHash('sha256').update(value).digest('hex');
const journalName='2026-10-30-tot-reconciliation-additions.json';
const chapter='00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md';

describe('full ToT tasks within the canonical reasoning chapter',()=>{
  it('preserves the full prior chapters and registry, rejecting unrecorded changes',()=>{
    const journal=JSON.parse(read('docs/audit/'+journalName));
    expect(journal.files).toHaveLength(5);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,journalName);
      expect(digest(current)).toBe(row.after_sha256);
      const prior=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-30-agent-reconciliation-additions.json');
      expect(digest(prior)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(read(row.source)+'\nunknown edit\n',row.source,root,
        '2026-10-30-agent-reconciliation-additions.json')).toThrow();
    }
    expect(digest(read('Concepts/Reasoning/Tree of Thoughts.md')))
      .toBe('6ef781048ed0cc9a8856e3da0e7fa26d56c2288ea1f974483d3cdb88c8359626');
    for(const value of ['tot-creative-writing','tot-crossword-dfs','five plans → one plan',
      'eight proposal samples','not network requests','TM_S_','7.67','49%'])
      expect(read(chapter)).toContain(value);
  });

  it('pins downloaded code, source license and all three complete original figures',()=>{
    const manifest=JSON.parse(read('05 Источники/Originals/tot-reconciliation-2026-10-30/manifest.json'));
    expect(manifest.commit).toBe('8050e67d0e3a0fddc424d7fa5801538722a4c4cc');
    for(const row of [...manifest.reusedSources,...manifest.downloaded,...manifest.derived]){
      const bytes=readFileSync(resolve(root,row.path));
      expect(bytes.length).toBe(row.bytes);
      expect(digest(bytes)).toBe(row.sha256);
    }
    expect(manifest.figures).toHaveLength(3);
    for(const row of manifest.figures){
      expect(digest(readFileSync(resolve(root,row.asset)))).toBe(row.sha256);
      expect(row.license).toBe('CC BY 4.0');
      expect(row.source_url).toBe('https://arxiv.org/abs/2305.10601v2');
    }
  });

  it('replays native vote parsing, overwrite detection, pruning and exact sibling-state restoration',()=>{
    const report=JSON.parse(execFileSync('uv',['run','--offline','--with','pillow','python',
      'publishing/tools/check_tot_reconciliation_examples.py'],{
        cwd:root,encoding:'utf8',timeout:30000,env:{...process.env,LC_ALL:'C',LANG:'C'}
      }));
    expect(report.native_vote_cases).toBe(3);
    expect(report.native_logged_children).toBe(2);
    expect(report.native_expanded_states).toBe(2);
    expect(report.parent_state_restored).toBe(true);
    expect(report.child_cap_exit_gap_reproduced).toBe(true);
    expect(report.exact_original_figures).toBe(3);
    expect(report.llm_run).toBe(false);
    expect(report.crossword_benchmark).toBe(false);
  },35000);
});
