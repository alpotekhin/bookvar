import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(value:string|Buffer)=>createHash('sha256').update(value).digest('hex');
const owner='00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md';
const packet='05 Источники/Originals/agent-reconciliation-2026-10-30/';

describe('source-faithful agent-note reconciliation',()=>{
  it('restores the whole preceding chapter and registry without accepting unknown edits',()=>{
    const journal=JSON.parse(read('docs/audit/2026-10-30-agent-reconciliation-additions.json'));
    expect(journal.files).toHaveLength(3);
    for(const row of journal.files){
      const accepted=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-30-agent-reconciliation-additions.json');
      expect(hash(accepted)).toBe(row.after_sha256);
      const previous=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-30-nanogpt-additions.json');
      expect(hash(previous)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(read(row.source)+'\nunknown edit\n',row.source,root,
        '2026-10-30-nanogpt-additions.json')).toThrow();
    }
  });

  it('pins source sections and retains the source-specific Toolformer illustration gap',()=>{
    const manifest=JSON.parse(read(packet+'manifest.json'));
    for(const row of [...manifest.reusedSources,...manifest.downloaded,...manifest.derived]){
      const bytes=readFileSync(resolve(root,row.path));
      expect(hash(bytes)).toBe(row.sha256);
      expect(bytes.length).toBe(row.bytes);
    }
    expect(manifest.downloaded).toHaveLength(1);
    expect(manifest.illustrationRightsGap.item).toBe('Toolformer Figure 2');
    expect(manifest.illustrationRightsGap.status).toBe('reuse-permission-unverified');
    expect(manifest.figures).toHaveLength(2);
    for(const figure of manifest.figures){
      expect(figure.source_url).toBe('https://arxiv.org/abs/2210.03629v3');
      expect(figure.license).toBe('CC BY 4.0');
    }
  });

  it('preserves the four immutable archive notes and the causal examples',()=>{
    const pins=[
      ['ReAct','db772fc7834f840df36b2386a72790ba3d6e7e142780d11956d2829fae81b883'],
      ['Tool Use','7fcc266613ab789abadfd800e3ecdb5d578fde1b48f29e2d96c64b889a566db7'],
      ['Toolformer','c22dab7ecf6adb342f4b95453045d2f808ab8e60c90beba1ab7f8f5f65006d88'],
      ['Tree of Thoughts','6ef781048ed0cc9a8856e3da0e7fa26d56c2288ea1f974483d3cdb88c8359626']
    ];
    for(const [name,digest] of pins)expect(hash(read('Concepts/Reasoning/'+name+'.md'))).toBe(digest);
    const text=read(owner);
    for(const value of ['1.3025','1.5474','idempotency_parameter_mismatch','run_episode',
      'Table 2 is a manually labeled analysis of 200 selected trajectories',
      'the mean','57%','3,000','25,000','10.3','60.2','57.8',
      'With five samples and counts','exactly-half','future continuation'])
      expect(text).toContain(value);
  });

  it('executes all five displayed programs, executor negative cases and exact original crops',()=>{
    const result=JSON.parse(execFileSync('uv',[
      'run','--offline','--with','numpy','--with','pillow',
      'python','publishing/tools/check_tool_use_examples.py'
    ],{cwd:root,encoding:'utf8',timeout:45000,maxBuffer:1024*1024,
      env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(result.code_blocks).toBe(5);
    expect(result.standalone_and_shared_runs).toBe(10);
    expect(result.original_figure_crops_verified).toBe(4);
    expect(result.json_examples_parsed).toBe(5);
    expect(result.llm_run).toBe(false);
    expect(result.agent_quality_benchmark).toBe(false);
  },50000);
});

