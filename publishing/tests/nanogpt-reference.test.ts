import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(b:string|Buffer)=>createHash('sha256').update(b).digest('hex');
const practice='06 Практика/20 Собрать языковую модель с нуля.md';
const packet='05 Источники/Originals/nanogpt-2026-10-30';

describe('worked upstream nanoGPT reference',()=>{
  it('preserves the complete preceding chapters, assignment, index and registry',()=>{
    const journal=JSON.parse(read('docs/audit/2026-10-30-nanogpt-additions.json'));
    expect(journal.files).toHaveLength(5);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-30-nanogpt-additions.json');
      expect(hash(current)).toBe(row.after_sha256);
      const previous=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-synthetic-recipes-additions.json');
      expect(hash(previous)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(read(row.source)+'\nunknown edit\n',row.source,root,
        '2026-10-29-synthetic-recipes-additions.json')).toThrow();
    }
    expect(hash(read('06 Практика/Contracts/stanford-cs336-a1.yml')))
      .toBe('39af049ba810cd63e6a51fd6ba16242b75c3fdb63deec0ffd0eed29d48fc6cbd');
  });
  it('keeps exact source excerpts, a full mechanism explanation and the original plot',()=>{
    const text=read(practice);
    expect(text).toContain('id="nanogpt-worked-reference"');
    expect(text).toContain('96 training targets');
    expect(text).toContain('110,392-byte');
    expect(text).toContain('does not fill');
    expect(text).toContain('6832-parameter');
    const snippets=[...text.matchAll(/^\x60{3}python title="Original nanoGPT excerpt[^"]*"\n(.*?)^\x60{3}/gms)];
    expect(snippets).toHaveLength(2);
    const trainer=read(packet+'/train.py');
    for(const match of snippets) expect(trainer).toContain(match[1].trimEnd()+'\n');
    expect(text).toContain('loss-original.png');
    expect(text).toContain('DCLM CORE');
    expect(text).not.toContain('invented ten-step');
  });
  it('pins all selected source files and a byte-identical, licensed author plot',()=>{
    const manifest=JSON.parse(read(packet+'/manifest.json'));
    expect(manifest.sources).toHaveLength(12);
    for(const row of manifest.sources){
      const bytes=readFileSync(resolve(root,packet,row.file));
      expect(hash(bytes)).toBe(row.sha256);
      expect(bytes.length).toBe(row.bytes);
      expect(row.readScope).toBeTruthy();
    }
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/nanogpt-2026-10-30/assets.json'));
    expect(rows).toHaveLength(1);
    const row=rows[0],png=readFileSync(resolve(root,row.asset));
    expect(png.equals(readFileSync(resolve(root,row.source_asset)))).toBe(true);
    expect(hash(png)).toBe(row.sha256);
    expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([1206,614]);
    expect(hash(read(row.public_license_copy))).toBe(row.license_evidence_sha256);
  });
  it('executes original preparation, training, resume and sampling on CPU',()=>{
    const result=JSON.parse(execFileSync('uv',[
      'run','--offline','--with','torch==2.14.1','--with','numpy==2.3.5',
      '--with','requests==2.32.5','--with','tiktoken==0.12.0',
      'python','06 Практика/Code/nanogpt_reference_lab.py'
    ],{cwd:root,encoding:'utf8',timeout:60000,maxBuffer:4*1024*1024,
      env:{...process.env,LC_ALL:'C',LANG:'C'}}));
    expect(result.status).toBe('passed');
    expect(result.model_checks.math_backend_forced).toBe(true);
    expect(result.model_checks.upstream_flash_attribute).toBe(true);
    expect(result.model_checks.generation_input_lengths).toEqual([6,7,8,8,8]);
    expect(result.model_checks.max_accumulation_gradient_error).toBeLessThan(1e-12);
    expect(result.native_execution.model_parameters).toBe(6784);
    expect(result.native_execution.optimizer_tokens_per_update).toBe(32);
    expect(result.native_execution.checkpoint_optimizer_steps).toBe(2);
    expect(result.native_execution.non_identical_resume_max_weight_error).toBeGreaterThan(0);
    expect(result.native_execution.sampled_token_ids).toEqual([2,1,1,1,1]);
  },65000);
});
