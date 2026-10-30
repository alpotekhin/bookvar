import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const hash=(value:string|Buffer)=>createHash('sha256').update(value).digest('hex');
const name='2026-10-30-bleu-reconciliation-additions.json';

describe('reference-based metrics inside the canonical evaluation chapter',()=>{
  it('restores every preceding chapter/index/registry byte and rejects unknown changes',()=>{
    const journal=JSON.parse(read('docs/audit/'+name));
    expect(journal.files).toHaveLength(4);
    for(const row of journal.files){
      const current=beforeTopicConsolidation(read(row.source),row.source,root,name);
      expect(hash(current)).toBe(row.after_sha256);
      const prior=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-30-tot-reconciliation-additions.json');
      expect(hash(prior)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(read(row.source)+'\nunknown edit\n',row.source,root,
        '2026-10-30-tot-reconciliation-additions.json')).toThrow();
    }
    expect(hash(read('Concepts/Evaluation/BLEU Score.md')))
      .toBe('d432707ef676024e1f9b8f639f9d1df3b2d327e1207fd8cf14cb40b2d9a67988');
  });

  it('pins the original code/image and license without claiming unavailable PDF downloads',()=>{
    const manifest=JSON.parse(read('05 Источники/Originals/bleu-reconciliation-2026-10-30/manifest.json'));
    expect(manifest.sacrebleu.commit).toBe('d4d9a26dd833fe23696e09ca2a70b2fc433f3d4b');
    for(const row of [...manifest.sacrebleu.files,...manifest.bertscore.files]){
      const bytes=readFileSync(resolve(root,row.path));
      expect(hash(bytes)).toBe(row.sha256);
      expect(createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex')).toBe(row.git_blob);
    }
    expect(manifest.network.localPDFs).toContain('Not downloaded');
    const figure=manifest.bertscore.figure;
    expect(figure.license).toBe('MIT');
    expect(readFileSync(resolve(root,figure.asset)))
      .toEqual(readFileSync(resolve(root,figure.source_asset)));
    expect(readFileSync(resolve(root,figure.license_evidence)))
      .toEqual(readFileSync(resolve(root,figure.public_license_copy)));
  });

  it('executes all twelve published programs and compares 7688 token cases to native scoring',()=>{
    const report=JSON.parse(execFileSync('uv',['run','--offline','--with','numpy','python',
      'publishing/tools/check_bleu_reconciliation_examples.py'],{
        cwd:root,encoding:'utf8',timeout:30000,env:{...process.env,LC_ALL:'C',LANG:'C'}
      }));
    expect(report.python_fences).toBe(12);
    expect(report.native_bleu_cases).toBe(7688);
    expect(report.chrf_two_order_score).toBeCloseTo(75.22123893805309,10);
    expect(report.original_figure_weighted_recall).toBeCloseTo(0.753398417835311,12);
    expect(report.reference_nonmonotonicity_verified).toBe(true);
    expect(report.figure_byte_identical).toBe(true);
    expect(report.installed_sacrebleu_package).toBe(false);
    expect(report.tokenizer_cli_run).toBe(false);
    expect(report.bert_encoder_run).toBe(false);
  },35000);
});
