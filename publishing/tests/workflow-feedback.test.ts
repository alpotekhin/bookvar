import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe,it,expect} from 'vitest';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const read=(p:string)=>readFile(join(root,p),'utf8');
const sha=(b:string|Buffer)=>createHash('sha256').update(b).digest('hex');
const chapter='00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md';

describe('Harvard workflow feedback extension',()=>{
  it('preserves the accepted chapter outside two additions, dates and the HTTP-completion clarification',async()=>{
    const old=beforeReaderRouteNavigation(await read(chapter), chapter).replace(/<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->[\s\S]*?<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->\n\n/,'')
      .replace(/<!-- BEGIN REVIEW: model export 2026-10-27 -->[\s\S]*?<!-- END REVIEW: model export 2026-10-27 -->\n\n/,'')
      .replace('last_updated: 2026-10-27','last_updated: 2026-10-26')
      .replace('last_verified: 2026-10-27','last_verified: 2026-10-26')
      .replace(/<!-- BEGIN REVIEW: workflow (iteration|feedback) 2026-10-26 -->[\s\S]*?<!-- END REVIEW: workflow \1 2026-10-26 -->\n\n/g,'')
      .replace('last_updated: 2026-10-26','last_updated: 2026-10-20')
      .replace('last_verified: 2026-10-26','last_verified: 2026-10-20')
      .replace('HTTP 200 сам по себе не подтверждает правильность классификации.\nДля потокового ответа он также не доказывает, что генерация завершилась:\nэто проверяется по терминальному событию [[02 Areas/ML & DL/06 Практика/14a HTTP streaming и жизненный цикл запроса|протокола]].\nЕсли часть обращений отправляется человеку, измеряют',
        'HTTP 200 подтверждает завершение протокольного пути, но не правильность\nклассификации. Если часть обращений отправляется человеку, измеряют');
    expect(sha(old)).toBe('498a80237b35db4bf8376d2bc0dade25d6065f955a71958ce53d4ec9c515115e');
  });
  it('executes all three visible new fences and 600 independent arithmetic cases',()=>{
    const result=execFileSync('python3',
      ['publishing/tools/check_workflow_feedback_examples.py'],{cwd:root,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}});
    const report=JSON.parse(result);
    expect(report.executed_new_blocks).toBe(3);
    expect(report.independent_budget_cases).toBe(300);
    expect(report.independent_cohort_cases).toBe(300);
  });
  it('preserves one original SVG and distinct source/license evidence',async()=>{
    const [asset]=JSON.parse(await read('00 Учебник/Assets/Figures/curated/workflow-feedback-2026-10-26/assets.json'));
    expect(asset.license).toBe('CC BY-NC-SA 4.0');
    expect(asset.source_viewbox).toEqual([0,0,72.63,99.9]);
    for(const[path,hash]of[[asset.asset,asset.sha256],[asset.source_asset,asset.source_asset_sha256],
      [asset.source_page_snapshot,asset.source_page_sha256],[asset.license_evidence,asset.license_evidence_sha256]])
      expect(sha(await readFile(join(root,path)))).toBe(hash);
    expect(await readFile(join(root,asset.asset))).toEqual(await readFile(join(root,asset.source_asset)));
    expect(await read(chapter)).not.toContain('iteration-curves.svg');
  });
});
