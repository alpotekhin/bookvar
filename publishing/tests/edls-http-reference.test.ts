import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';
import { loadManifest } from '../adapter/manifest.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const read = (p: string) => readFile(join(root, p), 'utf8');
const sha = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
const chapter = '06 Практика/14a HTTP streaming и жизненный цикл запроса.md';
const parent = '06 Практика/14 Собрать минимальный inference engine.md';
const archive = '05 Источники/Originals/edls-http-2026-10-26';
const section = (s:string,name:string) => s.split('<!-- BEGIN SOURCE: '+name+' -->')[1]
  .split('<!-- END SOURCE: '+name+' -->')[0].trim();

describe('EDLS HTTP lifecycle continuation', () => {
  it('preserves the accepted parent outside its bounded continuation link', async () => {
    const prior = beforeTextCheckpointRuntime(await read(parent)).replace(/\n<!-- BEGIN REVIEW: HTTP followup 2026-10-26 -->[\s\S]*?<!-- END REVIEW: HTTP followup 2026-10-26 -->\n/, '');
    expect(sha(prior)).toBe('d8f167f2d531b6d893a3db2a244d9eb0b195918527965fcc79ac26a21d6aaee5');
  });

  it('wires a single linear continuation into the manifest and Markdown index', async () => {
    const pages = loadManifest(join(root,'publishing/navigation.yml')).sections.find(s=>s.id==='practice')!.pages;
    const at = pages.findIndex(p=>p.source===parent);
    expect(pages[at+1]).toEqual({source:chapter,route:'practice/14a-http-streaming-lifecycle'});
    expect(pages[at+2]).toEqual({source:'06 Практика/14b Планировщик запросов и KV-память.md',route:'practice/inference-scheduler'});
    expect(pages[at+3].source).toBe('06 Практика/15 Реализовать W8A8 и SmoothQuant.md');
    expect(await read('06 Практика/_index.md')).toContain(chapter.replace(/\.md$/,''));
  });

  it('retains the complete visible HTTP implementation and identical download', async () => {
    const body = await read(chapter);
    const fence = body.split('<!-- BEGIN SOLUTION: EDLS HTTP module -->')[1]
      .split('<!-- END SOLUTION: EDLS HTTP module -->')[0].trim();
    const module = await read('06 Практика/Code/edls_http_reference.py');
    expect(fence).toBe('```python\n'+module.trimEnd()+'\n```');
    expect(await read('site/public/examples/edls_http_reference.py')).toBe(module);
  });

  it('preserves source explanations and the original collector, with only indentation/blank-space normalization', async () => {
    const body = await read(chapter);
    const overview = await read(archive+'/vllm-architecture.md');
    expect(overview).toContain(section(body,'vLLM engine responsibilities'));
    const original = await read('05 Источники/Originals/edls-inference-2026-10-26/api_server.py');
    const method = ('    async def _response_collector'+original.split('    async def _response_collector')[1])
      .split('    async def get_result')[0].replace(/^    /gm,'').replace(/[ \t]+$/gm,'').trim();
    const quoted = section(body,'EDLS response collector').replace(/^```python\n/,'').replace(/\n```$/,'')
      .replace(/[ \t]+$/gm,'').trim();
    expect(quoted).toBe(method);
  });

  it('binds actual loopback contracts to current source, without claiming a performance benchmark', async () => {
    const dataPath = '06 Практика/Results/edls-http-2026-10-26/contract.json';
    const bytes = await readFile(join(root,dataPath));
    expect(sha(bytes)).toBe('1651afb21b01773a123064245a612cb64ab340ec4bf423b5a3708ca0d701b0a9');
    expect(await readFile(join(root,'site/public/examples/edls-http-contract-2026-10-26.json'))).toEqual(bytes);
    const data = JSON.parse(bytes.toString());
    for(const [name, hash] of Object.entries(data.source_hashes)) {
      const path = name.startsWith('check_') ? 'publishing/tools/'+name : '06 Практика/Code/'+name;
      expect(sha(await readFile(join(root,path)))).toBe(hash);
    }
    expect(data.scope).toContain('real loopback HTTP');
    expect(data.scope).toContain('not a benchmark');
    expect(data.records.stream.first_token_before_engine_terminal).toBe(true);
    expect(data.records.final_json.generated_tokens).toEqual([32,27,34]);
    expect(data.records.completion_after_client_deadline.at(-1)).toMatchObject({
      status:'cancelled',reason:'timeout',engine_reason:'length'});
    expect(data.records.fatal_loop.events.at(-1)).toMatchObject({status:'error',reason:'model loop failed'});
    expect(data.records.failed_start_served_ready).toBe(false);
    expect(data.records.timed_out_start_served_ready).toBe(false);
    for(const key of ['shutdown_cleanup','limit_cleanup','nonstream_disconnect_cleanup','response_start_failure_cleanup'])
      expect(Object.values(data.records[key])).toEqual([0,0,0,0,0,0]);
  });

  it('pins the unchanged historical source figure, overview and full Apache license', async () => {
    const prefix='00 Учебник/Assets/Figures/curated/http-lifecycle-2026-10-26/';
    const [asset] = JSON.parse(await read(prefix+'assets.json'));
    expect(asset.dimensions_px).toEqual([1874,1178]);
    expect(asset.license).toBe('Apache-2.0');
    expect(asset.modifications).toContain('not a current V1');
    for(const [path,hash] of [[asset.asset,asset.sha256],[asset.source_asset,asset.source_asset_sha256],
      [asset.source_page_snapshot,asset.source_page_sha256],[asset.license_evidence,asset.license_evidence_sha256]])
      expect(sha(await readFile(join(root,path)))).toBe(hash);
    expect(await read('site/public/licenses/vllm-b8b302cd-Apache-2.0.txt')).toBe(await read(asset.license_evidence));
  });
});
