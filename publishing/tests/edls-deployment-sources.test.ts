import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const read = (p:string) => readFile(join(root,p),'utf8');
const hash = (b:string|Buffer) => createHash('sha256').update(b).digest('hex');
const archive = '05 Источники/Originals/edls-deployment-2026-10-26/';
const chapter = '06 Практика/17 Развернуть наблюдаемый ML сервис.md';
const files = {
  goodapp:'00_basics/non-smokers-repo/app/main.py',
  model:'00_basics/non-smokers-repo/app/model.py',
  baddocker:'00_basics/smokers-repo/Dockerfile.bad',
  gooddocker:'00_basics/non-smokers-repo/Dockerfile.good',
  trace:'train_model.py', metrics:'03_metrics/server.py',
  scrapes:'03_metrics/prom/prometheus.yml',
  proto:'04_microservices/protos/inference.proto',
  supervisor:'supervisord/supervisord.conf',
};

describe('EDLS deployment walkthrough source fidelity',()=>{
  it('retains nine complete source files without rewriting their comments or code',async()=>{
    const body=await read(chapter);
    for(const [name,path] of Object.entries(files)) {
      const quoted=body.split('<!-- BEGIN SOURCE: '+name+' -->')[1]
        .split('<!-- END SOURCE: '+name+' -->')[0].trim()
        .replace(/^```[^\n]*\n/,'').replace(/\n```$/,'');
      expect(quoted,name).toBe((await read(archive+'week07_application_deployment/'+path)).trimEnd());
    }
    expect(body).toContain('<summary>Original Bookvar assignment</summary>');
    expect(body).toContain('Проведите load test с фиксированным arrival process');
    expect(await read('site/public/licenses/edls-deployment-MIT.txt')).toBe(await read(archive+'LICENSE'));
  });

  it('binds every downloaded course file to the pinned repository Git blob',async()=>{
    const manifest=JSON.parse(await read(archive+'source-manifest.json'));
    expect(manifest.commit).toBe('e632aa89ca9e6638d52e1b686095e7442faffbb0');
    expect(manifest.files).toHaveLength(61);
    for(const row of manifest.files) {
      const bytes=await readFile(join(root,archive,row.path));
      expect(hash(bytes)).toBe(row.sha256);
      expect(createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex'))
        .toBe(row.git_blob_sha1);
    }
  });

  it('records executed source failures and instrumentation without claiming deployment',async()=>{
    const bytes=await readFile(join(root,'06 Практика/Results/edls-deployment-2026-10-26/source-checks.json'));
    expect(await readFile(join(root,'site/public/examples/edls-deployment-source-checks-2026-10-26.json'))).toEqual(bytes);
    const data=JSON.parse(bytes.toString());
    expect(data.original_pytest).toMatchObject({exit_code:1,passed:4,failed:1});
    expect(data.seed7_first_score).toBeCloseTo(0.00006476655296663248,16);
    expect(data.health_while_loading).toEqual({status:200,body:{status:'not ready'},predict_status:503});
    expect(data.source_metrics_deltas).toEqual({predictions_total:3,api_invocations_total:4});
    expect(data.gauge_after_error).toBe(0);
    expect(data.histogram.buckets).toEqual({'0.1':1,'0.3':2,'1.0':4,'+Inf':4});
    expect(data.histogram.slo_300ms_fraction).toBe(0.5);
    expect(data.scope).toContain('stubbed VGG');
    expect(data.scope).toContain('no Docker, gRPC');
    expect(data.checker_sha256).toBe(hash(await read('publishing/tools/check_edls_deployment_sources.py')));
    for(const [path,expected] of Object.entries(data.source_hashes))
      expect(hash(await readFile(join(root,path)))).toBe(expected);
  });

  it('preserves full original figures, source licenses and Prometheus NOTICE',async()=>{
    const assets=JSON.parse(await read('00 Учебник/Assets/Figures/curated/deployment-2026-10-26/assets.json'));
    expect(assets).toHaveLength(2);
    for(const asset of assets) {
      expect(asset.license).toBe('Apache-2.0');
      expect(await readFile(join(root,asset.asset))).toEqual(await readFile(join(root,asset.source_asset)));
      for(const [path,expected] of [[asset.asset,asset.sha256],[asset.source_page_snapshot,asset.source_page_sha256],
        [asset.license_evidence,asset.license_evidence_sha256]])
        expect(hash(await readFile(join(root,path)))).toBe(expected);
      expect(asset.crop_box_px).toBeUndefined();
    }
    expect(await read('site/public/licenses/prometheus-3576591-NOTICE.txt')).toBe(await read(archive+'prometheus/NOTICE'));
  });

  it('keeps the existing reader route and practice index',async()=>{
    const section=loadManifest(join(root,'publishing/navigation.yml')).sections.find(s=>s.id==='practice')!;
    expect(section.pages.find(p=>p.source===chapter)?.route).toBe('practice/17-razvernut-nablyudaemyy-ml-servis');
    expect(await read('06 Практика/_index.md')).toContain(chapter.replace(/\.md$/,''));
  });
});
