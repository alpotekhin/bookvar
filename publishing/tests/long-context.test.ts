import {readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/long-context-figures.json'),'utf8'));
describe('long-context explanation and reading order',()=>{
  it('preserves all originals and adds complete chunking, CLA and RULER figures',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(12);
    for(const spec of specs)expect(embeds).toContain(spec.asset);
    for(const path of embeds){
      expect(existsSync(resolve(root,path))).toBe(true);
      expect(assets.find((a:any)=>a.asset===path)?.used_in).toContain(file);
    }
  });
  it('includes executable causal, chunk-cache, layer-sharing and indexer examples',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(6);
    for(const s of ['1/24','window_attention','h2_shared','h2_separate',
      'kind="stable"','13.25','1.375 GiB','2048','943.7184','пересчитать затронутый префикс',
      'разрешены четыре позиции, включая текущую','неполная группа здесь первая','SQuAD и HotpotQA',
      'recall 1','precision $2/3$','в §8'])expect(body).toContain(s);
    expect(body).not.toContain('## Как связаны рассмотренные методы');
  });
  it('uses the unmodified author MoBA diagram and distinguishes an incomplete tail from a completed block',()=>{
    const path='00 Учебник/Assets/Figures/curated/moba/moba-running-example.png';
    const license='site/public/licenses/moonshot-moba-b5d58363-MIT.txt';
    const bytes=readFileSync(resolve(root,path));
    const digest=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
    expect(digest(bytes)).toBe('34861ce26910fe40efea3b0abaf2a2dfbc756547dada22a1e84279f93d420e7f');
    expect([bytes.readUInt32BE(16),bytes.readUInt32BE(20)]).toEqual([2552,2328]);
    expect(assets.find((a:any)=>a.asset===path)).toMatchObject({
      license:'MIT',sha256:digest(bytes),license_evidence:license,
      license_evidence_sha256:digest(readFileSync(resolve(root,license))),
      source_image_url:'https://raw.githubusercontent.com/MoonshotAI/MoBA/b5d58363311d3ca946f1ec444182727c15e338b5/figures/running_example.png',
      used_in:[file],
    });
    expect(body).toContain(`![[${path}|`);
    expect(body).toContain('moba_visible(11, 4, [2, 3], 2) == list(range(4, 12))');
    expect(body).toContain('qsa_visible(11, 4, [.7, 1.2, .2], 2) == list(range(8))');
  });
  it('keeps stable anchors, three course units and the MLA→long-context→FFN sequence',()=>{
    expect(fm.source_unit_id).toHaveLength(3);
    expect(fm.previous).toContain('02 MLA');
    expect(fm.next).toContain('02 Mixture of Experts');
    for(const id of ['sliding-window-receptive-field','cross-layer-kv-sharing','dsa-content-adaptive-selection','block-routing-moba-qsa'])
      expect(body.match(new RegExp('id="'+id+'"','g'))).toHaveLength(1);
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Дальше:**');
  });
});
