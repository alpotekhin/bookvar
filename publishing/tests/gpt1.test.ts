import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const chapter='00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md';
const body=read(chapter).toString();
const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');

describe('GPT-1 chapter contracts',()=>{
  it('keeps the full two-stage example and unchanged original excerpt distinct',()=>{
    expect(body.match(/\x60\x60\x60python\n/g)).toHaveLength(2);
    const excerpt=body.match(/\x60\x60\x60python title="Original implementation — MIT"\n(.*?)\n\x60\x60\x60/s)![1];
    expect(read('05 Источники/Originals/gpt1-2026-10-22/train.py').toString()).toContain(excerpt);
    expect(read('site/public/licenses/openai-gpt1-MIT.txt')).toEqual(read('05 Источники/Originals/gpt1-2026-10-22/LICENSE'));
    for(const fact of ['norm_first=False','START, DELIM, EXTRACT = 12, 13, 14',
      'model.lm_logits(h[:, :-1])','per_candidate.mean()',
      'classification_loss + .5 * lm_loss','target_mask.sum(1)',
      'head.grad','scores.retain_grad()','token_loss.retain_grad()',
      '0.020833','0.718351','2.085067','как отдельный обучаемый параметр',
      '85.8%','74.7','75.0','2048','CoLA','SST-2','RACE','DPRD'])
      expect(body).toContain(fact);
    expect(body).not.toContain('\x60\x60\x60mermaid');
  });
  it('keeps the original complete figure, precise source proof and unresolved paper rights',()=>{
    const entries=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
    const rows=entries.filter((r:any)=>r.id==='curated-gpt1-figure1-0000000000c9');
    expect(rows).toHaveLength(1);
    const r=rows[0];
    expect(r.used_in).toContain(chapter);
    expect(r.source_page).toBe(4);
    expect(r.render_dpi).toBe(360);
    expect(r.crop_box_px).toEqual([500,300,2550,1150]);
    expect(r.metadata_status).toBe('needs-confirmation');
    expect(r.rights_status).toBe('paper-figure-license-unverified');
    expect(hash(read(r.asset))).toBe(r.sha256);
    expect(hash(read(r.source_asset))).toBe(r.source_asset_sha256);
    const png=read(r.asset);
    expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([2050,850]);
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(4);
    expect(embeds).toContain(r.asset);
    for(const name of ['gpt1-figure1-block.png', 'gpt1-figure2-transfer.png', 'gpt1-figure2-zero-shot.png']) {
      const panel=entries.find((entry:any)=>entry.asset.endsWith('/'+name));
      expect(panel.used_in).toContain(chapter);
      expect(embeds).toContain(panel.asset);
      expect(hash(read(panel.asset))).toBe(panel.sha256);
      expect(hash(read(panel.source_asset))).toBe(panel.source_asset_sha256);
      expect(panel.rights_status).toBe('paper-figure-license-unverified');
    }
  });
  it('preserves BERT to GPT-1 to GPT-2 navigation',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('03 BERT, RoBERTa и DeBERTa');
    expect(fm.next).toContain('05 GPT-2 — zero-shot через язык');
    expect(body.slice(body.lastIndexOf('**Назад:**'))).toContain(fm.previous.slice(0,-2)+'|');
    expect(body.slice(body.lastIndexOf('**Назад:**'))).toContain(fm.next.slice(0,-2)+'|');
  });
});
