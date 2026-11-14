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
  it('archives the original OpenAI extracts without treating the code license as paper permission',()=>{
    const entries=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
    const rows=entries.filter((r:any)=>r.id==='curated-gpt1-figure1-0000000000c9');
    expect(rows).toHaveLength(1);
    const r=rows[0];
    expect(r.used_in).toEqual([]);
    expect(r.publication_status).toBe('retired');
    expect(r.source_page).toBe(4);
    expect(r.render_dpi).toBe(360);
    expect(r.crop_box_px).toEqual([500,300,2550,1150]);
    expect(r.metadata_status).toBe('needs-confirmation');
    expect(r.rights_status).toBe('paper-figure-license-unverified');
    expect(hash(read(r.asset))).toBe(r.sha256);
    expect(hash(read(r.source_asset))).toBe(r.source_asset_sha256);
    const png=read(r.asset);
    expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([2050,850]);
    for(const name of ['gpt1-figure1-block.png', 'gpt1-figure2-transfer.png', 'gpt1-figure2-zero-shot.png']) {
      const panel=entries.find((entry:any)=>entry.asset.endsWith('/'+name));
      expect(panel).toMatchObject({used_in:[],publication_status:'retired'});
      expect(hash(read(panel.asset))).toBe(panel.sha256);
      expect(hash(read(panel.source_asset))).toBe(panel.source_asset_sha256);
      expect(panel.rights_status).toBe('paper-figure-license-unverified');
    }
  });
  it('ties both editions to the licensed figures and original experiment pages',()=>{
    const english='en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md';
    const entries=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
    const figures=[
      {name:'bert-gpt-elmo-figure3.png',license:'CC BY 4.0',source:'https://aclanthology.org/N19-1423.pdf#page=13'},
      {name:'d2l-gpt-decoder.png',license:'CC BY-SA 4.0',source:'#fig-gpt-decoder-only'},
      {name:'cs224n-decoder-classifier.png',license:'Stanford CS224N course resource reuse notice',
        source:'cs224n-2023-lecture9-pretraining.pdf#page=41'},
    ];
    for(const {name,license,source} of figures){
      const entry=entries.find((r:any)=>r.asset.endsWith('/'+name));
      expect(entry).toMatchObject({license,metadata_status:'verified'});
      expect(hash(read(entry.asset))).toBe(entry.sha256);
      expect(hash(read(entry.source_asset))).toBe(entry.source_asset_sha256);
      for(const path of [chapter,english]){
        expect(entry.used_in).toContain(path);
        const note=read(path).toString();
        expect(note).toContain(entry.asset+'|');
        expect(note).toContain(source);
        expect(note).not.toMatch(/source-audit-25-31\/gpt1-figure[12]/);
        expect(note).toContain('https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4');
        expect(note.match(/language_understanding_paper\.pdf#page=7/g)).toHaveLength(2);
      }
    }
    const acl=entries.find((r:any)=>r.asset.endsWith('/bert-gpt-elmo-figure3.png'));
    expect(hash(read(acl.license_evidence))).toBe(acl.license_evidence_sha256);
    const crop=read(acl.asset);
    expect([crop.readUInt32BE(16),crop.readUInt32BE(20)]).toEqual([2280,548]);
    const stanford=entries.find((r:any)=>r.asset.endsWith('/cs224n-decoder-classifier.png'));
    expect(stanford).toMatchObject({source_page:41,render_dpi:240,
      crop_box_px:[1835,399,2910,1319],rights_status:'source-permission-verified'});
    const panel=read(stanford.asset);
    expect([panel.readUInt32BE(16),panel.readUInt32BE(20)]).toEqual([1075,920]);
    for(const path of [chapter,english]){
      const note=read(path).toString();
      expect(note).toContain('cs224n-2023-lecture9-pretraining.pdf#page=44');
      expect(note).toContain('#gpt2-relation-prefix|');
      expect(note).toContain(stanford.license_evidence_url);
    }
    expect(YAML.parse(read(english).toString().split('---')[1]).canonical_revision).toBe(hash(read(chapter)));
  });
  it('links T5 to GPT-1 to GPT-2',()=>{
    const fm=YAML.parse(body.split('---')[1]);
    expect(fm.previous).toContain('07 T5');
    expect(fm.next).toContain('05 GPT-2 — zero-shot через язык');
    expect(body.slice(body.lastIndexOf('**Назад:**'))).toContain(fm.previous.slice(0,-2)+'|');
    expect(body.slice(body.lastIndexOf('**Назад:**'))).toContain(fm.next.slice(0,-2)+'|');
  });
});
