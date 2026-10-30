import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe, expect, it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(data:string|Buffer)=>createHash('sha256').update(data).digest('hex');
const dir='05 Источники/Originals/consistency-training-2026-10-29';
const robust='00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем.md';
const semi='00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md';
const atConsistencyRevision=(p:string)=>beforeTopicConsolidation(read(p),p,root,
  '2026-10-29-consistency-training-additions.json');

describe('AT, VAT and UDA full explanations and source figures',()=>{
  it('restores the exact previous explanations and rejects an unknown edit',()=>{
    const journal=JSON.parse(read('docs/audit/2026-10-29-consistency-training-additions.json'));
    expect(journal.files).toHaveLength(3);
    for(const row of journal.files){
      const text=atConsistencyRevision(row.source);
      expect(sha(text)).toBe(row.after_sha256);
      const before=beforeTopicConsolidation(read(row.source),row.source,root,
        '2026-10-29-corpus-releases-additions.json');
      expect(sha(before)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(text+'\nunknown change\n',row.source,root,
        '2026-10-29-corpus-releases-additions.json')).toThrow();
    }
  });
  it('keeps whole reader programs, derivations and concrete method distinctions',()=>{
    const a=atConsistencyRevision(robust), b=atConsistencyRevision(semi);
    expect(a.match(/^```python$/gm)).toHaveLength(3);
    expect(b.match(/^```python$/gm)).toHaveLength(10);
    expect(a.match(/^!\[\[/gm)).toHaveLength(9);
    expect(b.match(/^!\[\[/gm)).toHaveLength(6);
    for(const id of ['embedding-adversarial-training','freelb','smart'])
      expect(a).toContain('id="'+id+'"');
    for(const id of ['virtual-adversarial-training','uda','training-signal-annealing'])
      expect(b).toContain('id="'+id+'"');
    expect(a).toContain('0.8036');
    expect(a).toContain('0.7892');
    expect(a).toContain('not a one-pass clean update at zero extra');
    expect(b).toContain('Hessian');
    expect(b).toContain('1.151293');
    expect(b).toContain('unsharpened');
    expect(b).toContain('correct class');
    expect(b).toContain('in-domain adaptation and UDA');
  });
  it('pins every downloaded source and the original implementation semantics',()=>{
    const manifest=JSON.parse(read(dir+'/manifest.json'));
    expect(manifest.sources).toHaveLength(16);
    for(const row of manifest.sources){
      const raw=readFileSync(resolve(root,dir,row.file));
      expect(sha(raw)).toBe(row.sha256);
      expect(raw.length).toBe(row.bytes);
      expect(row.readScope).toBeTruthy();
    }
    const code=read(dir+'/uda-text.py');
    for(const text of ['correct_label_probs, tsa_threshold',
      'tf.maximum(tf.reduce_sum(loss_mask), 1)',
      'tf.reduce_mean(per_example_kl_loss)',
      'largest_prob, FLAGS.uda_confidence_thresh'])
      expect(code).toContain(text);
    expect(read(dir+'/tf-adversarial_losses.py')).toContain('length - 1');
    expect(manifest.openComparisons.join(' ')).toContain('MixText');
    expect(manifest.openComparisons.join(' ')).toContain('Noisy Student');
  });
  it('binds complete native figures and source-specific permissions without redrawing',()=>{
    const rows=JSON.parse(read('00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/assets.json'));
    expect(rows).toHaveLength(3);
    for(const row of rows){
      expect(sha(readFileSync(resolve(root,row.asset)))).toBe(row.sha256);
      expect(sha(readFileSync(resolve(root,row.source_asset)))).toBe(row.source_asset_sha256);
      expect(sha(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
      if(row.asset.endsWith('.gif')){
        expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
        expect(row.frames).toBe(14);
        expect(row.license).toBe('MIT');
      }else{
        expect(row.license).toBe('CC BY 4.0');
        expect(row.render_dpi).toBe(400);
        expect(row.modifications).toContain('full caption');
        expect(row.modifications).toContain('no redrawing');
      }
    }
  });
});

