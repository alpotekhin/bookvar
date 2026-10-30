import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const read = (p: string) => readFileSync(resolve(root,p));
const sha = (b: Buffer) => createHash('sha256').update(b).digest('hex');
const chapter = '00 Учебник/05 Attention и Transformer/03 Полный Transformer.md';
const body = read(chapter).toString();
const figures = JSON.parse(read('publishing/tools/full-transformer-figures.json').toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;

describe('full Transformer chapter contracts', () => {
  it('keeps the architecture introduction on the BERT/GPT reading route', () => {
    const intro = read('00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md').toString();
    const fm = YAML.parse(intro.split('---')[1]);
    expect(fm.previous).toContain('03 Полный Transformer');
    expect(fm.next).toContain('03 BERT, RoBERTa и DeBERTa');
    const footer = intro.slice(intro.lastIndexOf('**Назад:**'));
    expect(footer).toContain(fm.previous);
    expect(footer).toContain(fm.next);
    expect(footer).not.toContain('01 LLaMA как базовая архитектура');
  });
  it('retains the five course units and the manual architecture anchors', () => {
    const fm = YAML.parse(body.split('---')[1]);
    expect(fm.source_unit_id).toEqual([
      'lecture-01-section-175-model-architecture','lecture-03-architecture-framing',
      'lecture-03-recap','assignment-01-task-transformer-block','assignment-01-task-transformer-lm']);
    for (const id of ['architecture-lineage','architecture-evidence-boundary'])
      expect(body).toContain('id="'+id+'"');
    expect(fm.previous).toContain('04 Позиционная информация');
    expect(body).toContain('**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]');
  });
  it('contains six executable examples and both complete model graphs', () => {
    expect(body.match(/\x60\x60\x60python\n/g)).toHaveLength(6);
    for (const fact of ['class TinyTranslation','class TinyCausalLM','class DecoderBlock',
      'self.head.weight = self.emb.weight','loss.backward()','0.549012',
      'cross-attention — \x60[1,2,5,4]\x60','код руководства',
      'Официальные адаптеры и тесты']) {
      expect(body).toContain(fact);
    }
    expect(body).not.toContain('Post-LN-like');
    expect(body).not.toContain('\x60\x60\x60mermaid');
  });
  it('preserves all seven original figures and exact source/licensing provenance', () => {
    expect(figures).toHaveLength(7);
    const embedded = [...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set(figures.map((f: any)=>f.asset)));
    for (const f of figures) {
      const rows = registry.filter((r: any)=>r.asset===f.asset);
      expect(rows).toHaveLength(1);
      const r = rows[0];
      expect(r.id).toBe(f.id);
      for (const key of ['author','source_url','source_image_url','license','license_url',
        'source_asset','source_page_snapshot','source_page_sha256','sha256'])
        expect(r[key]).toBe(f[key]);
      expect(r.metadata_status).toBe('verified');
      expect(r.source_asset_sha256).toBe(f.sha256);
      expect(r.used_in).toContain(chapter);
      const image = read(f.asset);
      expect(sha(image)).toBe(f.sha256);
      expect(image.equals(read(f.source_asset))).toBe(true);
      const page = read(f.source_page_snapshot);
      expect(sha(page)).toBe(f.source_page_sha256);
      expect(r.license_evidence).toBe(f.source_page_snapshot);
      expect(r.license_evidence_sha256).toBe(f.source_page_sha256);
      expect(page.toString()).toContain('creativecommons.org/licenses/by-nc-sa/4.0/');
      if (f.frames === 1) {
        expect(image.subarray(1,4).toString()).toBe('PNG');
        expect([image.readUInt32BE(16),image.readUInt32BE(20)]).toEqual([f.width,f.height]);
      } else {
        expect(image.subarray(0,6).toString()).toMatch(/^GIF8[79]a$/);
        expect([image.readUInt16LE(6),image.readUInt16LE(8)]).toEqual([f.width,f.height]);
      }
    }
  });
  it('keeps the original-language assignment excerpt and full MIT notice', () => {
    expect(body).toContain('A Transformer block contains two');
    expect(body).toContain('a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=26');
    const source = read('05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/LICENSE');
    const copy = read('site/public/licenses/stanford-cs336-assignment1-MIT.txt');
    expect(copy.equals(source)).toBe(true);
    expect(body).toContain('/licenses/stanford-cs336-assignment1-MIT.txt');
  });
});
