import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const read = (p: string) => readFileSync(resolve(root,p));
const chapter = '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md';
const body = read(chapter).toString();
const specs = JSON.parse(read('publishing/tools/architecture-figures.json').toString());
const registry = YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
const sha = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

describe('architecture patterns chapter contracts', () => {
  it('keeps all four independently executable examples and explicit objective boundaries', () => {
    expect(body.match(/\x60\x60\x60python\n/g)).toHaveLength(4);
    for (const fact of ['1.832581','0.880797','11.125','12.25','75/25=3',
      'ignore_index=-100','corrupt_spans','decoder_ids','_shift_right','MLM','UniLM'])
      expect(body).toContain(fact);
    expect(body).toContain('не воспроизводит случайный отбор T5');
    expect(body).toContain('не заявляет');
    expect(body).not.toContain('\x60\x60\x60mermaid');
  });
  it('embeds exactly the six reviewed originals with source and licensing evidence', () => {
    expect(specs).toHaveLength(6);
    const embedded = [...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
    expect(new Set(embedded)).toEqual(new Set(specs.map((s: any)=>s.asset)));
    for (const s of specs) {
      const rows = registry.filter((r: any)=>r.asset===s.asset);
      expect(rows).toHaveLength(1);
      const row = rows[0];
      expect(row.used_in).toContain(chapter);
      for (const [key,value] of Object.entries(s)) {
        if (['name','width','height'].includes(key)) continue;
        expect(row[key]).toEqual(value);
      }
      for (const [path,hash] of [[s.asset,s.sha256],[s.source_asset,s.source_asset_sha256],
        [s.source_page_snapshot,s.source_page_sha256],[s.license_evidence,s.license_evidence_sha256]])
        expect(sha(read(path))).toBe(hash);
      const png = read(s.asset);
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([s.width,s.height]);
    }
  });
  it('does not keep obsolete figure usages or erase their other chapters', () => {
    for (const id of ['curated-bert-5f2b6bb2468e','curated-gpt-30-7128f23d2a59',
      'curated-t5-60a4f5c68c9e','curated-t5-961e963bd519']) {
      const row = registry.find((r: any)=>r.id===id);
      expect(row.used_in).not.toContain(chapter);
      if(id.startsWith('curated-t5-')){
        // The later T5 review replaced the concept's final legacy uses too.
        expect(row.used_in).toEqual([]);
        expect(read(row.asset).length).toBeGreaterThan(0);
        expect(row.modifications).toContain('Historical raw copy retained');
      }else expect(row.used_in.length).toBeGreaterThan(0);
    }
    const mlm = registry.find((r: any)=>r.id==='curated-bert-b81329241b13');
    expect(mlm.used_in).toContain('00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md');
  });
});
