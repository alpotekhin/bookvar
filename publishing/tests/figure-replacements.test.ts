import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const journal = JSON.parse(read('docs/audit/2026-10-30-figure-replacements-additions.json'));
const packet = JSON.parse(read('05 Источники/Originals/figure-replacements-2026-10-30/manifest.json'));
const ru = journal.files[0].source;
const en = journal.files[1].source;
const grpo = journal.files[2].source;

describe('licensed visual replacements', () => {
  it('restores preceding chapter and registry bytes exactly and rejects unrecorded edits', () => {
    for (const row of journal.files) {
      const current = read(row.source);
      expect(sha(current)).toBe(row.after_sha256);
      const restored = beforeTopicConsolidation(current, row.source, root,
        '2026-10-30-atlas-release-label-additions.json');
      expect(sha(restored)).toBe(row.before_sha256);
      const fences = (text: string) => [...text.matchAll(/^(?:`{3}|~{3})python\n([\s\S]*?)^(?:`{3}|~{3})\s*$/gm)].map(x=>x[1]);
      expect(fences(current)).toEqual(fences(restored));
      expect(() => beforeTopicConsolidation(current + '\nUNREVIEWED\n', row.source, root,
        '2026-10-30-atlas-release-label-additions.json')).toThrow();
    }
  });

  it('retains all eight factual configurations identically in RU and EN', () => {
    const rows = (path: string) => {
      const table = read(path).split('\n\n').find(block=>block.startsWith('| Model | Parameters |'));
      if (!table) throw new Error('Missing configuration table in ' + path);
      return table.split('\n').slice(2);
    };
    const expected = [
      ['GPT-3 Small','125M','12','768','12','64','0.5M','$6.0\\times10^{-4}$'],
      ['GPT-3 Medium','350M','24','1024','16','64','0.5M','$3.0\\times10^{-4}$'],
      ['GPT-3 Large','760M','24','1536','16','96','0.5M','$2.5\\times10^{-4}$'],
      ['GPT-3 XL','1.3B','24','2048','24','128','1M','$2.0\\times10^{-4}$'],
      ['GPT-3 2.7B','2.7B','32','2560','32','80','1M','$1.6\\times10^{-4}$'],
      ['GPT-3 6.7B','6.7B','32','4096','32','128','2M','$1.2\\times10^{-4}$'],
      ['GPT-3 13B','13.0B','40','5140','40','128','2M','$1.0\\times10^{-4}$'],
      ['GPT-3 175B','175.0B','96','12288','96','128','3.2M','$0.6\\times10^{-4}$'],
    ];
    const actual = rows(ru).map(x=>x.split('|').slice(1,-1).map(x=>x.trim()));
    expect(actual).toEqual(expected);
    expect(rows(en)).toEqual(rows(ru));
    expect(YAML.parse(read(en).split('---')[1]).canonical_revision).toBe(sha(read(ru)));
  });

  it('uses the two unchanged Lambert originals and retains links to the paper comparison', () => {
    for (const item of packet.published.slice(1)) {
      expect(createHash('sha256').update(readFileSync(resolve(root, item.asset))).digest('hex'))
        .toBe(item.sha256);
      expect(item.license).toBe('MIT');
      expect(read(grpo)).toContain(item.asset);
      expect(read('site/public/licenses/rlhf-book-MIT.txt')).toContain('Permission is hereby granted');
    }
    expect(read(grpo)).toContain('https://arxiv.org/pdf/2402.03300#page=13');
    expect(read(grpo)).not.toContain('ppo-vs-grpo.png');
  });

  it('preserves retired source history without changing unverified rights into permission', () => {
    const assets = YAML.parse(read('05 Источники/asset-registry.yml')).assets;
    expect(packet.retired).toHaveLength(3);
    for (const retired of packet.retired) {
      expect(existsSync(resolve(root, retired.asset))).toBe(true);
      const row = assets.find((x: any)=>x.asset===retired.asset);
      expect(row.metadata_status).toBe('needs-confirmation');
      expect(row.publication_status).toBe('retired');
      expect(row.used_in).toEqual(retired.current_usage);
      for (const chapter of [ru,en,grpo]) expect(read(chapter)).not.toContain(retired.asset);
    }
  });
});
