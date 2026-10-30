import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { loadManifest } from '../adapter/manifest.js';
import { readingPaths } from '../adapter/reading-paths.js';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const source = '00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts.md';
const read = (p: string) => readFileSync(resolve(root, p), 'utf8');
const hash = (p: string) => createHash('sha256').update(readFileSync(resolve(root, p))).digest('hex');
const digest = (text: string) => createHash('sha256').update(text).digest('hex');
const body = read(source);

describe('canonical fine-tuning and PEFT chapter', () => {
  it('places one source-native English explanation between SFT and preference data', () => {
    const book = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.find(s => s.id === 'textbook')!;
    const route = 'textbook/post-training/fine-tuning-peft';
    const page = book.pages.find(p => p.route === route)!;
    expect(page.source).toBe(source);
    expect(page.sourceEn).toBeUndefined();
    expect(body).toContain('source_language: en');
    const paths = readingPaths(book).filter(p => p.routes.includes(route));
    expect(paths).toHaveLength(1);
    const index = paths[0].routes.indexOf(route);
    expect(paths[0].routes.slice(index - 1, index + 2)).toEqual([
      'textbook/post-training/sft-instruction-data', route, 'textbook/post-training/preference-data',
    ]);
    for (const p of ['00 Учебник/_index.md', 'en/00 Textbook/_index.md',
      '01 Справочник/_index.md', 'en/01 Reference/_index.md'])
      expect(read(p)).toContain(source.replace(/\.md$/, ''));
    expect(read('00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md'))
      .toContain('next: "[[' + source.replace(/\.md$/, '') + ']]"');
  });

  it('keeps six original figures and explicit rights evidence, without synthetic replacement diagrams', () => {
    const specs = JSON.parse(read('00 Учебник/Assets/Figures/curated/peft-2026-10-29/assets.json'));
    const registry = parse(read('05 Источники/asset-registry.yml')).assets;
    expect(specs).toHaveLength(6);
    const expected = [
      '6ffea276f84b38e57c4ac45cdfdf07e856885a696d944b25d2ee5c697e19431c',
      '398088f0cf9b564859a3fcb5b66002b070bf60d0001f69ae561d1a1a17bf4773',
      '4313422c5f2755897fb8ddfc5b99251358f679647ec0f2d120a3f1ff060defe7',
      'f8e55f300e76b69cfd48c146bd71e9f40d4173c610fdb3c0562f369689a88908',
      '097897ac636473f8434692b0db5fbbc10395ec014b1e5ee136a501bd6aa1e86c',
      '5035d5990255985976069eb59ee2833c7f99605d9a7d79b70e3e9e4b8c4df595',
    ];
    for (const [i, spec] of specs.entries()) {
      expect(body).toContain(spec.asset);
      expect(hash(spec.asset)).toBe(expected[i]);
      expect(spec.sha256).toBe(expected[i]);
      expect(hash(spec.source_asset)).toBe(spec.source_asset_sha256);
      expect(registry.find((r: {asset: string}) => r.asset === spec.asset).used_in).toContain(source);
      if (spec.derivation === 'source-raster-copy') {
        expect(hash(spec.source_asset)).toBe(spec.sha256);
        expect(spec.commit).toBe('861b6c920fe27492d35256c39f266592a518d50c');
      }
      if (spec.license_evidence) expect(hash(spec.license_evidence)).toBe(spec.license_evidence_sha256);
    }
    expect(body).not.toContain('```mermaid');
  });

  it('preserves four HF source excerpts with only documentation-link/typography normalization', () => {
    const normalize = (s: string) => s.replace(/^> ?/gm, '')
      .replace(/\[\x60~LoraModel\.([a-z_]+)\x60\]/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/[\x60*]/g, '').replace(/\s+/g, ' ').trim();
    const quotes = [...body.matchAll(/(?:^>[^\n]*(?:\n|$))+/gm)].map(m => normalize(m[0]));
    const cases = [
      ['hf-lora.md', 'LoRA represents the weight updates'],
      ['hf-lora.md', 'To eliminate latency, use the'],
      ['hf-prompt-tuning.md', 'Prompt tuning adds a task-specific'],
      ['hf-prefix-tuning.md', 'Prefix tuning prefixes a series'],
    ];
    for (const [file, start] of cases) {
      const quote = quotes.find(q => q.startsWith(start));
      expect(quote).toBeDefined();
      expect(normalize(read('05 Источники/Originals/peft-2026-10-29/' + file))).toContain(quote);
    }
    expect(hash('site/public/licenses/peft-fee923eb-Apache-2.0.txt'))
      .toBe(hash('05 Источники/Originals/peft-2026-10-29/peft-Apache-2.0.txt'));
  });

  it('retains the worked mechanisms and executable scope rather than universal quality/VRAM recipes', () => {
    expect(body.match(/^```python$/gm)).toHaveLength(18);
    for (const anchor of ['choosing-trainable-parameters', 'bottleneck-adapters', 'lora-update',
      'merging-and-serving', 'qlora-training', 'soft-prompts', 'prefix-keys-values'])
      expect(body).toContain('id="' + anchor + '"');
    const prose = body.replace(/\s+/g, ' ');
    for (const phrase of ['first update', '0.81', '18 million parameters', '87.5',
      'not a universal', 'zero prompt changes', 'not the inference KV-cache',
      'Report failed or degenerate runs', 'not a full-model fine-tuning result'])
      expect(prose.includes(phrase), phrase).toBe(true);
  });

  it('reverses the precise additions without changing historical acceptance hashes', () => {
    const journal = JSON.parse(read('docs/audit/2026-10-29-peft-additions.json'));
    for (const row of journal.files) {
      let text = beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-peft-additions.json');
      expect(digest(text)).toBe(row.after_sha256);
      for (const change of [...row.changes].reverse()) {
        expect(text.split(change.new)).toHaveLength(2);
        text = text.replace(change.new, () => change.old);
      }
      expect(digest(text)).toBe(row.before_sha256);
      expect(beforeTopicConsolidation(read(row.source), row.source, root,
        '2026-10-29-scheduler-lab-additions.json')).toBe(text);
      expect(() => beforeTopicConsolidation(read(row.source) + '\nUnreviewed addition.\n',
        row.source, root, '2026-10-29-scheduler-lab-additions.json')).toThrow();
    }
    expect(() => beforeTopicConsolidation('', 'unknown', root, 'unknown-journal')).toThrow();
  });
});
