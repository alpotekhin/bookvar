import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (p:string) => readFileSync(resolve(root,p));
const hash = (s:Buffer|string) => createHash('sha256').update(s).digest('hex');

describe('tool reward explanations in the canonical RLVR chapter', () => {
  it('runs both saved examples and compares the declared scope to actual ToolRL code', () => {
    const out = execFileSync('python3',['publishing/tools/check_tool_reward_examples.py'],
      {cwd:root,encoding:'utf8'});
    const r = JSON.parse(out.slice(0,out.indexOf('\nPASS ')));
    expect(r.new_published_python_fences).toBe(2);
    expect(r.pinned_code_comparisons).toBe(72);
    expect(r.f1_cases).toBe(408);
    expect(r.greedy_candidate_order_rewards[0]).toBeCloseTo(3/7,12);
    expect(r.greedy_candidate_order_rewards[1]).toBeCloseTo(15/7,12);
    expect(r.model_calls).toBe(0);
    expect(r.environment_actions).toBe(0);
  });

  it('preserves the complete author figure and distributes the actual repository license', () => {
    const [a] = JSON.parse(read('00 Учебник/Assets/Figures/curated/toolrl/assets.json').toString());
    const registry = parse(read('05 Источники/asset-registry.yml').toString()).assets;
    expect(registry.filter((r:any)=>r.asset===a.asset)).toEqual([a]);
    expect(hash(read(a.asset))).toBe(a.sha256);
    expect(a.dimensions_px).toEqual([6696,2365]);
    expect(hash(read(a.license_evidence))).toBe(a.license_evidence_sha256);
    expect(hash(read(a.source_page_snapshot))).toBe(a.source_page_sha256);
    expect(read(a.license_public_copy).toString().trimEnd())
      .toBe(read(a.license_evidence).toString().trimEnd());
    expect(a.license).toBe('Apache-2.0');
    expect(a.modifications).toContain('no crop, redraw');
  });

  it('restores the preceding revisions exactly and rejects unknown changes', () => {
    const j = JSON.parse(read('docs/audit/2026-10-29-tool-rewards-additions.json').toString());
    expect(j.files).toHaveLength(2);
    for(const row of j.files) {
      const live = read(row.source).toString();
      const accepted = beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-tool-rewards-additions.json');
      expect(hash(accepted)).toBe(row.after_sha256);
      const old = beforeTopicConsolidation(live,row.source,root,
        '2026-10-29-constitutional-ai-additions.json');
      expect(hash(old)).toBe(row.before_sha256);
      expect(()=>beforeTopicConsolidation(live+'\nUnknown edit.',row.source,root,
        '2026-10-29-constitutional-ai-additions.json')).toThrow();
    }
    expect(hash(read('Concepts/Training/GRPO.md'))).toBe(
      'd9ef78345403abaac74b315dee091199a2c750e70652936b455aa9db2dc1bc55');
    expect(hash(read('Concepts/Training/RLVR.md'))).toBe(
      '16e44366ba81a677e062875451c4c71254568f958f6af938ba582581ff1046b4');
  });
});
