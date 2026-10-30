import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const read = (p: string) => readFile(join(root, p), 'utf8');
const sha = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');
const chapter = '06 Практика/14 Собрать минимальный inference engine.md';
const results = '06 Практика/Results/edls-admission-2026-10-26';

describe('EDLS neural admission experiment', () => {
  it('preserves the prior practice except its explicitly superseded pending-experiment sentence', async () => {
    const prior = beforeTextCheckpointRuntime(await read(chapter))
      .replace(/<!-- BEGIN REVIEW: EDLS neural admission 2026-10-26 -->[\s\S]*?<!-- END REVIEW: EDLS neural admission 2026-10-26 -->\n\n/, '')
      .replace('explains that extension. The following section runs a bounded neural CPU\ncomparison; the assigned receipts and equal-cost trace are not its measurements.',
        'explains that extension. The policy benchmark remains an experiment to run;\nthe assigned receipts and equal-cost scheduler trace are not its results.');
    expect(sha(prior)).toBe('2adf0af6a979805bd8136d27a075b1e103b0eb3e01c5c640d9e952b0d2e0686e');
  });

  it('publishes the complete scheduler rather than a shortened snippet', async () => {
    const text = await read(chapter);
    const code = text.split('<!-- BEGIN SOLUTION: EDLS admission module -->')[1]
      .split('<!-- END SOLUTION: EDLS admission module -->')[0].trim();
    const module = await read('06 Практика/Code/edls_admission_reference.py');
    expect(code).toBe('```python\n'+module.trimEnd()+'\n```');
    expect(await read('site/public/examples/edls_admission_reference.py')).toBe(module);
  });

  it('binds recorded runs to the actual core, scheduler, experiment and receipt calculation', async () => {
    const bytes = await readFile(join(root, results, 'runs.json'));
    expect(sha(bytes)).toBe('9f7b5094faefb3503dc0edf8cf1248612e3a281b0e0b8525b3c2994900067103');
    expect(await readFile(join(root, 'site/public/examples/edls-admission-cpu-2026-10-26.json'))).toEqual(bytes);
    const data = JSON.parse(bytes.toString('utf8'));
    for (const [name, hash] of Object.entries(data.source_hashes)) {
      const path = name.startsWith('check_') ? 'publishing/tools/'+name : '06 Практика/Code/'+name;
      expect(sha(await readFile(join(root, path)))).toBe(hash);
    }
    const fence = (await read(chapter)).match(/<!-- example: edls-metric-receipts -->\s*```python\n([\s\S]*?)\n```/)![1];
    expect(sha(fence)).toBe(data.receipt_fence_sha256);
    expect(data.scope).toContain('separate stdlib producer process');
    expect(data.runtime.device).toBe('cpu');
    expect(data.runtime.parameters).toBe(22720);
  });

  it('contains all 54 cells with preserved IDs, token counts, monotonic events and no hidden failures', async () => {
    const data = JSON.parse(await read(results+'/runs.json'));
    expect(data.runs).toHaveLength(54);
    const cells = new Set<string>();
    const outputs = new Map<string, number[]>();
    for (const run of data.runs) {
      cells.add([run.workload, run.slots, run.repeat, run.policy].join(':'));
      expect(run.receipts).toHaveLength(12);
      expect(run.metrics.outcomes).toEqual({completed:12,cancelled:0,rejected:0,error:0});
      expect(run.metrics.emitted_tokens).toBe(run.workload === 'decode_heavy' ? 156 : 18);
      expect(run.metrics.output_tokens_per_second).toBeCloseTo(run.metrics.emitted_tokens/run.duration_s, 9);
      expect(new Set(run.receipts.map((r: {id:string})=>r.id))).toEqual(new Set(data.workloads[run.workload].map((j:{id:string})=>j.id)));
      for (const receipt of run.receipts) {
        const events = [receipt.scheduled, receipt.submitted, receipt.observed,
          receipt.admitted, ...receipt.tokens, receipt.finished];
        expect(events.every(Number.isFinite)).toBe(true);
        for (let i=1;i<events.length;i++) expect(events[i]).toBeGreaterThanOrEqual(events[i-1]);
        expect(receipt.output_ids).toHaveLength(receipt.tokens.length);
        if (outputs.has(receipt.id)) expect(receipt.output_ids).toEqual(outputs.get(receipt.id));
        outputs.set(receipt.id, receipt.output_ids);
      }
      let ended = 0;
      for (const phase of run.phases) {
        expect(phase.start).toBeGreaterThanOrEqual(ended);
        expect(phase.end).toBeGreaterThanOrEqual(phase.start);
        expect(phase.error).toBeNull();
        expect(phase.ids.length).toBeLessThanOrEqual(run.slots);
        expect(phase.reserved_positions).toBeLessThanOrEqual(256);
        ended = phase.end;
      }
    }
    expect(cells.size).toBe(54);
    expect(outputs.size).toBe(24);
  });

  it('keeps the measured plot distinct from reused source illustrations', async () => {
    const manifest = JSON.parse(await read(results+'/manifest.json'));
    expect(manifest.origin).toBe('recorded-cpu-experiment');
    for (const [path, hash] of Object.entries(manifest.sha256))
      expect(sha(await readFile(join(root, path)))).toBe(hash);
    const png = await readFile(join(root, results, 'phase-timeline.png'));
    expect([...png.subarray(0,8)]).toEqual([137,80,78,71,13,10,26,10]);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1800,990]);
  });
});
