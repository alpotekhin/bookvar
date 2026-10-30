import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { beforeTextCheckpointRuntime } from './helpers/practice-runtime-baseline';

const root = resolve(import.meta.dirname, '../..');
const base = '06 Практика/Results/runtime-2026-10-29/';
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const json = (name: string) => JSON.parse(read(base + name));
const hash = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const archive = resolve(root, 'site/public/examples/runtime-labs-2026-10-29.zip');

describe('Executable local practice receipts, 29 October', () => {
  it('keeps the complete prior engine explanation outside the bounded text/checkpoint addition', () => {
    expect(hash(beforeTextCheckpointRuntime(read('06 Практика/14 Собрать минимальный inference engine.md'))))
      .toBe('df419eaed79373346f3e66f900cd635714e733bcb754fa97d335fbcd6dd31fc4');
  });

  it('retains actual 34-app execution and the separately completed UI report', () => {
    const initial = json('marimo.json'), corrected = json('marimo-source-runtime.json');
    expect(initial.passed).toBe(30);
    expect(corrected.passed).toBe(34);
    expect(corrected.total).toBe(34);
    expect(corrected.records.every((r: any) => r.exit_code === 0 && r.result.definitions > 0)).toBe(true);
    const ui = json('marimo-lab10-ui.json');
    expect(ui.incomplete_fields).toEqual([]);
    expect(ui.knob_settings.demand_qps).toBe(200);
    expect(ui.evidence_summary.actual_crossover_days).toBe(12.86);
    expect(corrected.scope).toContain('not all prediction gates');
  });

  it('retains CPU/MPS checkpoint checks without claiming pretrained quality or CUDA', () => {
    for (const device of ['cpu', 'mps']) {
      const result = json(`text-${device}.json`);
      expect(result.device).toBe(device);
      expect(result.oracle_matches).toBe(4);
      expect(result.tokenizer_roundtrip).toBe(true);
      expect(result.checkpoint_logits_exact).toBe(true);
      expect(result.checkpoint_bytes).toBe(84944);
      expect(result.parameters).toBe(19776);
      expect(result.scope).toContain('random checkpoint');
      for (const row of result.rows) expect(row.cache_positions).toBe(row.input_ids.length + 4);
    }
    const capabilities = json('capabilities.json');
    expect(capabilities.cuda).toBe(false);
    for (const program of ['text_checkpoint_lab.py', 'ddp_step_lab.py']) {
      expect(capabilities[program].exit_code).not.toBe(0);
      expect(capabilities[program].result_created).toBe(false);
    }
  });

  it('checks every recorded rank result and the global-batch DDP comparison', () => {
    const chapter = read('06 Практика/09 Реализовать ring all-reduce.md');
    expect(chapter).toContain('упрощённое дерево без сегментации');
    expect(chapter).toContain('эвристика источника, а не точная универсальная граница');
    const ring = json('ring.json');
    expect(ring.configurations).toBe(24);
    let ranks = 0;
    for (const run of ring.results) {
      for (const cases of run.ranks) for (const entry of cases) {
        ranks++;
        expect(entry.output).toEqual(Array.from({ length: entry.n }, (_, i) =>
          run.world_size * i + 5 * run.world_size * (run.world_size - 1)));
        expect(entry.trace).toHaveLength(2 * (run.world_size - 1));
      }
    }
    expect(ranks).toBe(72);
    for (const rank of [0, 1]) {
      const ddp = json(`ddp-gloo-rank-${rank}.json`);
      expect(ddp.step_matches_global_batch).toBe(true);
      expect(ddp.gradient_max_abs_error).toBeLessThan(1e-7);
      expect(ddp.all_reduce_sum).toBe(3);
    }
  });

  it('reconciles all offered traffic with real scraper and protocol receipts', () => {
    const result = json('service.json');
    expect(result.initial_ready_status).toBe(503);
    expect(result.grpc_label).toBe('green');
    expect(result.grpc_invalid).toBe('INVALID_ARGUMENT');
    let offered = 0, completed = 1; // one pre-load HTTP correctness request
    for (const run of result.runs) {
      offered += run.offered;
      completed += run.completed;
      expect(run.offered).toBe(run.records.length);
      expect(run.completed + run.rejected).toBe(run.offered);
      for (const r of run.records) {
        expect(r.sent).toBeGreaterThanOrEqual(r.scheduled);
        expect(r.completed).toBeGreaterThan(r.sent);
        expect([200, 429]).toContain(r.status);
      }
    }
    expect(offered).toBe(620);
    expect(Number(result.scrape_value[0].value[1])).toBe(completed);
    expect(Number(result.inflight[0].value[1])).toBe(0);
    expect(result.targets[0].health).toBe('up');
    expect(result.scope).toContain('synthetic 20ms service delay');
    for (const [file, expected] of Object.entries(result.source_hashes)) {
      expect(hash(read('06 Практика/Code/observable_service/' + file))).toBe(expected);
    }
  });

  it('ships exact executable sources, receipts and attribution in the downloadable package', () => {
    for (const file of ['ring_collective_lab.py', 'ddp_step_lab.py', 'text_checkpoint_lab.py',
      'edls_kv_reference.py', 'run_harvard_lab.py', 'prepare_harvard_runtime.py',
      'check_harvard_runtime.py', 'run_observable_lab.py', 'check_runtime_capabilities.py']) {
      expect(execFileSync('unzip', ['-p', archive, 'Code/' + file], { encoding: 'utf8' }))
        .toBe(read('06 Практика/Code/' + file));
    }
    expect(execFileSync('unzip', ['-p', archive, 'LICENSE-MIT.txt'], { encoding: 'utf8' }))
      .toBe(read('05 Источники/Originals/edls-inference-2026-10-26/LICENSE-MIT.txt'));
    expect(execFileSync('unzip', ['-p', archive, 'Results/runtime-2026-10-29/service.json'], { encoding: 'utf8' }))
      .toBe(read(base + 'service.json'));
  });
});
