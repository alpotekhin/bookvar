import { describe, expect, it } from 'vitest';
import {
  cacheBlocks, createScheduler, defaultSchedulerRequests, defaultSchedulerSettings,
  KV_BYTES_PER_TOKEN, runScheduler, schedulerFinished, schedulerStalled, stepScheduler,
} from '../src/lib/inference-scheduler';

const settings = defaultSchedulerSettings;
const create = () => createScheduler(defaultSchedulerRequests, settings);

describe('interactive scheduling model', () => {
  it('emits the first token in prefill; output is not an additional KV entry', () => {
    let state = createScheduler([{ id: 'A', arrival: 0, prompt: 3, output: 3 }], settings);
    state = stepScheduler(state);
    expect(state.requests[0]).toMatchObject({ prefilled: 3, emitted: 1, cached: 3, firstTokenAt: 1 });
    expect(state.history[0].work).toEqual([{ id: 'A', phase: 'prefill', tokens: 3, emitted: 1 }]);
    state = stepScheduler(state);
    expect(state.requests[0]).toMatchObject({ emitted: 2, cached: 4 });
    state = stepScheduler(state);
    expect(state.history[2].memoryDuring.reduce((n, block) => n + block.used, 0)).toBe(5);
    expect(state.requests[0]).toMatchObject({ emitted: 3, cached: 0, phase: 'finished' });
    expect(cacheBlocks(state).every(block => block.owner === null)).toBe(true);
    expect(state.tick).toBe(3);
    expect(stepScheduler(state)).toBe(state);
  });

  it('releases a one-token request at the end of prefill with no decode pass', () => {
    const state = runScheduler(createScheduler([{ id: 'A', arrival: 0, prompt: 6, output: 1 }], settings));
    expect(state.history.map(iteration => iteration.work)).toEqual([
      [{ id: 'A', phase: 'prefill', tokens: 4, emitted: 0 }],
      [{ id: 'A', phase: 'prefill', tokens: 2, emitted: 1 }],
    ]);
    expect(state.requests[0].firstTokenAt).toBe(2);
    expect(state.requests[0].finishedAt).toBe(2);
    expect(state.history.at(-1)!.memoryAfter.every(block => block.owner === null)).toBe(true);
  });

  it('matches a hand-worked 5-iteration refill trace and 7-iteration closed cohort', () => {
    const continuous = runScheduler(create());
    const cohort = runScheduler(createScheduler(defaultSchedulerRequests, { ...settings, admission: 'cohort' }));
    expect(continuous.history.map(row => row.work.map(work => work.id + ':' + work.phase + ':' + work.tokens))).toEqual([
      ['A:prefill:2', 'B:prefill:2'], ['A:decode:1', 'B:decode:1'],
      ['B:decode:1', 'C:prefill:1'], ['B:decode:1', 'D:prefill:1'],
      ['B:decode:1', 'D:decode:1'],
    ]);
    expect(continuous.tick).toBe(5);
    expect(cohort.tick).toBe(7);
    expect(continuous.requests.map(row => row.finishedAt)).toEqual([2, 5, 3, 5]);
    expect(cohort.requests.map(row => row.finishedAt)).toEqual([2, 5, 6, 7]);
    for (const state of [continuous, cohort]) {
      expect(state.requests.reduce((n, row) => n + row.emitted, 0)).toBe(10);
      expect(state.history.reduce((n, row) => n + row.usedBudget, 0)).toBe(12);
    }
  });

  it('schedules existing decode before filling the remainder with a prompt chunk', () => {
    let state = createScheduler([
      { id: 'A', arrival: 0, prompt: 2, output: 4 },
      { id: 'B', arrival: 1, prompt: 12, output: 2 },
    ], settings);
    state = stepScheduler(stepScheduler(state));
    expect(state.history[1].work).toEqual([
      { id: 'A', phase: 'decode', tokens: 1, emitted: 1 },
      { id: 'B', phase: 'prefill', tokens: 3, emitted: 0 },
    ]);
  });

  it('reports an impossible whole-prompt schedule instead of inventing progress', () => {
    const state = runScheduler(createScheduler([{ id: 'A', arrival: 0, prompt: 8, output: 1 }], {
      ...settings, chunkedPrefill: false,
    }));
    expect(schedulerStalled(state)).toBe(true);
    expect(state.history[0].waits).toEqual([{ id: 'A', reason: 'whole-prompt' }]);
    expect(state.requests[0].emitted).toBe(0);
    expect(stepScheduler(state)).toBe(state);
  });

  it('accounts for physical-block rounding, memory stalls and reduced concurrency', () => {
    const jobs = [{ id: 'A', arrival: 0, prompt: 4, output: 3 }, { id: 'B', arrival: 0, prompt: 4, output: 3 }];
    const crowded = runScheduler(createScheduler(jobs, { ...settings, tokenBudget: 8, totalBlocks: 2 }));
    expect(schedulerStalled(crowded)).toBe(true);
    expect(crowded.requests.map(row => [row.cached, row.emitted])).toEqual([[4, 1], [4, 1]]);
    expect(crowded.history.at(-1)!.waits).toEqual([
      { id: 'A', reason: 'kv-capacity' }, { id: 'B', reason: 'kv-capacity' },
    ]);
    const serial = runScheduler(createScheduler(jobs, { ...settings, tokenBudget: 8, totalBlocks: 2, maxRunning: 1 }));
    expect(schedulerFinished(serial)).toBe(true);
    expect(serial.peakBlocks).toBe(2);
    expect(serial.requests.map(row => row.emitted)).toEqual([3, 3]);
    expect(KV_BYTES_PER_TOKEN).toBe(131072);
  });

  it('does not reuse finishing blocks within the same forward pass', () => {
    const state = stepScheduler(createScheduler([
      { id: 'A', arrival: 0, prompt: 4, output: 1 },
      { id: 'B', arrival: 0, prompt: 4, output: 1 },
    ], { ...settings, tokenBudget: 8, totalBlocks: 1 }));
    expect(state.history[0].work.map(row => row.id)).toEqual(['A']);
    expect(state.requests[1].prefilled).toBe(0);
    expect(schedulerFinished(runScheduler(state))).toBe(true);
  });

  it('preserves logical block order even when a lower physical index becomes free', () => {
    let state = createScheduler([
      { id: 'A', arrival: 0, prompt: 1, output: 1 },
      { id: 'B', arrival: 0, prompt: 4, output: 3 },
    ], { ...settings, tokenBudget: 8, totalBlocks: 3 });
    state = stepScheduler(stepScheduler(state));
    expect(state.requests[1].blocks).toEqual([1, 0]);
    expect(cacheBlocks(state).slice(0, 2)).toEqual([
      { index: 0, owner: 'B', logicalIndex: 1, used: 1 },
      { index: 1, owner: 'B', logicalIndex: 0, used: 4 },
    ]);
  });

  it('waits for a future arrival rather than diagnosing an empty initial queue as deadlock', () => {
    const state = runScheduler(createScheduler([{ id: 'A', arrival: 3, prompt: 1, output: 1 }], settings));
    expect(state.history.slice(0, 3).every(row => row.work.length === 0 && !row.stalled)).toBe(true);
    expect(state.requests[0]).toMatchObject({ admittedAt: 3, finishedAt: 4 });
  });

  it('keeps prior states and caller configuration unchanged', () => {
    const state = create();
    const before = JSON.stringify(state);
    stepScheduler(state);
    expect(JSON.stringify(state)).toBe(before);
    const config = { ...settings };
    const copy = createScheduler(defaultSchedulerRequests, config);
    config.totalBlocks = 1;
    expect(copy.settings.totalBlocks).toBe(8);
  });

  it('satisfies token, ownership and cache-length invariants across 1,536 schedules', () => {
    let cases = 0;
    for (const tokenBudget of [1, 2, 4, 8]) for (const totalBlocks of [1, 2, 4, 8])
      for (const blockSize of [1, 4]) for (const maxRunning of [1, 2, 3])
        for (const admission of ['continuous', 'cohort'] as const) for (const prompt of [1, 3])
          for (const output of [1, 4]) for (const chunkedPrefill of [true, false]) {
            let state = createScheduler([
              { id: 'A', arrival: 0, prompt, output },
              { id: 'B', arrival: 1, prompt: 2, output: 2 },
              { id: 'C', arrival: 2, prompt: 1, output: 1 },
            ], { ...settings, tokenBudget, totalBlocks, blockSize, maxRunning, admission, chunkedPrefill });
            for (let step = 0; step < 64 && !schedulerFinished(state) && !schedulerStalled(state); step++) {
              state = stepScheduler(state);
              const trace = state.history.at(-1)!;
              expect(trace.usedBudget).toBeLessThanOrEqual(tokenBudget);
              expect(trace.usedBudget).toBe(trace.work.reduce((n, work) => n + work.tokens, 0));
              expect(trace.memoryDuring.filter(block => block.owner).length).toBeLessThanOrEqual(totalBlocks);
              const ids = state.requests.flatMap(row => row.blocks);
              expect(new Set(ids).size).toBe(ids.length);
              for (const row of state.requests) {
                expect(row.emitted).toBeLessThanOrEqual(row.output);
                if (row.phase === 'decode') expect(row.cached).toBe(row.prompt + row.emitted - 1);
                if (row.phase === 'prefill') expect(row.cached).toBe(row.prefilled);
                if (row.phase === 'finished') expect(row.blocks).toEqual([]);
              }
            }
            expect(schedulerFinished(state) || schedulerStalled(state)).toBe(true);
            if (schedulerFinished(state)) {
              const expected = state.requests.reduce((n, row) => n + row.prompt + row.output - 1, 0);
              expect(state.history.reduce((n, row) => n + row.usedBudget, 0)).toBe(expected);
            }
            cases++;
          }
    expect(cases).toBe(1536);
  });

  it.each([0, -1, NaN, Infinity, 1.5])('rejects invalid token budgets: %s', tokenBudget => {
    expect(() => createScheduler(defaultSchedulerRequests, { ...settings, tokenBudget })).toThrow();
  });
  it('rejects ambiguous IDs and invalid modes', () => {
    expect(() => createScheduler([defaultSchedulerRequests[0], defaultSchedulerRequests[0]], settings)).toThrow();
    expect(() => createScheduler([{ id: '<script>', arrival: 0, prompt: 1, output: 1 }], settings)).toThrow();
    expect(() => createScheduler(defaultSchedulerRequests, { ...settings, admission: 'other' as 'cohort' })).toThrow();
  });
});
