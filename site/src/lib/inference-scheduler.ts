/** Discrete teaching model: iterations and KV blocks, never GPU time. */
export interface SchedulerSettings {
  tokenBudget: number;
  maxRunning: number;
  totalBlocks: number;
  blockSize: number;
  chunkedPrefill: boolean;
  admission: 'continuous' | 'cohort';
}

export interface RequestSpec {
  id: string;
  arrival: number;
  prompt: number;
  output: number;
}

export interface RequestState extends RequestSpec {
  phase: 'queued' | 'prefill' | 'decode' | 'finished';
  prefilled: number;
  emitted: number;
  cached: number;
  blocks: number[];
  admittedAt: number | null;
  firstTokenAt: number | null;
  finishedAt: number | null;
}

export interface CacheBlock {
  index: number;
  owner: string | null;
  logicalIndex: number | null;
  used: number;
}

export interface ScheduledWork {
  id: string;
  phase: 'prefill' | 'decode';
  tokens: number;
  emitted: number;
}

export interface SchedulerIteration {
  tick: number;
  admitted: string[];
  work: ScheduledWork[];
  released: string[];
  usedBudget: number;
  memoryDuring: CacheBlock[];
  memoryAfter: CacheBlock[];
  waits: Array<{ id: string; reason: 'token-budget' | 'whole-prompt' | 'kv-capacity' }>;
  stalled: boolean;
}

export interface SchedulerState {
  settings: SchedulerSettings;
  requests: RequestState[];
  cohort: string[];
  tick: number;
  peakBlocks: number;
  history: SchedulerIteration[];
}

export const KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2;

export const defaultSchedulerSettings: SchedulerSettings = {
  tokenBudget: 4, maxRunning: 2, totalBlocks: 8, blockSize: 4,
  chunkedPrefill: true, admission: 'continuous',
};

export const defaultSchedulerRequests: RequestSpec[] = [
  { id: 'A', arrival: 0, prompt: 2, output: 2 },
  { id: 'B', arrival: 0, prompt: 2, output: 5 },
  { id: 'C', arrival: 1, prompt: 1, output: 1 },
  { id: 'D', arrival: 2, prompt: 1, output: 2 },
];

function boundedInteger(value: number, min: number, max: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new Error(`${name}: integer in [${min}, ${max}] required`);
}

export function createScheduler(requests: RequestSpec[], settings: SchedulerSettings): SchedulerState {
  boundedInteger(settings.tokenBudget, 1, 64, 'tokenBudget');
  boundedInteger(settings.maxRunning, 1, 6, 'maxRunning');
  boundedInteger(settings.totalBlocks, 1, 32, 'totalBlocks');
  boundedInteger(settings.blockSize, 1, 16, 'blockSize');
  if (typeof settings.chunkedPrefill !== 'boolean'
    || !['continuous', 'cohort'].includes(settings.admission)) throw new Error('Invalid scheduler mode');
  boundedInteger(requests.length, 1, 6, 'requests');
  if (new Set(requests.map(request => request.id)).size !== requests.length)
    throw new Error('Unique request IDs required');
  for (const request of requests) {
    if (!/^[A-Z]$/.test(request.id)) throw new Error('Request IDs must be single ASCII letters');
    boundedInteger(request.arrival, 0, 32, 'arrival');
    boundedInteger(request.prompt, 1, 64, 'prompt');
    boundedInteger(request.output, 1, 32, 'output');
  }
  return {
    settings: { ...settings }, tick: 0, peakBlocks: 0, cohort: [], history: [],
    requests: requests.map(request => ({
      ...request, phase: 'queued', prefilled: 0, emitted: 0, cached: 0, blocks: [],
      admittedAt: null, firstTokenAt: null, finishedAt: null,
    })),
  };
}

export function cacheBlocks(state: Pick<SchedulerState, 'settings' | 'requests'>): CacheBlock[] {
  return Array.from({ length: state.settings.totalBlocks }, (_, index) => {
    const owner = state.requests.find(request => request.blocks.includes(index));
    const logicalIndex = owner ? owner.blocks.indexOf(index) : null;
    return {
      index, owner: owner?.id ?? null, logicalIndex,
      used: owner ? Math.min(state.settings.blockSize,
        Math.max(0, owner.cached - logicalIndex! * state.settings.blockSize)) : 0,
    };
  });
}

export function schedulerFinished(state: SchedulerState): boolean {
  return state.requests.every(request => request.phase === 'finished');
}

export function schedulerStalled(state: SchedulerState): boolean {
  return state.history.at(-1)?.stalled ?? false;
}

export function stepScheduler(previous: SchedulerState): SchedulerState {
  if (schedulerFinished(previous) || schedulerStalled(previous)) return previous;
  const state: SchedulerState = {
    ...previous, cohort: [...previous.cohort], history: [...previous.history],
    requests: previous.requests.map(request => ({ ...request, blocks: [...request.blocks] })),
  };
  const config = state.settings;
  const iteration: SchedulerIteration = {
    tick: state.tick, admitted: [], work: [], released: [], usedBudget: 0,
    memoryDuring: [], memoryAfter: [], waits: [], stalled: false,
  };
  const active = () => state.requests.filter(request => ['prefill', 'decode'].includes(request.phase));
  if (state.cohort.every(id => state.requests.find(request => request.id === id)!.phase === 'finished'))
    state.cohort = [];
  if (config.admission === 'continuous' || state.cohort.length === 0) {
    const waiting = state.requests.map((request, order) => ({ request, order }))
      .filter(({ request }) => request.phase === 'queued' && request.arrival <= state.tick)
      .sort((a, b) => a.request.arrival - b.request.arrival || a.order - b.order);
    for (const { request } of waiting) {
      if (active().length >= config.maxRunning) break;
      request.phase = 'prefill';
      request.admittedAt = state.tick;
      state.cohort.push(request.id);
      iteration.admitted.push(request.id);
    }
  }

  const remainingBudget = () => config.tokenBudget - iteration.usedBudget;
  const reserve = (request: RequestState, tokenCount: number) => {
    const needed = Math.ceil(tokenCount / config.blockSize) - request.blocks.length;
    const occupied = new Set(state.requests.flatMap(row => row.blocks));
    const free = Array.from({ length: config.totalBlocks }, (_, i) => i).filter(i => !occupied.has(i));
    if (free.length < needed) return false;
    request.blocks.push(...free.slice(0, needed));
    return true;
  };
  const emit = (request: RequestState) => {
    request.emitted++;
    request.firstTokenAt ??= state.tick + 1;
    if (request.emitted === request.output) {
      request.phase = 'finished';
      request.finishedAt = state.tick + 1;
      iteration.released.push(request.id);
    } else request.phase = 'decode';
  };

  // Classify participants before executing the iteration: a request cannot
  // finish prefill and consume its first output in that same forward pass.
  const decoding = active().filter(request => request.phase === 'decode');
  const prefilling = active().filter(request => request.phase === 'prefill');
  for (const request of decoding) {
    if (!remainingBudget()) {
      iteration.waits.push({ id: request.id, reason: 'token-budget' });
      continue;
    }
    if (!reserve(request, request.cached + 1)) {
      iteration.waits.push({ id: request.id, reason: 'kv-capacity' });
      continue;
    }
    request.cached++;
    iteration.usedBudget++;
    emit(request);
    iteration.work.push({ id: request.id, phase: 'decode', tokens: 1, emitted: 1 });
  }
  for (const request of prefilling) {
    const remaining = request.prompt - request.prefilled;
    let count = Math.min(remaining, remainingBudget());
    if (!count) {
      iteration.waits.push({ id: request.id, reason: 'token-budget' });
      continue;
    }
    if (!config.chunkedPrefill && remaining > remainingBudget()) {
      iteration.waits.push({ id: request.id, reason: 'whole-prompt' });
      continue;
    }
    if (config.chunkedPrefill) {
      const freeBlocks = config.totalBlocks - state.requests.reduce((n, row) => n + row.blocks.length, 0);
      const freePositions = (request.blocks.length + freeBlocks) * config.blockSize - request.cached;
      count = Math.min(count, freePositions);
    }
    if (!count || !reserve(request, request.cached + count)) {
      iteration.waits.push({ id: request.id, reason: 'kv-capacity' });
      continue;
    }
    request.prefilled += count;
    request.cached += count;
    iteration.usedBudget += count;
    const emitted = request.prefilled === request.prompt ? 1 : 0;
    if (emitted) emit(request);
    iteration.work.push({ id: request.id, phase: 'prefill', tokens: count, emitted });
  }
  // A forward pass needs every participant's state until it ends. Only then
  // may another request reuse the blocks of a completed participant.
  iteration.memoryDuring = cacheBlocks(state);
  state.peakBlocks = Math.max(state.peakBlocks,
    iteration.memoryDuring.filter(block => block.owner !== null).length);
  for (const request of state.requests.filter(request => iteration.released.includes(request.id))) {
    request.blocks = [];
    request.cached = 0;
  }
  iteration.memoryAfter = cacheBlocks(state);
  iteration.stalled = iteration.work.length === 0 && !schedulerFinished(state)
    && !state.requests.some(request => request.phase === 'queued' && request.arrival > state.tick);
  state.tick++;
  state.history.push(iteration);
  return state;
}

export function runScheduler(initial: SchedulerState, maxSteps = 1024): SchedulerState {
  boundedInteger(maxSteps, 1, 4096, 'maxSteps');
  let state = initial;
  for (let i = 0; i < maxSteps && !schedulerFinished(state) && !schedulerStalled(state); i++)
    state = stepScheduler(state);
  return state;
}
