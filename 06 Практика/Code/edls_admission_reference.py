"""EDLS Week 8 teaching scheduler around the separate KVGreedyEngine.

Adapted from the MIT course's scheduler/metric contract, commit
e632aa89ca9e6638d52e1b686095e7442faffbb0, Copyright (c) 2021 course authors.
See /licenses/edls-e632aa89-MIT.txt. Single consumer; no HTTP/paging/chunking.
The KV budget counts reserved logical positions, not peak physical bytes.
"""
from collections import deque
from math import isfinite
from time import perf_counter

from edls_kv_reference import Request


POLICIES = ('single', 'refill', 'wait_bound')


class AdmissionScheduler:
    def __init__(self, engine, policy='single', slots=2, kv_positions=256,
                 max_waiting=100, max_wait_s=.02, clock=perf_counter):
        if policy not in POLICIES:
            raise ValueError('unknown admission policy')
        if any(not isinstance(n, int) or n < 1 for n in [slots, kv_positions, max_waiting]):
            raise ValueError('positive integer capacities required')
        if not isfinite(max_wait_s) or max_wait_s < 0:
            raise ValueError('finite nonnegative wait bound required')
        self.engine, self.policy, self.clock = engine, policy, clock
        self.slots, self.kv_positions = slots, kv_positions
        self.max_waiting, self.max_wait_s = max_waiting, max_wait_s
        self.waiting, self.active = deque(), []
        self.requests, self.receipts, self.reservations = {}, {}, {}
        self.phases = []

    @property
    def has_work(self):
        return bool(self.waiting or self.active)

    def _reservation(self, request):
        return min(self.engine.max_context,
                   len(request.prompt_ids) + request.max_new_tokens - 1)

    def submit(self, request, submitted=None, scheduled=None):
        now = self.clock()
        submitted = now if submitted is None else submitted
        scheduled = submitted if scheduled is None else scheduled
        if not all(isfinite(t) for t in [now, submitted, scheduled]) or not scheduled <= submitted <= now:
            raise ValueError('scheduled <= submitted <= consumer time required')
        if request.request_id in self.requests:
            raise ValueError('duplicate external request ID')
        self.engine._validate([request])
        if request.is_finished or request.generated_tokens or request.past_key_values is not None:
            raise ValueError('fresh request state required')
        rid = request.request_id
        self.requests[rid] = request
        self.receipts[rid] = dict(id=rid, scheduled=scheduled, submitted=submitted,
                                 observed=now, admitted=None, finished=None,
                                 tokens=[], output_ids=[], status='waiting', reason=None)
        if request.max_new_tokens == 0:
            request.stop_reason = 'length'
            self._finish(request, 'completed', now, 'length')
        elif self._reservation(request) > self.kv_positions or len(self.waiting) >= self.max_waiting:
            self._finish(request, 'rejected', now, 'capacity')
        else:
            self.waiting.append(request)
        return rid

    def _finish(self, request, status, now, reason):
        receipt = self.receipts[request.request_id]
        if receipt['finished'] is not None:
            return
        receipt.update(status=status, finished=now, reason=reason)
        request.stop_reason = reason
        request.past_key_values = None
        self.reservations.pop(request.request_id, None)

    def cancel(self, request_id):
        request = self.requests[request_id]
        if self.receipts[request_id]['finished'] is not None:
            return False
        self._finish(request, 'cancelled', self.clock(), 'cancelled')
        self.waiting = deque(r for r in self.waiting if r is not request)
        self.active = [r for r in self.active if r is not request]
        return True

    def _decide_prefill_batch_size(self):
        free_slots = self.slots - len(self.active)
        if not self.waiting or free_slots <= 0:
            return 0
        if self.policy == 'single' and self.active:
            return 0
        age = self.clock() - self.receipts[self.waiting[0].request_id]['submitted']
        if self.policy == 'wait_bound' and self.active and age < self.max_wait_s:
            return 0
        limit = 1 if self.policy == 'single' else free_slots
        free_positions = self.kv_positions - sum(self.reservations.values())
        count = 0
        for request in self.waiting:
            cost = self._reservation(request)
            if count == limit or cost > free_positions:
                break  # FIFO: do not bypass a large head request
            count += 1
            free_positions -= cost
        return count

    def _phase(self, stage, batch):
        if not batch:
            return
        start = self.clock()
        event = dict(stage=stage, ids=[r.request_id for r in batch], start=start,
                     reserved_positions=sum(self.reservations.values()))
        try:
            result = getattr(self.engine, stage)(batch)
            expected = {r.request_id for r in batch}
            if (set(result.request_ids) != expected or len(result.request_ids) != len(expected)
                    or not len(result.request_ids) == len(result.new_tokens) == len(result.finished)
                    or any(len(tokens) != 1 for tokens in result.new_tokens)):
                raise ValueError('one token per participating request required')
            ended = self.clock()
            for rid, tokens in zip(result.request_ids, result.new_tokens):
                self.receipts[rid]['tokens'].append(ended)
                self.receipts[rid]['output_ids'].extend(tokens)
            for request in batch:
                if request.is_finished:
                    self._finish(request, 'completed', ended, request.stop_reason)
            event.update(end=ended, emitted=len(result.request_ids), error=None)
        except Exception as error:
            ended = self.clock()
            for request in batch:
                # Internal candidate tokens from an incomplete call were not published.
                request.generated_tokens[:] = self.receipts[request.request_id]['output_ids']
                self._finish(request, 'error', ended, type(error).__name__)
            event.update(end=ended, emitted=0, error=type(error).__name__)
        self.phases.append(event)
        self.active = [r for r in self.active if not r.is_finished]

    def step(self):
        count = self._decide_prefill_batch_size()
        batch = [self.waiting.popleft() for _ in range(count)]
        now = self.clock()
        for request in batch:
            self.receipts[request.request_id].update(admitted=now, status='active')
            self.reservations[request.request_id] = self._reservation(request)
        self.active.extend(batch)
        self._phase('prefill', batch)
        # Prefill-finished requests were removed; new unfinished requests can decode.
        self._phase('decode', self.active.copy())
