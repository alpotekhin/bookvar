"""Correctness plus optional bounded CPU neural admission experiment.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
 --with transformers==4.57.1 python publishing/tools/check_edls_admission_reference.py --experiment
JSON goes to stdout; diagnostics to stderr. No checkpoint, tokenizer or HTTP.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import io
import itertools
import json
import platform
import queue
import random
import re
import select
import subprocess
import sys
import threading
import time

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / '06 Практика/Code'
sys.path.insert(0, str(CODE))
from edls_kv_reference import Request, BatchResult, KVGreedyEngine, uncached_greedy
from edls_admission_reference import AdmissionScheduler, POLICIES

chapter = (ROOT / '06 Практика/14 Собрать минимальный inference engine.md').read_text()
receipt_fence = re.search(r'<!-- example: edls-metric-receipts -->\s*```python\n(.*?)\n```', chapter, re.S).group(1)
receipt_ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(receipt_fence, 'published receipt calculation', 'exec'), receipt_ns)
request_report = receipt_ns['request_report']


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


class StateEngine:
    """Deterministic state tests, explicitly not neural timing."""
    max_context = 128

    def __init__(self, clock):
        self.clock, self.calls, self.fail = clock, [], False

    def _validate(self, requests):
        if any(not r.prompt_ids or r.max_new_tokens < 0 for r in requests):
            raise ValueError('invalid fake-engine input')

    def _call(self, stage, requests):
        self.calls.append((stage, [r.request_id for r in requests]))
        self.clock.t += .01
        if self.fail:
            requests[0].generated_tokens.append(999)
            raise RuntimeError('injected failure after an internal candidate')
        tokens = []
        for request in requests:
            token = len(request.generated_tokens) + 10
            request.generated_tokens.append(token)
            request.past_key_values = object()
            if len(request.generated_tokens) == request.max_new_tokens:
                request.stop_reason = 'length'
            tokens.append([token])
        return BatchResult([r.request_id for r in requests], tokens,
                           [r.is_finished for r in requests])

    def prefill(self, requests):
        return self._call('prefill', requests)

    def decode(self, requests):
        return self._call('decode', requests)


def check_invariants(scheduler):
    assert len(scheduler.active) <= scheduler.slots
    assert sum(scheduler.reservations.values()) <= scheduler.kv_positions
    assert set(scheduler.reservations) == {r.request_id for r in scheduler.active}
    for rid, receipt in scheduler.receipts.items():
        request = scheduler.requests[rid]
        assert len(receipt['tokens']) == len(receipt['output_ids'])
        assert receipt['output_ids'] == request.generated_tokens
        if receipt['finished'] is not None:
            assert request.past_key_values is None and rid not in scheduler.reservations
            assert request.is_finished


def drain(scheduler):
    for _ in range(1000):
        check_invariants(scheduler)
        if not scheduler.has_work:
            return
        scheduler.step()
    raise AssertionError('scheduler did not drain')


def correctness():
    # Direct decision cases: identical capacity/state, different admissions.
    for policy in POLICIES:
        clock = Clock()
        scheduler = AdmissionScheduler(StateEngine(clock), policy, slots=2, clock=clock)
        scheduler.submit(Request('external-A', (1, 2), 6))
        scheduler.step()
        scheduler.submit(Request('external-B', (1, 3), 3))
        assert scheduler._decide_prefill_batch_size() == (1 if policy == 'refill' else 0)
        clock.t += .02
        assert scheduler._decide_prefill_batch_size() == (0 if policy == 'single' else 1)
        drain(scheduler)
        assert all(r['status'] == 'completed' for r in scheduler.receipts.values())
        request_report(list(scheduler.receipts.values()), 0, clock() + .001)

    # Arrival ordering, slot limits and lengths; no lost IDs or late emissions.
    cases = 0
    for policy, slots, order in itertools.product(POLICIES, [1, 2, 3], itertools.permutations(range(3))):
        clock = Clock()
        scheduler = AdmissionScheduler(StateEngine(clock), policy, slots=slots, clock=clock)
        for i in order:
            scheduler.submit(Request(f'ID-{i+900}', (1, i+2), i+1))
        drain(scheduler)
        assert set(scheduler.receipts) == {f'ID-{i+900}' for i in range(3)}
        for i in range(3):
            receipt = scheduler.receipts[f'ID-{i+900}']
            assert receipt['output_ids'] == list(range(10, 10+i+1))
        count = len(scheduler.phases)
        scheduler.step()
        assert len(scheduler.phases) == count
        cases += 1

    clock = Clock()
    engine = StateEngine(clock)
    scheduler = AdmissionScheduler(engine, slots=2, kv_positions=8, max_waiting=1, clock=clock)
    scheduler.submit(Request('zero', (1,), 0))
    assert scheduler.receipts['zero']['status'] == 'completed' and engine.calls == []
    scheduler.submit(Request('large', (1,)*9, 1))
    assert scheduler.receipts['large']['status'] == 'rejected'
    scheduler.submit(Request('queued', (1,), 5))
    scheduler.submit(Request('overflow', (1,), 2))
    assert scheduler.receipts['overflow']['status'] == 'rejected'
    assert scheduler.cancel('queued') and not scheduler.cancel('queued')
    scheduler.submit(Request('active', (1,), 5))
    scheduler.step()
    published = list(scheduler.receipts['active']['output_ids'])
    assert scheduler.cancel('active')
    scheduler.step()
    assert scheduler.receipts['active']['output_ids'] == published
    check_invariants(scheduler)

    # Reservation boundary/FIFO: the second request waits until the first retires.
    scheduler = AdmissionScheduler(StateEngine(clock), 'refill', slots=2, kv_positions=6, clock=clock)
    scheduler.submit(Request('head', (1, 2, 3), 3))  # reserve 5
    scheduler.submit(Request('next', (1,), 3))       # reserve 3
    assert scheduler._decide_prefill_batch_size() == 1
    drain(scheduler)
    assert scheduler.receipts['next']['admitted'] >= scheduler.receipts['head']['finished']

    broken = StateEngine(clock)
    scheduler = AdmissionScheduler(broken, 'refill', clock=clock)
    scheduler.submit(Request('broken', (1,), 5))
    broken.fail = True
    scheduler.step()
    assert scheduler.receipts['broken']['status'] == 'error'
    assert scheduler.requests['broken'].generated_tokens == []
    assert not scheduler.has_work
    check_invariants(scheduler)
    broken.fail = False
    scheduler.submit(Request('after-error', (1,), 2))
    drain(scheduler)
    assert scheduler.receipts['after-error']['status'] == 'completed'

    for kwargs in [dict(policy='unknown'), dict(slots=0), dict(kv_positions=0),
                   dict(max_waiting=0), dict(max_wait_s=-1)]:
        try:
            AdmissionScheduler(StateEngine(clock), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid config accepted')
    try:
        scheduler.submit(Request('after-error', (1,), 2))
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate ID accepted')
    print(f'PASS {cases} deterministic admission cases; IDs, FIFO/reservations, '
          'zero/one budgets, cancellation/rejection/errors and release', file=sys.stderr)


def make_model():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                        attention_dropout=0., bos_token_id=1, eos_token_id=2,
                        pad_token_id=0, use_sliding_window=False)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).eval()


def manifest(kind):
    rng = random.Random(26)
    jobs = []
    for i in range(12):
        length = [24, 32, 40][i % 3] if kind == 'prefill_heavy' else [2, 3, 4][i % 3]
        output = 1 + i % 2 if kind == 'prefill_heavy' else 10 + i % 3 * 3
        jobs.append(dict(id=f'{kind}-{i:02}', offset_s=i*.002,
                         prompt_ids=[1]+[rng.randrange(3, 64) for _ in range(length-1)],
                         max_new_tokens=output))
    return jobs


PRODUCER_CODE = '''
import json, sys, time
jobs = json.loads(sys.stdin.readline())
print('READY', flush=True)
origin = float(sys.stdin.readline())
for index, job in enumerate(jobs):
    time.sleep(max(0., origin + job['offset_s'] - time.perf_counter()))
    submitted = time.perf_counter() - origin
    print(json.dumps(dict(index=index, submitted=submitted)), flush=True)
'''


def real_run(engine, jobs, policy, slots):
    incoming = queue.Queue()
    done = threading.Event()
    producer_errors = []
    process = subprocess.Popen([sys.executable, '-u', '-c', PRODUCER_CODE],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1)
    process.stdin.write(json.dumps(jobs)+'\n')
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 2)[0] or process.stdout.readline().strip() != 'READY':
        process.kill()
        process.wait()
        raise RuntimeError('producer did not become ready')
    origin = time.perf_counter() + .01  # reader/consumer ready before target offset zero
    process.stdin.write(str(origin)+'\n')
    process.stdin.close()
    clock = lambda: time.perf_counter() - origin
    scheduler = AdmissionScheduler(engine, policy, slots=slots, kv_positions=256,
                                   max_wait_s=.02, clock=clock)

    def collect_submissions():
        try:
            for line in process.stdout:
                packet = json.loads(line)
                incoming.put((jobs[packet['index']], packet['submitted']))
        except Exception as error:
            producer_errors.append(repr(error))
        finally:
            done.set()

    producer = threading.Thread(target=collect_submissions)
    producer.start()
    try:
        while not done.is_set() or not incoming.empty() or scheduler.has_work:
            if clock() > 10:
                raise TimeoutError('bounded CPU fixture exceeded 10 seconds')
            while True:
                try:
                    job, submitted = incoming.get_nowait()
                except queue.Empty:
                    break
                scheduler.submit(Request(job['id'], tuple(job['prompt_ids']), job['max_new_tokens']),
                                 submitted=submitted, scheduled=job['offset_s'])
            if scheduler.has_work:
                scheduler.step()
                check_invariants(scheduler)
            elif not done.is_set():
                time.sleep(.0002)
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        producer.join(timeout=2)
        if producer.is_alive():
            raise RuntimeError('submission reader did not terminate')
    if process.returncode != 0 or producer_errors:
        raise RuntimeError(f'producer failed: {process.stderr.read()} {producer_errors}')
    ended = clock()
    receipts = list(scheduler.receipts.values())
    assert set(scheduler.receipts) == {job['id'] for job in jobs}
    assert all(r['status'] == 'completed' for r in receipts)
    rows, totals = request_report(receipts, 0, ended)
    return dict(policy=policy, slots=slots, duration_s=ended,
                receipts=receipts, phases=scheduler.phases, metrics=totals,
                max_ttft_s=max(r['ttft'] for r in rows),
                max_itl_s=max((dt for r in rows for dt in r['itls']), default=None),
                max_submission_lateness_s=max(r['submitted']-r['scheduled'] for r in receipts),
                max_consumer_delay_s=max(r['observed']-r['submitted'] for r in receipts))


def experiment():
    model = make_model()
    engine = KVGreedyEngine(model)
    # Real neural termination/release checks, separate from timed runs.
    prompt = (1, 7, 8, 9, 10)
    first_two = uncached_greedy(model, prompt, 2)
    assert first_two[0] != first_two[1]
    for policy in POLICIES:
        for index in [0, 1]:
            scheduler = AdmissionScheduler(engine, policy)
            rid = f'eos-{index}'
            scheduler.submit(Request(rid, prompt, 10, eos_token_id=first_two[index]))
            drain(scheduler)
            assert scheduler.receipts[rid]['output_ids'] == first_two[:index+1]
            assert scheduler.receipts[rid]['reason'] == 'eos'
        limited = AdmissionScheduler(KVGreedyEngine(model, max_context=4), policy)
        limited.submit(Request('context', (1, 7), 10))
        drain(limited)
        assert limited.receipts['context']['reason'] == 'context'
        assert limited.receipts['context']['output_ids'] == uncached_greedy(model, (1, 7), 3)
        scheduler = AdmissionScheduler(engine, policy)
        scheduler.submit(Request('zero', (1, 7), 0))
        scheduler.submit(Request('cancel', prompt, 10))
        scheduler.step()
        scheduler.cancel('cancel')
        count = len(scheduler.receipts['cancel']['tokens'])
        scheduler.step()
        assert scheduler.receipts['zero']['tokens'] == []
        assert len(scheduler.receipts['cancel']['tokens']) == count
        check_invariants(scheduler)
    print('PASS 15 real neural terminal cases: EOS first/second, context, zero and cancellation', file=sys.stderr)
    workloads = {name: manifest(name) for name in ['prefill_heavy', 'decode_heavy']}
    expected = {job['id']: uncached_greedy(model, tuple(job['prompt_ids']), job['max_new_tokens'])
                for jobs in workloads.values() for job in jobs}
    for batch in [1, 2, 4]:
        requests = [Request(f'warm-{i}', (1,)*40, 3) for i in range(batch)]
        engine.prefill(requests)
        engine.decode(requests)
    runs = []
    rng = random.Random(17)
    for name, jobs in workloads.items():
        for slots in [1, 2, 4]:
            for repeat in range(3):
                order = list(POLICIES)
                rng.shuffle(order)
                for policy in order:
                    run = real_run(engine, jobs, policy, slots)
                    run.update(workload=name, repeat=repeat)
                    for receipt in run['receipts']:
                        assert receipt['output_ids'] == expected[receipt['id']]
                    runs.append(run)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    result = dict(
        scope='CPU random Qwen3, separate stdlib producer process, in-process consumer; no HTTP/GPU/quality claim',
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
                     python=platform.python_version(), os=platform.system(), machine=platform.machine(),
                     device='cpu', dtype=str(next(model.parameters()).dtype),
                     threads=torch.get_num_threads(), parameters=sum(p.numel() for p in model.parameters()),
                     clock=vars(time.get_clock_info('perf_counter')),
                     model_seed=8, manifest_seed=26, order_seed=17),
        limits=dict(kv_reserved_positions=256, wait_threshold_s=.02, max_waiting=100),
        source_hashes={path.name: sha(path.read_bytes()) for path in
                       [CODE/'edls_kv_reference.py', CODE/'edls_admission_reference.py', Path(__file__)]},
        receipt_fence_sha256=sha(receipt_fence.encode()), workloads=workloads, runs=runs)
    print(f'PASS {len(runs)} actual CPU runs; every output matches independent uncached greedy', file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    correctness()
    if '--experiment' in sys.argv:
        experiment()
