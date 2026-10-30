"""Real loopback HTTP/NDJSON checks, plus one ASGI send-failure injection.

The slow wrapper adds 10 ms per real neural call solely to observe streaming;
this is correctness evidence, not a latency or throughput benchmark.
"""
from contextlib import contextmanager
from pathlib import Path
import asyncio
import hashlib
import json
import socket
import sys
import threading
import time

import fastapi
import httpx
import starlette
import torch
import transformers
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / '06 Практика/Code'
sys.path.insert(0, str(CODE))
from edls_http_reference import create_app, tiny_engine
from edls_kv_reference import uncached_greedy


class ObservableEngine:
    def __init__(self, base):
        self.base, self.calls = base, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _run(self, phase, requests):
        self.calls.append((phase, [r.request_id for r in requests]))
        time.sleep(.01)  # injected protocol-test delay, never benchmarked
        if any(r.prompt_ids[0] == 63 for r in requests):
            raise RuntimeError('injected model-call failure')
        return getattr(self.base, phase)(requests)

    def prefill(self, requests):
        return self._run('prefill', requests)

    def decode(self, requests):
        return self._run('decode', requests)


@contextmanager
def serving(app, expect_failure=False):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, loop='asyncio', lifespan='on', log_level='critical',
                            timeout_graceful_shutdown=.5)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.005)
    try:
        if expect_failure:
            assert not server.started and not thread.is_alive()
        else:
            assert server.started and thread.is_alive(), 'owned loopback server did not start'
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        thread.join(timeout=4)
        sock.close()
        assert not thread.is_alive(), 'owned server did not stop'


async def server_state(service):
    async def inspect():
        return dict(live=len(service.live), active=len(service.scheduler.active),
                    waiting=len(service.scheduler.waiting), receipts=len(service.scheduler.receipts),
                    requests=len(service.scheduler.requests), reservations=len(service.scheduler.reservations))
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(inspect(), service.loop))


async def quiescent(service):
    for _ in range(100):
        state = await server_state(service)
        if all(value == 0 for value in state.values()):
            return state
        await asyncio.sleep(.01)
    raise AssertionError(f'live state was not released: {state}')


def body(rid, output=5, **kwargs):
    return dict(request_id=rid, prompt_ids=[1, 7, 8, 9, 10], max_new_tokens=output, **kwargs)


async def read_stream(client, payload):
    events, received = [], []
    async with client.stream('POST', '/generate/stream', json=payload) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('application/x-ndjson')
        async for line in response.aiter_lines():
            if line:
                events.append(json.loads(line))
                received.append(time.perf_counter())
    assert events[0]['event'] == 'accepted'
    assert events[-1]['event'] == 'terminal'
    assert sum(e['event'] == 'terminal' for e in events) == 1
    assert all(e['request_id'] == payload['request_id'] for e in events)
    tokens = [e for e in events if e['event'] == 'token']
    assert [e['index'] for e in tokens] == list(range(len(tokens)))
    return events, received


async def normal_checks(app, url, engine, reference):
    service = app.state.service
    records = {}
    async with httpx.AsyncClient(base_url=url, timeout=2.) as client:
        assert (await client.get('/health/ready')).json() == {'ready': True}
        events, received = await read_stream(client, body('external-947', 16))
        tokens = [e['token_id'] for e in events if e['event'] == 'token']
        assert tokens == reference(16)
        first = next(i for i, e in enumerate(events) if e['event'] == 'token')
        assert received[first] < events[-1]['server_time'], 'response was buffered until completion'
        records['stream'] = dict(events=events, client_received=received,
                                 first_token_before_engine_terminal=True)
        await quiescent(service)

        # Concurrent clients preserve external IDs and token histories.
        results = await asyncio.gather(*(read_stream(client, body(f'concurrent-{n}', n))
                                         for n in [1, 3, 5]))
        for n, (ee, _) in zip([1, 3, 5], results):
            assert [e['token_id'] for e in ee if e['event'] == 'token'] == reference(n)
        records['concurrent_outputs_match'] = True
        await quiescent(service)

        before = len(engine.calls)
        zero, _ = await read_stream(client, body('zero', 0))
        assert [e['event'] for e in zero] == ['accepted', 'terminal']
        assert len(engine.calls) == before
        eos, _ = await read_stream(client, body('eos', 8, eos_token_id=reference(1)[0]))
        assert len([e for e in eos if e['event'] == 'token']) == 1
        assert eos[-1]['reason'] == 'eos'
        await quiescent(service)

        response = await client.post('/generate', json=body('final-json', 3))
        assert response.status_code == 200 and response.json()['generated_tokens'] == reference(3)
        records['final_json'] = response.json()
        await quiescent(service)

        for override in [dict(max_new_tokens=-1), dict(max_new_tokens=33),
                         dict(prompt_ids=[]), dict(prompt_ids=[64]), dict(prompt_ids=[-1]),
                         dict(request_id=''), dict(timeout_s=0.)]:
            assert (await client.post('/generate', json={**body('bad'), **override})).status_code == 422
        assert (await client.delete('/requests/missing')).status_code == 404
        await quiescent(service)

        async with client.stream('POST', '/generate/stream', json=body('cancel-live', 32)) as response:
            lines = response.aiter_lines()
            assert json.loads(await anext(lines))['event'] == 'accepted'
            assert (await client.post('/generate', json=body('cancel-live'))).status_code == 409
            acknowledgement = await client.delete('/requests/cancel-live')
            assert acknowledgement.status_code == 202 and acknowledgement.json()['cancel_queued']
            rest = [json.loads(line) async for line in lines if line]
            assert rest[-1]['status'] == 'cancelled' and rest[-1]['reason'] == 'explicit'
            assert sum(e['event'] == 'terminal' for e in rest) == 1
        records['cancel'] = rest
        await quiescent(service)
        pending = asyncio.create_task(client.post('/generate', json=body('cancel-json', 32)))
        await asyncio.sleep(.02)
        acknowledgement = await client.delete('/requests/cancel-json')
        assert acknowledgement.status_code == 202 and acknowledgement.json()['cancel_queued']
        cancelled_json = await pending
        assert cancelled_json.status_code == 409 and cancelled_json.json()['outcome']['reason'] == 'explicit'
        records['cancel_json'] = cancelled_json.json()
        await quiescent(service)

        timeout, _ = await read_stream(client, body('deadline', 8, timeout_s=.002))
        assert timeout[-1]['status'] == 'cancelled' and timeout[-1]['reason'] == 'timeout'
        response = await client.post('/generate', json=body('json-deadline', 8, timeout_s=.002))
        assert response.status_code == 408 and response.json()['outcome']['reason'] == 'timeout'
        records['deadline'] = timeout
        await quiescent(service)
        late, _ = await read_stream(client, body('deadline-one-token', 1, timeout_s=.002))
        assert late[-1]['status'] == 'cancelled' and late[-1]['reason'] == 'timeout'
        assert late[-1]['engine_reason'] == 'length'
        assert not any(e['event'] == 'token' for e in late)
        records['completion_after_client_deadline'] = late
        await quiescent(service)

        failure, _ = await read_stream(client, {**body('model-failure'), 'prompt_ids':[63]})
        assert failure[-1]['status'] == 'error'
        assert failure[-1]['reason'] == 'RuntimeError'
        response = await client.post('/generate', json={**body('json-failure'), 'prompt_ids':[63]})
        assert response.status_code == 500 and response.json()['outcome']['status'] == 'error'
        assert (await client.get('/health/ready')).status_code == 200
        records['stream_http_200_can_end_in_error'] = failure
        await quiescent(service)

        # Abandon a live stream; model work must stop before its full 32-token budget.
        before = len(engine.calls)
        async with client.stream('POST', '/generate/stream', json=body('disconnect', 32)) as response:
            async for line in response.aiter_lines():
                if line and json.loads(line)['event'] == 'token':
                    break
        await quiescent(service)
        calls_after = len(engine.calls)
        await asyncio.sleep(.03)
        assert len(engine.calls) == calls_after and calls_after-before < 32
        records['disconnect_model_calls'] = calls_after-before

        # Reuse after full cleanup is a new request, not cached/idempotent retry.
        again, _ = await read_stream(client, body('external-947', 2))
        assert [e['token_id'] for e in again if e['event'] == 'token'] == reference(2)
        await quiescent(service)

        # Non-streaming peer disconnect: no complete JSON response is read.
        port = int(url.rsplit(':', 1)[1])
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        encoded = json.dumps(body('json-disconnect', 32)).encode()
        writer.write(b'POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '
                     +str(len(encoded)).encode()+b'\r\n\r\n'+encoded)
        await writer.drain()
        await asyncio.sleep(.02)
        writer.close()
        await writer.wait_closed()
        records['nonstream_disconnect_cleanup'] = await quiescent(service)

        # No iterator has started if sending response-start fails.
        async def send_failure():
            payload = json.dumps(body('send-failure', 32)).encode()
            sent_body = False
            observed_statuses = []
            async def receive():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {'type':'http.request', 'body':payload, 'more_body':False}
                await asyncio.sleep(10)
                return {'type':'http.disconnect'}
            async def send(message):
                if message['type'] == 'http.response.start':
                    observed_statuses.append(message['status'])
                    raise OSError('injected response-start failure')
            scope = dict(type='http', asgi={'version':'3.0','spec_version':'2.4'},
                http_version='1.1', scheme='http', method='POST', path='/generate/stream',
                raw_path=b'/generate/stream', query_string=b'', root_path='',
                headers=[(b'content-type',b'application/json')], server=('127.0.0.1',port),
                client=('127.0.0.1',1))
            try:
                await app(scope, receive, send)
            except Exception:
                pass
            return observed_statuses
        send_statuses = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(send_failure(), service.loop))
        assert send_statuses and send_statuses[0] == 200, 'failure was not injected at a valid stream start'
        records['response_start_failure_statuses'] = send_statuses
        records['response_start_failure_cleanup'] = await quiescent(service)

        # Stop the owned model loop while a response is live.
        async with client.stream('POST', '/generate/stream', json=body('shutdown', 32)) as response:
            lines = response.aiter_lines()
            assert json.loads(await anext(lines))['event'] == 'accepted'
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(service.stop(), service.loop))
            rest = [json.loads(line) async for line in lines if line]
            assert rest[-1]['status'] == 'cancelled' and rest[-1]['reason'] == 'shutdown'
        assert (await client.get('/health/ready')).status_code == 503
        assert (await client.post('/generate', json=body('after-stop'))).status_code == 503
        records['shutdown_cleanup'] = await quiescent(service)
    return records


async def limit_check(app, url):
    async with httpx.AsyncClient(base_url=url, timeout=2.) as client:
        async with client.stream('POST', '/generate/stream', json=body('held',32)) as response:
            lines = response.aiter_lines()
            assert json.loads(await anext(lines))['event'] == 'accepted'
            assert (await client.post('/generate', json=body('too-many'))).status_code == 429
        return await quiescent(app.state.service)


async def fatal_check(app, url):
    service = app.state.service
    async def break_step():
        def fail():
            raise RuntimeError('injected scheduler-loop failure')
        service.scheduler.step = fail
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(break_step(), service.loop))
    async with httpx.AsyncClient(base_url=url, timeout=2.) as client:
        events, _ = await read_stream(client, body('fatal-loop'))
        assert events[-1]['status'] == 'error' and events[-1]['reason'] == 'model loop failed'
        assert (await client.get('/health/ready')).status_code == 503
        assert (await client.post('/generate', json=body('after-fatal'))).status_code == 503
        return dict(events=events, cleanup=await quiescent(service))


if __name__ == '__main__':
    engine = ObservableEngine(tiny_engine())
    expected = {n:uncached_greedy(engine.model,(1,7,8,9,10),n) for n in [1,2,3,5,16]}
    app = create_app(lambda:engine)
    with serving(app) as url:
        records = asyncio.run(normal_checks(app,url,engine,lambda n:expected[n]))
    limited = create_app(lambda:engine,limit=1)
    with serving(limited) as url:
        records['limit_cleanup'] = asyncio.run(limit_check(limited,url))
    fatal = create_app(lambda:engine)
    with serving(fatal) as url:
        records['fatal_loop'] = asyncio.run(fatal_check(fatal,url))
    def fail_start():
        raise RuntimeError('injected model-load failure')
    broken = create_app(fail_start)
    with serving(broken, expect_failure=True) as url:
        try:
            httpx.get(url+'/health/ready',timeout=.2)
        except httpx.TransportError:
            records['failed_start_served_ready'] = False
        else:
            raise AssertionError('failed model startup served traffic')
    def slow_start():
        time.sleep(.03)
        return engine
    timed_out = create_app(slow_start,startup_timeout=.001)
    with serving(timed_out, expect_failure=True) as url:
        try:
            httpx.get(url+'/health/ready',timeout=.2)
        except httpx.TransportError:
            records['timed_out_start_served_ready'] = False
        else:
            raise AssertionError('timed-out model startup served traffic')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    output = dict(scope='real loopback HTTP and NDJSON, random neural CPU; controlled protocol delay, not a benchmark',
        versions=dict(fastapi=fastapi.__version__,starlette=starlette.__version__,
                      uvicorn=uvicorn.__version__,httpx=httpx.__version__,
                      torch=torch.__version__,transformers=transformers.__version__),
        source_hashes={p.name:sha(p) for p in [CODE/'edls_http_reference.py',CODE/'edls_admission_reference.py',
                                             CODE/'edls_kv_reference.py',Path(__file__)]},records=records)
    print(json.dumps(output,indent=2))
    print('PASS real TCP HTTP/NDJSON, early token delivery, oracle IDs, terminal uniqueness, '
          'cancel/timeout/disconnect, capacity/duplicates, model/startup errors and cleanup',file=sys.stderr)
