---
title: HTTP streaming и жизненный цикл запроса
type: practice
status: canonical
last_updated: 2026-10-26
last_verified: 2026-10-26
source_language: en
---

# HTTP streaming и жизненный цикл запроса

A client can close its connection while the model still owns the request's
KV-cache. Removing the HTTP response object does not remove that request from
the decode batch. The API has to pass cancellation to the owner of the model
state, then release the delivery channel after the request stops.

This continues
[[02 Areas/ML & DL/06 Практика/14 Собрать минимальный inference engine|the neural engine and admission practice]].
The original assignment is
[[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|EDLS Week 8 homework]].
Its [complete API source](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/edlang/server/api_server.py)
uses a separate scheduler process and returns completed JSON. The reference
below uses one model-state owner, offloads each synchronous step to a thread,
and delivers bounded token-ID streams. The code is an attributed solution,
not an unchanged copy of a finished course server.

## Separate the HTTP interface from model-state ownership

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/http-lifecycle-2026-10-26/vllm-llm-engine.png]]

*vLLM contributors, [Architecture Overview](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/arch_overview.md#llm-engine),
v0.11.0, [Apache-2.0](/licenses/vllm-b8b302cd-Apache-2.0.txt).
Original LLMEngine/AsyncLLMEngine interface diagram, not current V1 process placement.*

The source describes the engine's responsibilities as follows:

<!-- BEGIN SOURCE: vLLM engine responsibilities -->
The `LLMEngine` class is the core component of the vLLM engine. It is
responsible for receiving requests from clients and generating outputs from the
model. The `LLMEngine` includes input processing, model execution (possibly
distributed across multiple hosts and/or GPUs), scheduling, and output
processing.
<!-- END SOURCE: vLLM engine responsibilities -->

*Original English passage, same pinned overview and Apache-2.0 license.*

In this exercise the caller supplies token IDs, so no tokenizer runs in the
input path. Scheduling and cache ownership come from practice 14; the model
runs one synchronous step at a time. The asynchronous API loop handles
connections and delivery while that step executes elsewhere. A second HTTP
handler must not mutate the scheduler's active list while the model thread
is packing its cache batch.

The `Service` therefore accepts commands through one queue. Only its owner
loop submits and cancels requests. While it awaits an offloaded model step,
handlers may enqueue commands but cannot change model state. After each
prefill/decode call, the model thread copies immutable output payloads and
uses [call_soon_threadsafe](https://docs.python.org/3.14/library/asyncio-dev.html#concurrency-and-multithreading)
to hand them back to the API loop. This lets a
prefill token reach a client while later decode calls are still running.

The production-engine discussion is in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines]].
This example does not reproduce vLLM's workers, IPC, paging or distributed
execution.

## A token event is not an HTTP success code

Send a three-token request with ID `demo-1`, prompt IDs
`[1, 7, 8, 9, 10]` and no EOS stop. The checked random model returns
`32, 27, 34`. The client receives these event types:

| Event | Fields to inspect | What has happened |
|---|---|---|
| accepted | request_id | The API registered a delivery channel; prefill need not have started |
| token | request_id, index 0, token_id 32, server_time | The first token became available after prefill |
| token | index 1, token_id 27 | A later model call produced the next token |
| token | index 2, token_id 34 | The configured output budget was reached |
| terminal | status completed, reason length | The response stream has reached its final outcome |

Every event retains the caller's ID. A token's index is consecutive within that
request, starting at zero. `server_time` records server availability on its
monotonic clock. The client records its own receipt times; those include API
scheduling, transport and parsing. Packets can contain several JSON lines,
so reading two tokens together does not prove that the model produced them
at the same instant.

A streaming response sends HTTP 200 before generation has necessarily
succeeded. If a later model call fails, that status cannot be changed to 500;
the stream ends with an error terminal event. Read the terminal outcome.
The non-streaming `/generate` endpoint instead waits for the final result,
so it can return 500 for an error or 408 for its request timeout.

For client TTFT, skip `accepted` and time the first `token` event. Timing the
first response byte would measure the control acknowledgement in this protocol.

This token-ID/NDJSON interface is not OpenAI-compatible or wire-compatible
with the original course benchmark, which sends prompt strings.

## Cancellation and completion can race

`DELETE /requests/demo-1` returns 202 when it queues a cancellation command.
It does not claim that an in-flight forward has already stopped. The command
is applied at the next scheduler-step boundary. Once the request is removed,
later steps cannot emit tokens for it and its cache reference is released.

A timeout can expire during a forward that finishes the model's output budget.
The API may then return a cancelled/timeout outcome while `engine_reason`
is `length`: the computation finished, but the API stopped delivering its
result. The checker exercises this case with a one-token budget and verifies
that no late token is delivered. An already published terminal event cannot be
overwritten by a later cancel or timeout.

Connection loss follows the same cancellation path. The response wrapper
closes the channel in `finally`, even if sending response headers fails before
the generator begins iterating. The JSON endpoint also polls for disconnection
while waiting for its result. Closing a consumer is idempotent; it removes a
live-ID entry only when that entry is the same channel and the model owner has
finished with it. A delayed finalizer must not remove a new request that reused
the same string ID.

IDs are unique among live requests. Reuse after complete cleanup creates a
new request; this example does not implement idempotent retries or a result
cache.

## Why the original pending-future collector needs more lifecycle handling

The course source contains this complete collector:

<!-- BEGIN SOURCE: EDLS response collector -->
```python
async def _response_collector(self):
    while True:
        if not self.response_queue.empty():
            response = self.response_queue.get()

            if response.request_id in self.pending_requests:
                self.pending_requests[response.request_id].set_result(response)
            else:
                print(f"Response for request_id {response.request_id} not found")

        await asyncio.sleep(0.01)
```
<!-- END SOURCE: EDLS response collector -->

*EDLS Week 8 `api_server.py`, original method with indentation dedented,
commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`,
[MIT license](/licenses/edls-e632aa89-MIT.txt).*

A future stores one pending result for the handler awaiting it. The method
sets that result, but does not remove the future from `pending_requests`.
A cancelled future no longer accepts a value: setting its result raises
`InvalidStateError`. If the worker dies before responding, no result
arrives to wake the waiter. Independently allocated API/scheduler ID counters
also depend on the same successful registration sequence.

The reference uses a bounded per-request event queue and one terminal flag
instead. A model-loop failure publishes an error outcome to waiting channels
and makes readiness fail. Cleanup removes the request, receipt, reservation
and live delivery entry; it does not retain an ever-growing dictionary of
completed futures.

## Complete runnable reference

Save this module beside the two files from practice 14. All three downloads
are available: [HTTP layer](/examples/edls_http_reference.py),
[admission scheduler](/examples/edls_admission_reference.py) and
[neural core](/examples/edls_kv_reference.py).
The repository path is `06 Практика/Code/edls_http_reference.py`.

<!-- BEGIN SOLUTION: EDLS HTTP module -->
```python
"""Loopback teaching API for the EDLS Week 8 request lifecycle.

Adapted from MIT course contracts, commit e632aa89ca9e6638d52e1b686095e7442faffbb0.
Copyright (c) 2021 course authors; /licenses/edls-e632aa89-MIT.txt.
Token IDs and bounded NDJSON, not OpenAI compatibility or a production service.
"""
import asyncio
from contextlib import asynccontextmanager, aclosing
from dataclasses import dataclass
import json
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from edls_kv_reference import Request, KVGreedyEngine
from edls_admission_reference import AdmissionScheduler


class Generate(BaseModel):
    model_config = ConfigDict(strict=True)
    request_id: str = Field(min_length=1, max_length=64)
    prompt_ids: list[int] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(default=8, ge=0, le=32)
    eos_token_id: int | None = None
    timeout_s: float = Field(default=5., gt=0, le=30)


@dataclass
class Channel:
    request: Request
    events: asyncio.Queue
    closed: bool = False
    terminal: bool = False
    worker_done: bool = False
    published: int = 0
    cancel_cause: str | None = None
    timer: asyncio.Task | None = None


class Service:
    def __init__(self, engine, limit=8):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError('positive live-request limit required')
        self.loop = asyncio.get_running_loop()
        self.live, self.ready = {}, True
        self.limit = limit
        self.commands = asyncio.Queue(maxsize=2*limit+1)
        service = self

        class ObservedScheduler(AdmissionScheduler):
            def _phase(self, stage, batch):
                super()._phase(stage, batch)
                # Copy immutable payloads before another phase mutates state.
                payload = service.snapshot(batch)
                service.loop.call_soon_threadsafe(service.observe, payload)

        self.scheduler = ObservedScheduler(engine, 'refill', slots=2,
                                            kv_positions=256, max_waiting=limit)
        self.task = asyncio.create_task(self.run())

    def snapshot(self, requests):
        result = []
        for request in requests:
            receipt = self.scheduler.receipts[request.request_id]
            result.append((request, tuple(receipt['output_ids']), tuple(receipt['tokens']),
                           receipt['status'], receipt['reason'], receipt['finished']))
        return tuple(result)

    def observe(self, payload):
        for request, ids, times, status, reason, finished in payload:
            channel = self.live.get(request.request_id)
            if channel is None or channel.request is not request or channel.terminal:
                continue
            if not channel.closed and channel.cancel_cause is None:
                for index in range(channel.published, len(ids)):
                    channel.events.put_nowait(dict(event='token', request_id=request.request_id,
                        index=index, token_id=ids[index], server_time=times[index]))
                channel.published = len(ids)
            if finished is not None:
                self.terminal(channel, 'cancelled' if channel.cancel_cause else status,
                              channel.cancel_cause or reason, finished, reason)

    def terminal(self, channel, status, reason, when, engine_reason=None):
        if channel.terminal:
            return
        channel.terminal = True
        if channel.timer:
            channel.timer.cancel()
        if not channel.closed:
            channel.events.put_nowait(dict(event='terminal', request_id=channel.request.request_id,
                status=status, reason=reason, engine_reason=engine_reason, server_time=when))

    def open(self, body):
        if not self.ready or self.task.done():
            raise HTTPException(503, 'model loop is not ready')
        if body.request_id in self.live:
            raise HTTPException(409, 'request ID is already live')
        if len(self.live) >= self.limit:
            raise HTTPException(429, 'live request limit reached')
        request = Request(body.request_id, tuple(body.prompt_ids), body.max_new_tokens,
                          eos_token_id=body.eos_token_id)
        try:
            self.scheduler.engine._validate([request])
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        channel = Channel(request, asyncio.Queue(maxsize=body.max_new_tokens+2))
        self.live[body.request_id] = channel
        channel.events.put_nowait(dict(event='accepted', request_id=body.request_id,
                                      server_time=perf_counter()))
        self.commands.put_nowait(('submit', channel))

        async def deadline():
            await asyncio.sleep(body.timeout_s)
            self.cancel(channel, 'timeout')

        channel.timer = asyncio.create_task(deadline())
        return channel

    def cancel(self, channel, cause):
        if channel.terminal or channel.cancel_cause is not None:
            return False
        channel.cancel_cause = cause
        self.commands.put_nowait(('cancel', channel))
        return True

    def close(self, channel):
        channel.closed = True
        self.cancel(channel, 'disconnect')
        if channel.timer:
            channel.timer.cancel()
        self.release(channel)

    def release(self, channel):
        rid = channel.request.request_id
        if channel.closed and channel.worker_done and self.live.get(rid) is channel:
            self.live.pop(rid)

    def collect(self):
        for rid, receipt in list(self.scheduler.receipts.items()):
            if receipt['finished'] is None:
                continue
            channel = self.live[rid]
            channel.worker_done = True
            self.scheduler.receipts.pop(rid)
            self.scheduler.requests.pop(rid)
            self.release(channel)

    def handle(self, kind, channel):
        request = channel.request
        if kind == 'submit':
            if channel.cancel_cause:
                channel.worker_done = True
                self.terminal(channel, 'cancelled', channel.cancel_cause, perf_counter())
                self.release(channel)
            else:
                self.scheduler.submit(request)
                self.observe(self.snapshot([request]))  # zero-output completion
        elif self.scheduler.requests.get(request.request_id) is request:
            self.scheduler.cancel(request.request_id)
            self.observe(self.snapshot([request]))

    async def run(self):
        try:
            stopping = False
            while not stopping:
                if self.commands.empty() and not self.scheduler.has_work:
                    command = await self.commands.get()
                    if command[0] == 'stop':
                        break
                    self.handle(*command)
                while not self.commands.empty():
                    kind, channel = self.commands.get_nowait()
                    if kind == 'stop':
                        stopping = True
                    else:
                        self.handle(kind, channel)
                if not stopping and self.scheduler.has_work:
                    # Only this owner submits/cancels; model work runs off-loop.
                    await asyncio.to_thread(self.scheduler.step)
                self.collect()
        except Exception:
            self.ready = False
            for channel in list(self.live.values()):
                request = channel.request
                if self.scheduler.requests.get(request.request_id) is request:
                    self.scheduler.cancel(request.request_id)
                self.terminal(channel, 'error', 'model loop failed', perf_counter())
                channel.worker_done = True
                self.release(channel)
            self.scheduler.active.clear()
            self.scheduler.waiting.clear()
            self.scheduler.requests.clear()
            self.scheduler.receipts.clear()
            self.scheduler.reservations.clear()
            raise
        finally:
            self.ready = False

    async def stop(self):
        if self.task.done():
            await self.task
            return
        self.ready = False
        for channel in list(self.live.values()):
            self.cancel(channel, 'shutdown')
        self.commands.put_nowait(('stop', None))
        await self.task

    async def events(self, channel, disconnected=None):
        try:
            while True:
                if disconnected and await disconnected():
                    return
                try:
                    event = await asyncio.wait_for(channel.events.get(), .02) if disconnected else await channel.events.get()
                except TimeoutError:
                    continue
                yield event
                if event['event'] == 'terminal':
                    break
        finally:
            self.close(channel)


def create_app(engine_factory, startup_timeout=5., limit=8):
    @asynccontextmanager
    async def lifespan(app):
        engine = await asyncio.wait_for(asyncio.to_thread(engine_factory), startup_timeout)
        app.state.service = Service(engine, limit=limit)
        try:
            yield
        finally:
            await app.state.service.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get('/health/ready')
    async def ready():
        service = app.state.service
        if not service.ready or service.task.done():
            raise HTTPException(503, 'model loop is not ready')
        return {'ready': True}

    @app.post('/generate/stream')
    async def stream(body: Generate):
        service = app.state.service
        channel = service.open(body)

        async def lines():
            async with aclosing(service.events(channel)) as events:
                async for event in events:
                    yield json.dumps(event)+'\n'

        class OwnedStream(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    try:
                        await self.body_iterator.aclose()
                    finally:
                        service.close(channel)  # also if response-start send fails

        return OwnedStream(lines(), media_type='application/x-ndjson',
                           headers={'Cache-Control': 'no-store'})

    @app.post('/generate')
    async def generate(body: Generate, request: HTTPRequest):
        service = app.state.service
        channel = service.open(body)
        tokens, terminal = [], None
        async for event in service.events(channel, request.is_disconnected):
            if event['event'] == 'token':
                tokens.append(event['token_id'])
            elif event['event'] == 'terminal':
                terminal = event
        if terminal is None:  # peer disconnected; events() already queued cleanup
            return Response(status_code=204)
        status = 200 if terminal['status'] == 'completed' else (
            408 if terminal['reason'] == 'timeout' else (
                409 if terminal['status'] == 'cancelled' else 500))
        return JSONResponse(dict(request_id=body.request_id, generated_tokens=tokens,
                                 outcome=terminal), status_code=status)

    @app.delete('/requests/{request_id}', status_code=202)
    async def cancel(request_id: str):
        service = app.state.service
        if request_id not in service.live:
            raise HTTPException(404, 'no live request with that ID')
        return {'cancel_queued': service.cancel(service.live[request_id], 'explicit')}

    return app


def tiny_engine():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(8)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, attention_dropout=0., bos_token_id=1,
        eos_token_id=2, pad_token_id=0, use_sliding_window=False)
    config._attn_implementation = 'eager'
    engine = KVGreedyEngine(Qwen3ForCausalLM(config))
    engine.prefill([Request('warmup', (1, 7), 1)])
    return engine


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(tiny_engine), host='127.0.0.1', port=4341)
```
<!-- END SOLUTION: EDLS HTTP module -->

Run it from the repository root:

```bash
uv run --with fastapi==0.121.3 --with starlette==0.50.0 \
  --with uvicorn==0.38.0 --with httpx==0.28.1 \
  --with torch==2.14.1 --with numpy==2.3.5 --with transformers==4.57.1 \
  python '06 Практика/Code/edls_http_reference.py'
```

It binds only `127.0.0.1:4341`, loads the same 22,720-parameter random Qwen3
and performs a warmup forward before accepting requests. No checkpoint or
tokenizer download is needed. In another terminal:

```bash
curl --no-buffer http://127.0.0.1:4341/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","prompt_ids":[1,7,8,9,10],"max_new_tokens":3}'
```

Use a longer output budget to inspect cancellation, then issue
`curl -X DELETE http://127.0.0.1:4341/requests/demo-1` while that ID is live.
A request that already finished and was cleaned up returns 404.
The tiny model can finish before a manually typed cancel reaches it. The
checker exercises cancellation programmatically with a controlled call delay.

## Readiness, errors and bounded state

FastAPI's [lifespan](https://fastapi.tiangolo.com/advanced/events/) initializes
the engine once before serving and closes the service after requests drain.
Readiness requires successful initialization/warmup and a live model-owner
task. A load failure prevents startup; a fatal owner-loop failure wakes active
channels with errors, changes readiness to 503 and rejects new work.

A model-call error is a different case. The scheduler can terminate the
participating batch with error outcomes while its owner loop remains alive.
This does not make readiness a language-quality or universal GPU-health test.

| Boundary | Behaviour in this reference |
|---|---|
| Malformed input, invalid IDs or generation limits | 422 before registration |
| Duplicate live request ID | 409 |
| Eight live requests already registered | 429 |
| Owner loop stopped or failed | Readiness and new requests return 503 |
| Request deadline | Queue cancellation; terminate delivery at a step boundary |
| Explicit cancellation of a final-JSON request | HTTP 409 plus cancelled outcome |
| Model error after stream starts | HTTP 200 plus error terminal event |
| Model error before final JSON response | HTTP 500 plus error outcome |
| Shutdown | Stop accepting, cancel remaining work, release state |

Each output budget is at most 32 tokens. An event queue holds at most that
budget plus `accepted` and `terminal`; the global live-request limit bounds
the number of such queues. This is bounded buffering, not a production
backpressure algorithm: a slow consumer does not throttle model generation.
A production stream needs an explicit slow-reader policy and byte budgets.

Request timeout and startup timeout bound waiting, not the execution time of
a stuck native/GPU call. Python cannot safely kill that worker thread. A
process supervisor and a different failure boundary are needed for that case.
Keep the loopback demo private: it has no authentication, TLS, container
isolation or production overload policy.

## Reproduce the protocol checks

```bash
uv run --offline --with fastapi==0.121.3 --with starlette==0.50.0 \
  --with uvicorn==0.38.0 --with httpx==0.28.1 \
  --with torch==2.14.1 --with numpy==2.3.5 --with transformers==4.57.1 \
  python publishing/tools/check_edls_http_reference.py
```

The checker starts owned servers on OS-assigned loopback ports and stops
them afterward. It uses real TCP HTTP/NDJSON, compares generated IDs with
uncached greedy, and checks token delivery before model completion,
concurrent IDs, EOS/zero output, duplicates, capacity, explicit cancellation,
timeouts, disconnections, model/startup/owner-loop failure and complete cleanup.
One additional ASGI injection forces response-header sending to fail; it is
reported separately from the network checks.

A controlled 10 ms delay per real neural call makes incremental delivery
observable in that test. Those times are not a performance result.
[Recorded protocol evidence](/examples/edls-http-contract-2026-10-26.json)
retains the returned contracts and code hashes.

Containerization, deployment metrics and load/SLO experiments continue in
[[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|the observable-service practice]].
