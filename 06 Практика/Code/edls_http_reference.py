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
