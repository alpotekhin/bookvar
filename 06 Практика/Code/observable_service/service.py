"""Bounded local HTTP/gRPC model service for EDLS transport/observability labs.

Bookvar adaptation, not the course's VGG endpoint. A separately labeled injected
delay exposes queue saturation reproducibly; it is never called model compute.
"""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time

import grpc
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import inference_pb2
import inference_pb2_grpc

MODEL = json.loads(Path(__file__).with_name('model.json').read_text())
DELAY = float(os.environ.get('LAB_SERVICE_DELAY_SECONDS', '0'))
READY = threading.Event()
ADMISSION = threading.BoundedSemaphore(4)  # one executing + at most three waiting
WORKER = threading.Lock()
OUTCOMES = Counter('bookvar_requests', 'Terminal prediction outcomes', ['transport', 'outcome'])
INFLIGHT = Gauge('bookvar_inflight', 'Admitted requests, waiting or executing')
STAGE = Histogram('bookvar_stage_seconds', 'Separate local service stages', ['stage'],
    buckets=[.00001, .0001, .001, .005, .01, .02, .04, .06, .1, .2, 1.])


class Rejected(Exception):
    pass


def predict(data, shape, transport):
    started = time.perf_counter()
    if not READY.is_set():
        OUTCOMES.labels(transport, 'not_ready').inc()
        raise Rejected('not_ready')
    if shape != MODEL['shape'] or len(data) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in data):
        OUTCOMES.labels(transport, 'invalid').inc()
        raise ValueError('Expected finite three-channel means with shape [1,3]')
    STAGE.labels('validation').observe(time.perf_counter() - started)
    if not ADMISSION.acquire(blocking=False):
        OUTCOMES.labels(transport, 'overload').inc()
        raise Rejected('overload')
    INFLIGHT.inc()
    try:
        queued = time.perf_counter()
        with WORKER:
            queue_seconds = time.perf_counter() - queued
            STAGE.labels('queue').observe(queue_seconds)
            before_model = time.perf_counter()
            logits = [sum(w*x for w, x in zip(row, data)) for row in MODEL['weights']]
            label = MODEL['labels'][max(range(len(logits)), key=logits.__getitem__)]
            model_seconds = time.perf_counter() - before_model
            STAGE.labels('model').observe(model_seconds)
            before_delay = time.perf_counter()
            time.sleep(DELAY)
            injected_seconds = time.perf_counter() - before_delay
            STAGE.labels('injected_delay').observe(injected_seconds)
        OUTCOMES.labels(transport, 'completed').inc()
        return dict(label=label, queue_seconds=queue_seconds,
                    model_seconds=model_seconds, injected_seconds=injected_seconds)
    finally:
        INFLIGHT.dec()
        ADMISSION.release()


class HTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, body, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/metrics':
            self.send(200, generate_latest(), CONTENT_TYPE_LATEST)
        elif self.path in ('/health', '/ready'):
            status = 200 if self.path == '/health' or READY.is_set() else 503
            self.send(status, json.dumps({'ready': READY.is_set()}).encode())
        else:
            self.send(404, b'{}')

    def do_POST(self):
        if self.path != '/predict':
            self.send(404, b'{}')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('body length outside 1..4096')
            payload = json.loads(self.rfile.read(length))
            result = predict(payload.get('data', []), payload.get('shape', []), 'http')
            started = time.perf_counter()
            body = json.dumps(result).encode()
            STAGE.labels('serialization').observe(time.perf_counter() - started)
            self.send(200, body)
        except Rejected as error:
            self.send(503 if str(error) == 'not_ready' else 429,
                      json.dumps({'error': str(error)}).encode())
        except (ValueError, TypeError, AttributeError) as error:
            self.send(400, json.dumps({'error': str(error)}).encode())


class RPC(inference_pb2_grpc.ImageClassifierServicer):
    def Predict(self, request, context):
        try:
            result = predict(list(request.data), list(request.shape), 'grpc')
            return inference_pb2.ImageClassifierOutput(label=result['label'])
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Rejected as error:
            code = grpc.StatusCode.UNAVAILABLE if str(error) == 'not_ready' else grpc.StatusCode.RESOURCE_EXHAUSTED
            context.abort(code, str(error))


if __name__ == '__main__':
    rpc = grpc.server(ThreadPoolExecutor(max_workers=8), maximum_concurrent_rpcs=8)
    inference_pb2_grpc.add_ImageClassifierServicer_to_server(RPC(), rpc)
    rpc.add_insecure_port('[::]:50051')
    rpc.start()
    threading.Timer(float(os.environ.get('LAB_STARTUP_SECONDS', '.2')), READY.set).start()
    ThreadingHTTPServer(('0.0.0.0', 8080), HTTPHandler).serve_forever()
