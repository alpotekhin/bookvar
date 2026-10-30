"""Exercise unchanged EDLS Week 7 examples; do not download VGG or start Docker."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Histogram, REGISTRY

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / '05 Источники/Originals/edls-deployment-2026-10-26'
SOURCE = ARCHIVE / 'week07_application_deployment'


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def sample(name):
    return sum(s.value for family in REGISTRY.collect() for s in family.samples
               if s.name == name)


def check():
    baseline = SOURCE / '00_basics/non-smokers-repo'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LC_ALL='C', LANG='C')
    original = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider', 'tests'],
        cwd=baseline, env=env, capture_output=True, text=True, check=False)
    assert original.returncode == 1, original.stdout + original.stderr
    assert 'test_seeded_score_matches_expected_value_with_high_precision' in original.stdout
    assert '1 failed, 4 passed' in original.stdout, original.stdout
    sys.path.insert(0, str(baseline))
    from app.config import Settings
    from app.main import create_app
    from app.model import ToxicityModel

    model = ToxicityModel(7)
    model.load()
    score = model.score('neutral text')
    assert abs(score - 0.00006476655296663248) < 1e-16
    app = create_app(Settings(model_startup_delay_seconds=0.4))
    with TestClient(app) as client:
        early_health = client.get('/health')
        early_predict = client.post('/predict', json={'text': 'hello'})
        assert early_health.status_code == 200
        assert early_health.json() == {'status': 'not ready'}
        assert early_predict.status_code == 503
        until = time.monotonic() + 2
        while client.get('/health').json()['status'] != 'ready':
            assert time.monotonic() < until
            time.sleep(0.01)
        assert client.post('/predict', json={'text': 'you are an idiot'}).json() == {'is_toxic': True}
        assert client.post('/predict', json={'text': []}).status_code == 422

    with patch.dict(os.environ, {'MODEL_LOAD_SECONDS': '0'}):
        with redirect_stderr(io.StringIO()):
            bad = runpy.run_path(str(SOURCE / '00_basics/smokers-repo/app.py'))
    with TestClient(bad['app']) as client:
        invalid = client.post('/predict', content='not-json', headers={'content-type': 'application/json'})
        assert invalid.status_code == 200 and invalid.json() == {'is_toxic': False}
        assert client.post('/predict', json={'text': []}).json() == {'is_toxic': False}
    random.seed(7)
    false_positives = sum(bad['classify_text']('thank you') for _ in range(100))
    assert 0 < false_positives < 100

    # Exercise the unchanged Flask route/exporter, not the unavailable VGG model.
    torch = ModuleType('torch')
    torch.tensor = lambda data: data
    def fixed_model(data):
        if data == 'injected failure':
            raise RuntimeError('controlled model failure, not a real inference')
        return SimpleNamespace(data=SimpleNamespace(numpy=lambda: SimpleNamespace(argmax=lambda: 0)))
    torch.jit = SimpleNamespace(load=lambda _: fixed_model)
    with tempfile.TemporaryDirectory(prefix='bookvar-edls-metrics-') as temp:
        labels = json.loads((SOURCE / '03_metrics/labels.json').read_text())
        (Path(temp) / 'labels.json').write_text(json.dumps(labels))
        with working_directory(temp), patch.dict(sys.modules, {'torch': torch}):
            metrics = runpy.run_path(str(SOURCE / '03_metrics/server.py'))
        app = metrics['app']
        app.logger.disabled = True
        client = app.test_client()
        before = {n: sample(n) for n in ('predictions_total', 'api_invocations_total')}
        for _ in range(3):
            response = client.post('/predict', json={'data': [1]})
            assert response.status_code == 200 and response.json['label'] == labels[0]
        assert client.post('/predict', json={'data': 'injected failure'}).status_code == 500
        exposition = client.get('/metrics')
        assert exposition.status_code == 200
        assert b'flask_http_request_duration_seconds_bucket' in exposition.data
        after = {n: sample(n) for n in before}
        assert after['predictions_total'] - before['predictions_total'] == 3
        assert after['api_invocations_total'] - before['api_invocations_total'] == 4
        assert sample('api_in_progress') == 0

    registry = CollectorRegistry()
    histogram = Histogram('example_latency_seconds', 'Deterministic observations',
                          buckets=[0.1, 0.3, 1], registry=registry)
    for value in [0.05, 0.2, 0.4, 0.8]:
        histogram.observe(value)
    samples = {s.labels.get('le', s.name): s.value
               for f in registry.collect() for s in f.samples}
    assert [samples[x] for x in ['0.1', '0.3', '1.0', '+Inf']] == [1, 2, 4, 4]
    assert abs(samples['example_latency_seconds_sum'] - 1.45) < 1e-12
    assert samples['example_latency_seconds_count'] == 4
    return {
        'scope': 'unchanged course logic and in-process ASGI/Flask instrumentation; stubbed VGG; no Docker, gRPC, Prometheus server, GPU or load benchmark',
        'original_pytest': {'exit_code': original.returncode, 'expected_failure': 'test_seeded_score_matches_expected_value_with_high_precision', 'passed': 4, 'failed': 1},
        'seed7_first_score': score,
        'health_while_loading': {'status': early_health.status_code, 'body': early_health.json(), 'predict_status': early_predict.status_code},
        'bad_source': {'invalid_json_status': invalid.status_code, 'invalid_json_body': invalid.json(), 'neutral_false_positives_in_100_seed7': false_positives},
        'source_metrics_deltas': {n: after[n] - before[n] for n in before},
        'gauge_after_error': sample('api_in_progress'),
        'histogram': {'buckets': {x: samples[x] for x in ['0.1', '0.3', '1.0', '+Inf']}, 'count': 4, 'sum': 1.45, 'slo_300ms_fraction': 0.5},
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(SOURCE.rglob('*')) if p.is_file()
                          and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)},
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {name: __import__('importlib.metadata').metadata.version(name)
                     for name in ['fastapi', 'starlette', 'httpx', 'Flask', 'prometheus-flask-exporter', 'prometheus-client', 'pytest', 'redis']},
        'python': sys.version,
    }, original.stdout + original.stderr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, diagnostic = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    args.output.with_suffix('.pytest.txt').write_text(diagnostic)
    print('PASS source boundary checks; original course suite has the documented 1 failure / 4 passes')
