"""Build/start/check/stop a loopback-only service and a real Prometheus scraper.

Only containers and the network carrying this run's UUID are removed. No remote
daemon or machine is used. Retains all offered-request records, not just p95.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

import grpc

SOURCE = Path(__file__).with_name('observable_service')
DOCKER_ENV = None


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, env=DOCKER_ENV).strip()


def get(url):
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def post(base, data, scheduled=None):
    sent = time.perf_counter()
    request = Request(base + '/predict', json.dumps(data).encode(),
                      {'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=3) as response:
            status, payload = response.status, json.load(response)
    except HTTPError as error:
        status, payload = error.code, json.load(error)
    ended = time.perf_counter()
    return dict(scheduled=scheduled, sent=sent, completed=ended, status=status,
                arrival_lateness=None if scheduled is None else sent-scheduled,
                latency=ended-sent, payload=payload)


def port(container, inside):
    return docker('port', container, str(inside)).split(':')[-1]


def wait(getter, predicate, timeout=30):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        try:
            result = getter()
            if predicate(result):
                return result
        except (URLError, ConnectionError, ValueError):
            pass
        time.sleep(.1)
    raise TimeoutError('local lab readiness deadline')


def main(output):
    global DOCKER_ENV
    image = 'bookvar-observable-lab:2026-10-29'
    unique = 'bookvar-lab-' + uuid.uuid4().hex[:10]
    names = [unique + '-app', unique + '-prom']
    # Reject nonlocal contexts before any mutation.
    context = json.loads(docker('context', 'inspect'))[0]
    endpoint = context['Endpoints']['docker']['Host']
    if not endpoint.startswith(('unix://', 'npipe://')):
        raise SystemExit('This lab only permits a local Docker daemon.')
    # All pulls are public; avoid touching the user's registry credential helper.
    config_dir = tempfile.TemporaryDirectory(prefix='bookvar-docker-config-')
    DOCKER_ENV = dict(os.environ, DOCKER_CONFIG=config_dir.name, DOCKER_HOST=endpoint)
    output.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(['docker', 'build', '-t', image, str(SOURCE)],
                           capture_output=True, text=True, check=True, env=DOCKER_ENV, timeout=300)
    (output / 'docker-build.txt').write_text(build.stdout + build.stderr)
    docker('network', 'create', unique)
    try:
        with tempfile.TemporaryDirectory(prefix='bookvar-prom-') as temp:
            config = Path(temp) / 'prometheus.yml'
            config.write_text('global:\n  scrape_interval: 1s\nscrape_configs:\n'
                              '  - job_name: bookvar\n    static_configs:\n'
                              '      - targets: [app:8080]\n')
            docker('run', '-d', '--name', names[0], '--network', unique,
                '--network-alias', 'app', '-p', '127.0.0.1::8080', '-p', '127.0.0.1::50051',
                '-e', 'LAB_SERVICE_DELAY_SECONDS=.02', '-e', 'LAB_STARTUP_SECONDS=2', image)
            base = 'http://127.0.0.1:' + port(names[0], 8080)
            grpc_target = '127.0.0.1:' + port(names[0], 50051)
            start = time.perf_counter()
            wait(lambda: get(base + '/health'), lambda _: True)
            try:
                get(base + '/ready')
                initial_ready_status = 200
            except HTTPError as error:
                initial_ready_status = error.code
            wait(lambda: get(base + '/ready'), lambda r: r['ready'])
            ready_after = time.perf_counter() - start
            assert initial_ready_status == 503, initial_ready_status
            valid = {'shape': [1, 3], 'data': [.1, .7, .2]}
            assert post(base, valid)['payload']['label'] == 'green'
            assert post(base, {'shape': [1, 2], 'data': [1, 2]})['status'] == 400

            subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', '-I' + str(SOURCE),
                '--python_out=' + temp, '--grpc_python_out=' + temp, str(SOURCE / 'inference.proto')], check=True)
            sys.path.insert(0, temp)
            proto, rpc = importlib.import_module('inference_pb2'), importlib.import_module('inference_pb2_grpc')
            with grpc.insecure_channel(grpc_target) as channel:
                stub = rpc.ImageClassifierStub(channel)
                reply = stub.Predict(proto.ImageClassifierInput(**valid), timeout=2)
                assert reply.label == 'green'
                try:
                    stub.Predict(proto.ImageClassifierInput(shape=[1, 2], data=[1, 2]), timeout=2)
                    raise AssertionError('invalid RPC accepted')
                except grpc.RpcError as error:
                    assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
            docker('run', '-d', '--name', names[1], '--network', unique,
                '-p', '127.0.0.1::9090', '-v', f'{config}:/etc/prometheus/prometheus.yml:ro',
                'prom/prometheus:v3.4.1')
            prom = 'http://127.0.0.1:' + port(names[1], 9090)
            def query(expression):
                return get(prom + '/api/v1/query?' + urlencode({'query': expression}))['data']['result']
            wait(lambda: query('up{job="bookvar"}'), lambda r: bool(r) and r[0]['value'][1] == '1')
            runs = []
            for rate in (10, 100, 200):
                origin = time.perf_counter() + .05
                futures = []
                with ThreadPoolExecutor(max_workers=48) as executor:
                    for i in range(rate * 2):
                        planned = origin + i / rate
                        time.sleep(max(0, planned-time.perf_counter()))
                        futures.append(executor.submit(post, base, valid, planned))
                    records = [f.result() for f in futures]
                duration = max(r['completed'] for r in records)-origin
                successes = [r for r in records if r['status'] == 200]
                p95 = sorted(r['latency'] for r in successes)[int(.95*(len(successes)-1))]
                good = sum(r['status'] == 200 and r['latency'] <= .06 for r in records)
                runs.append(dict(target_rps=rate, offered=len(records), duration=duration,
                    completed=len(successes), rejected=sum(r['status'] == 429 for r in records),
                    successful_p95_seconds=p95, slo_goodput=good/duration,
                    slo_fraction_of_offered=good/len(records), records=records))
            assert runs[0]['rejected'] == 0
            assert any(r['rejected'] > 0 for r in runs[1:])
            expected = 1 + sum(r['completed'] for r in runs)
            scraped = wait(lambda: query('bookvar_requests_total{transport="http",outcome="completed"}'),
                lambda r: bool(r) and float(r[0]['value'][1]) == expected)
            inflight = wait(lambda: query('bookvar_inflight'),
                lambda r: bool(r) and r[0]['value'][1] == '0')
            metrics = get(prom + '/api/v1/targets')['data']['activeTargets']
            report = dict(scope='local container/HTTP/gRPC/Prometheus execution; synthetic 20ms service delay, no VGG or LLM capacity claim',
                initial_ready_status=initial_ready_status, ready_after_probe_seconds=ready_after,
                grpc_label=reply.label, grpc_invalid='INVALID_ARGUMENT',
                scrape_value=scraped, inflight=inflight, targets=metrics, runs=runs,
                service_image=json.loads(docker('image', 'inspect', image))[0]['Id'],
                prometheus_image=json.loads(docker('image', 'inspect', 'prom/prometheus:v3.4.1'))[0]['Id'],
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in SOURCE.iterdir() if p.is_file()},
                model='3x3 identity linear classifier over RGB means; deterministic fixture',
                limits='one executing, three waiting; 4 KiB HTTP body; 8 concurrent RPCs')
            (output / 'service.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps([{k:v for k,v in r.items() if k != 'records'} for r in runs], indent=2))
    finally:
        for name in names:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, env=DOCKER_ENV)
        subprocess.run(['docker', 'network', 'rm', unique], stdout=subprocess.DEVNULL,
                       check=True, env=DOCKER_ENV)
        config_dir.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
