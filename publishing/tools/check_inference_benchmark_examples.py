#!/usr/bin/env python3
"""Chapter 58b: CPU metric examples and source identity, not a serving run."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md'
SOURCE = ROOT / '05 Источники/Originals/inference-benchmark-2026-10-20'
FIGURES = ROOT / '00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10'


def must_reject(fn, *args):
    try:
        fn(*args)
    except AssertionError:
        return
    raise AssertionError(f'{fn.__name__} accepted invalid input')


def main():
    chapter = before_serving_operations(CHAPTER.read_text(), CHAPTER)
    blocks = re.findall(r'^```python\n(.*?)^```', chapter, re.M | re.S)
    assert len(blocks) == 7, len(blocks)
    ns = {'__name__': '__chapter_examples__'}
    for i, block in enumerate(blocks):
        exec(compile(block, f'{CHAPTER}:python-block-{i + 1}', 'exec'), ns)
    rng = np.random.default_rng(20261020)
    counts = dict(timestamps=0, cohorts=0, quantiles=0, client_queue=0,
                  context_work=0, phase_accounting=0, burn_rate=0)

    metrics = ns['request_metrics']
    for _ in range(300):
        sent = float(rng.uniform(0, 100))
        tokens = sent + np.cumsum(rng.uniform(1, 100, int(rng.integers(1, 40))))
        result = metrics(sent, tokens)
        np.testing.assert_allclose(result['itl'], tokens[1:] - tokens[:-1])
        np.testing.assert_allclose(result['ttft'] + sum(result['itl']), result['e2e'])
        if len(tokens) == 1:
            assert result['tpot'] is None
        else:
            np.testing.assert_allclose(result['tpot'], (tokens[-1] - tokens[0]) / (len(tokens) - 1))
        counts['timestamps'] += 1
    for sent, times in [(0, []), (2, [1]), (0, [2, 1])]:
        must_reject(metrics, sent, times)
    # Observed chunk gaps are not reconstructed token timestamps.
    np.testing.assert_allclose(np.diff([200., 300., 500.]), [100., 200.])
    assert (500 - 200) / (6 - 1) == 60
    assert (500 - 200) / (6 - 2) == 75

    summarize = ns['summarize']
    for _ in range(300):
        records, good, successful, total_tokens = [], 0, 0, 0
        for index in range(int(rng.integers(1, 50))):
            if rng.random() < .2:
                records.append(dict(id=index, tokens=None))
                continue
            intervals = rng.uniform(1, 140, int(rng.integers(0, 15)))
            first = float(rng.uniform(1, 600))
            times = [first, *(first + np.cumsum(intervals)).tolist()]
            records.append(dict(id=index, tokens=times))
            successful += 1
            total_tokens += len(times)
            passes = first <= 300 and (len(intervals) == 0 or sum(intervals) / len(intervals) <= 80)
            good += int(passes)
        duration = float(rng.uniform(1, 50))
        actual = summarize(records, duration, 300., 80.)
        np.testing.assert_allclose([actual['completed_rps'], actual['goodput_rps'],
                                   actual['output_tps'], actual['attainment_all']],
                                  [successful / duration, good / duration,
                                   total_tokens / duration, good / len(records)])
        counts['cohorts'] += 1
    assert ns['summary']['attainment_all'] == 2 / 6
    assert ns['summary']['goodput_rps'] == .2
    ttft_ok, tpot_ok = np.ones(100, bool), np.ones(100, bool)
    ttft_ok[:10], tpot_ok[10:20] = False, False
    assert ttft_ok.mean() == tpot_ok.mean() == .9
    assert (ttft_ok & tpot_ok).mean() == .8

    percentile = ns['nearest_rank']
    for _ in range(300):
        values = rng.integers(1, 1000, int(rng.integers(1, 80))).tolist()
        for p in [.01, .5, .95, .99, 1.]:
            assert percentile(values, p) == np.quantile(values, p, method='inverted_cdf')
            counts['quantiles'] += 1
    for n in [10, 100, 1000, 10000]:
        q = ns['zero_failure_upper_bound'](n)
        np.testing.assert_allclose((1 - q)**n, .05, rtol=1e-10)
    for args in [([], .95), ([1.], 0), ([1.], 1.1)]:
        must_reject(percentile, *args)
    assert np.percentile([500.] + [10.] * 99, 95) == 10.
    assert percentile([10.] * 1000 + [1000.] * 10, .99) == 10.
    assert np.mean([percentile([10.] * 1000, .99), percentile([1000.] * 10, .99)]) == 505.

    dispatch = ns['dispatch_with_limit']
    for _ in range(300):
        arrivals = np.cumsum(rng.uniform(0, 1, int(rng.integers(1, 60))))
        concurrency = int(rng.integers(1, 10))
        service = float(rng.uniform(.01, 2))
        actual, ends = dispatch(arrivals, service, concurrency)
        # Independent indexed-worker reference rather than a heap.
        available = [0.] * concurrency
        expected = []
        for arrival in arrivals:
            worker = int(np.argmin(available))
            sent = max(float(arrival), available[worker])
            expected.append(sent)
            available[worker] = sent + service
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(ends - actual, service)
        assert np.all(actual >= arrivals)
        events = sorted([(float(s), 1) for s in actual] + [(float(e), -1) for e in ends])
        running = peak = 0
        for _, delta in events:
            running += delta
            peak = max(peak, running)
            assert running >= 0
        assert peak <= concurrency and running == 0
        unrestricted, _ = dispatch(arrivals, service, len(arrivals))
        np.testing.assert_allclose(unrestricted, arrivals)
        counts['client_queue'] += 1
    must_reject(dispatch, [0., 1.], 1., 0)
    must_reject(dispatch, [-1.], 1., 1)
    must_reject(dispatch, [2., 1.], 1., 1)

    work = ns['decode_key_positions']
    for _ in range(500):
        prompt, output = int(rng.integers(0, 10000)), int(rng.integers(1, 1000))
        assert work(prompt, output) == sum(range(prompt + 1, prompt + output))
        counts['context_work'] += 1
    assert ns['a_positions'] == 450672
    assert ns['b_positions'] == 4321392
    for _ in range(300):
        prefill = float(rng.uniform(0, 1))
        decode = float(rng.uniform(.01, 2))
        output = int(rng.integers(2, 1000))
        result = ns['phase_accounting'](prefill, prefill + decode, output)
        np.testing.assert_allclose(result['decode_tpot_s'] * (output - 1), decode)
        np.testing.assert_allclose(result['decode_output_tps'] * decode, output - 1)
        np.testing.assert_allclose(result['full_output_tps'] * (prefill + decode), output)
        counts['phase_accounting'] += 1

    burn = ns['burn_rate']
    for _ in range(300):
        eligible = int(rng.integers(1, 100000))
        bad = int(rng.integers(eligible + 1))
        slo = float(rng.uniform(.9, .9999))
        np.testing.assert_allclose(burn(bad, eligible, slo), bad / (eligible * (1 - slo)))
        counts['burn_rate'] += 1
    for args in [(0, 0), (2, 1), (-1, 2), (0, 1, 1.)]:
        must_reject(burn, *args)
    assert not ns['two_window_page'](20., 0.)
    assert not ns['two_window_page'](0., 20.)
    assert ns['two_window_page'](20., 20.)
    np.testing.assert_allclose(60 * (8 - 6.2), 108)
    np.testing.assert_allclose(14.4 / (30 * 24), .02)
    assert abs((1 + 100) / (10 + 100000) - (.1 + .001) / 2) > .01

    manifest = json.loads((SOURCE / 'references.json').read_text())
    for item in manifest['sources']:
        assert hashlib.sha256((SOURCE / item['file']).read_bytes()).hexdigest() == item['sha256']
    paper = ROOT / manifest['paper']['file']
    assert hashlib.sha256(paper.read_bytes()).hexdigest() == manifest['paper']['sha256']
    embeds = [v.removeprefix('02 Areas/ML & DL/') for v in re.findall(r'!\[\[([^\]]+)\]\]', chapter)]
    assert set(embeds) == {f['asset'] for f in manifest['figures']} and len(embeds) == 5
    for figure in manifest['figures']:
        assert hashlib.sha256((ROOT / figure['asset']).read_bytes()).hexdigest() == figure['sha256']
    records = json.loads((FIGURES / 'assets.json').read_text())
    for record in records:
        if record['derivation'] == 'pdf-page-render-crop':
            with tempfile.TemporaryDirectory(prefix='bookvar-benchmark-') as temporary:
                target = Path(temporary) / 'page2'
                subprocess.run(['pdftoppm', '-f', '2', '-l', '2', '-singlefile', '-scale-to',
                                str(record['render_scale_to']), '-png', str(paper), str(target)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                page = Image.open(target.with_suffix('.png')).convert('RGB')
                assert list(page.size) == record['render_page_size_px']
                expected = np.asarray(page.crop(record['crop_box_px']))
                np.testing.assert_array_equal(np.asarray(Image.open(ROOT / record['asset']).convert('RGB')), expected)
        elif record.get('source_asset'):
            assert (ROOT / record['asset']).read_bytes() == (ROOT / record['source_asset']).read_bytes()
    assert (FIGURES / 'mlperf-loadgen.svg').read_bytes() == (SOURCE / 'mlperf-loadgen.svg').read_bytes()
    assert (ROOT / 'site/public/licenses/mlperf-3fbc329-Apache-2.0.txt').read_bytes() == (SOURCE / 'mlperf-LICENSE.md').read_bytes()
    assert manifest['rejected_figure']['asset'] not in embeds
    notebook = json.loads((SOURCE / 'edls-seminar.ipynb').read_text())
    assert not any(cell.get('outputs') for cell in notebook['cells'])
    for marker in ['workload-важнее-названия-gpu', 'минимальный-честный-отчёт',
                   'воспроизводимое-измерение-qwen3-4b']:
        assert f'id="{marker}"' in chapter
    assert '[[58a2 Раздельное обслуживание prefill и decode|' in chapter
    assert '[[58c Queueing и capacity planning|' in chapter
    print(json.dumps(dict(chapter_blocks=len(blocks), cpu_cases=counts,
                          source_snapshots=len(manifest['sources']), figures=len(embeds),
                          reproduced_pdf_crops=1, unchanged_licensed_figures=3,
                          scope='CPU arithmetic and source fidelity only; no measured serving performance'),
                     indent=2))


if __name__ == '__main__':
    main()

