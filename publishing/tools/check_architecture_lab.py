#!/usr/bin/env python3
"""Read the source scenario, not hardware: no Marimo/bootstrap/ledger/model run."""
import ast
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import random
import re
from fractions import Fraction
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / '05 Источники/Originals/nn-architectures-2026-10-26'
PAGE = ROOT / '06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md'
LOCAL_HELPERS = (
    'v1_06_track_amount_system', 'v1_06_family_topology', 'v1_06_budget_ratios',
    'v1_06_topology_summary', 'v1_06_attention_wall', 'v1_06_bias_match',
    'v1_06_bias_frontier', 'v1_06_review_summary',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_engine():
    """Keep core source bodies unchanged; omit only its type-only relative import."""
    path = SOURCES / 'architecture-45ecc8d.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    module = ModuleType('_bookvar_reviewed_architecture')
    sys.modules[module.__name__] = module
    try:
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    finally:
        del sys.modules[module.__name__]
    lab = ast.parse((SOURCES / 'lab06-45ecc8d.py').read_text())
    local_nodes = [n for n in ast.walk(lab)
                   if isinstance(n, ast.FunctionDef) and n.name in LOCAL_HELPERS]
    assert sorted(n.name for n in local_nodes) == sorted(LOCAL_HELPERS)
    exec(compile(ast.Module(body=local_nodes, type_ignores=[]), 'reviewed-local-helpers', 'exec'),
         module.__dict__)
    return module


def original_variants():
    tree = ast.parse((SOURCES / 'variants-45ecc8d.py').read_text())
    variants = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '_variant'):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        lab_id = keywords.get('lab_id')
        if not (isinstance(lab_id, ast.Constant) and lab_id.value == 'v1_06_architecture_tax'):
            continue
        values = {key: ast.literal_eval(value) for key, value in keywords.items()}
        variants[values['track_id']] = values
    assert set(variants) == {'iphone', 'oura_ring', 'robotaxi', 'cloud_fleet'}
    return variants


def plotting_nodes():
    tree = ast.parse((SOURCES / 'lab06-45ecc8d.py').read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = next((i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == '_labels' for t in n.targets)), None)
        if start is None:
            continue
        for end in range(start, len(node.body)):
            n = node.body[end]
            if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == 'apply_plotly_theme'
                    and isinstance(n.value.args[0], ast.Name) and n.value.args[0].id == '_signature_fig'):
                return node.body[start:end + 1]
    raise AssertionError('original plotting segment missing')


def make_profile(engine, values):
    return engine.architecture_track_profile(
        SimpleNamespace(track_id=values['track_id'], label=values['track_id']),
        SimpleNamespace(**values, hardware_ref='unresolved-registry', system_ref=None),
        SimpleNamespace(), SimpleNamespace())


def plotting_source():
    nodes = plotting_nodes()
    lines = (SOURCES / 'lab06-45ecc8d.py').read_text().splitlines(keepends=True)
    return ''.join(lines[nodes[0].lineno - 1:nodes[-1].end_lineno])


def source_report():
    engine = original_engine()
    results = {}
    for track_id, values in original_variants().items():
        # Descriptive registry metadata is not used in the arithmetic; do not
        # pretend to resolve a hardware registry or instantiate an ML model.
        profile = make_profile(engine, values)
        signature = engine.architecture_signature(profile)
        curve = engine.architecture_scaling_curve(profile, samples=36)
        topology = engine.v1_06_topology_summary(profile, signature)
        attention = engine.v1_06_attention_wall(profile, signature, curve, profile.default_scale)
        bias = engine.v1_06_bias_frontier(profile, signature, 1.0)
        review = engine.v1_06_review_summary(profile, signature,
                   engine.architecture_decision(profile, architecture_id=profile.candidates[0].architecture_id))
        results[track_id] = {
            'defaults': values['defaults'], 'signature': [asdict(x) for x in signature],
            'curve': asdict(curve), 'topology': topology, 'attention': attention,
            'bias': bias, 'review': {key: asdict(value) if key in ('selected', 'leaderboard', 'headroom_pick') else value
                                   for key, value in review.items()},
        }
    return results


def main(output=None):
    results = source_report()
    engine = original_engine()
    expected_feasible = {
        'iphone': ['mobile_cnn', 'efficient_transformer'],
        'oura_ring': ['streaming_tcn', 'tiny_cnn'],
        'robotaxi': ['bounded_cnn_detector', 'hybrid_vit_detector'],
        'cloud_fleet': ['bert_encoder', 'moe_transformer'],
    }
    for track_id, expected in expected_feasible.items():
        assert [x['architecture_id'] for x in results[track_id]['signature'] if x['feasible']] == expected
    rng = random.Random(20261026)
    cases = 0
    for values in original_variants().values():
        profile = make_profile(engine, values)
        for _ in range(75):
            scale = rng.uniform(profile.scale_min, profile.scale_max)
            signature = engine.architecture_signature(profile, scale_value=scale)
            for candidate, evaluation in zip(profile.candidates, signature):
                ratio = scale / profile.default_scale
                # Derive the constraints independently from the literal scenario.
                amounts = [candidate.activation_mb * ratio ** candidate.activation_exponent,
                           candidate.latency_ms * ratio ** candidate.latency_exponent,
                           candidate.power_w * ratio ** min(candidate.op_exponent, 1.5)]
                allowed = [profile.memory_budget_mb, profile.latency_budget_ms, profile.power_budget_w]
                expected = (all(a <= b for a, b in zip(amounts, allowed))
                            and candidate.quality_pct >= profile.quality_floor_pct
                            and candidate.kernel_support_pct >= profile.kernel_support_floor_pct)
                assert evaluation.feasible == expected
                assert evaluation.params_m == candidate.params_m  # all four tracks hold weights fixed
            cases += 1

    mobile = results['iphone']
    assert mobile['topology']['best_label'] == 'Efficient mobile transformer'
    assert mobile['bias']['best']['label'] == 'Mobile CNN / inverted residuals'
    assert mobile['review']['headroom_pick']['architecture_id'] == 'mobile_cnn'
    assert math.isclose(mobile['review']['headroom'], 15 / 7)
    desktop_first = mobile['curve']['points_by_candidate']['desktop_vit'][0]
    assert desktop_first['scale_value'] == 112 and not desktop_first['feasible']

    page = PAGE.read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', page, re.M | re.S)
    assert len(blocks) == 4
    for i, block in enumerate(blocks):
        exec(compile(block, str(PAGE) + f':block-{i}', 'exec'), engine.__dict__)
    original = next(n for n in ast.parse((SOURCES / 'architecture-45ecc8d.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_scaled')
    copied = next(n for n in ast.parse(blocks[0]).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(copied) == ast.dump(original)
    quoted = re.search(r'<!-- source-excerpt: architecture-four-selection-steps -->\n'
                       r'(.*?)\n<!-- /source-excerpt -->', page, re.S)[1]
    quoted = '\n'.join(re.sub(r'^> ?', '', line) for line in quoted.splitlines())
    source = (SOURCES / 'chapter-2bd97c5.qmd').read_text()
    assert ' '.join(quoted.split()) in ' '.join(source.split())

    # Preserve the entire accepted practice outside this case and its dates.
    prior = re.sub(r'<!-- BEGIN REVIEW: Wildlife budgets 2026-10-27 -->\n.*?'
                   r'<!-- END REVIEW: Wildlife budgets 2026-10-27 -->\n\n', '', page, flags=re.S)
    prior = prior.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26') \
                 .replace('last_verified: 2026-10-27', 'last_verified: 2026-10-26')
    assert digest(prior.encode()) == 'd9871982d0377913404e35754d102f03a2b87bf3c245527612e6062fdabd6db6'
    # Read the complete original case's literal inputs without importing mlsysim.
    source_case = source.split('#| label: wildlife-model-sizing\n')[1].split('\n```')[0]
    source_class = next(n for n in ast.parse(source_case).body
                        if isinstance(n, ast.ClassDef) and n.name == 'WildlifeModelSizing')
    constants = {}
    for node in source_class.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    assert constants['mnv2_params_m'] == 2.2
    assert constants['mnv2_flops_mflops'] == 150
    assert constants['inference_power_mw'] == 200
    assert constants['inferences_per_day'] == 100
    assert constants['edge_int8_tops'] == .002
    assert constants['activation_channels'] == 64 and constants['activation_bytes'] == 4
    assert constants['os_buffers_mb'] == 50 and constants['edge_ram_mb'] == 512
    assert constants['model_budget_mb'] == 100 and constants['target_latency_ms'] == 500
    assert constants['accuracy_target_pct'] == 90 and constants['species_count'] == 50
    daily = engine.camera_daily_energy
    for count in [0, 1, 100, 10_000, 1_000_000, 1_152_000]:
        # Independent baseline plus incremental draw, not active + full-day idle.
        expected = Fraction(864) + Fraction(19, 100) * count * Fraction(3, 40)
        assert daily(count) == expected
    assert daily(1_152_000) == Fraction(1, 5) * 86_400
    for invalid in [-1, 1_152_001]:
        try:
            daily(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('two disjoint time states exceeded the day')
    for cin, cout, side in [(3,16,3), (32,64,3), (64,128,5), (8,8,1)]:
        dense = cin * cout * side * side
        separated = cin * side * side + cin * cout
        assert Fraction(separated, dense) == Fraction(1, cout) + Fraction(1, side*side)

    plot = json.loads((ROOT / '06 Практика/Results/architecture-lab-2026-10-26/source-plot.json').read_text())
    expected_traces = [[100 * row[field] / budget for row in mobile['signature']]
                       for field, budget in [('activation_mb', 220), ('latency_ms', 25), ('power_w', 4.5)]]
    assert [t['y'] for t in plot['data']] == expected_traces
    # Hash original source text: ast.dump formatting differs across Python versions.
    plot_source_sha = digest(plotting_source().encode())
    assert plot_source_sha == '7e097ec3195471135814668c6922200e881d85900343569f113c8ae5e40f5461'
    assert plot['layout']['shapes'][0]['y0'] == plot['layout']['shapes'][0]['y1'] == 100

    # The lecture's numbers are checked, not silently adopted as measurements.
    source_findings = {
        'weight_count_ratio': 1_048_576 / 9_408,
        'gpt2_literal_formula': 12 * 12 * 768 ** 2 * 1024,
        'gpt3_literal_formula': 12 * 96 * 12_288 ** 2 * 2048,
        'five_images_in_25ms_per_second': 5 / .025,
        'materialized_scores_bytes': 4096 ** 2 * 2 * 32 * 32,
        'kv_cache_bytes': 2 * 32 * 32 * 2048 * 128 * 2,
    }
    assert source_findings['weight_count_ratio'] > 111
    assert source_findings['gpt3_literal_formula'] / source_findings['gpt2_literal_formula'] == 4096
    assert source_findings['five_images_in_25ms_per_second'] == 200
    assert source_findings['kv_cache_bytes'] == 2 ** 30
    report = {
        'scope': 'Scenario arithmetic from reviewed original helpers; no Marimo, ledger, hardware registry, model, accelerator, benchmark or safety validation',
        'source_hashes': {p.name: digest(p.read_bytes()) for p in sorted(SOURCES.glob('*')) if p.is_file() and p.name != 'README.md'},
        'page_sha256': digest(page.encode()), 'checker_sha256': digest(Path(__file__).read_bytes()),
        'independent_sweeps': cases, 'source_candidates': 12,
        'executed_printed_blocks': len(blocks), 'verbatim_source_excerpts': 1,
        'local_source_helpers': list(LOCAL_HELPERS), 'source_plot_source_sha256': plot_source_sha,
        'wildlife_budget_checks': {
            'source_literal_constants': constants,
            'partial_memory_bytes': engine.camera_partial_bytes,
            'conditional_model_time_s': str(engine.camera_time_s),
            'daily_energy_100_j': str(daily(100)),
            'capacity_180_days_wh': str(daily(100)*180/3600),
            'extra_idle_power_w': '1/100; Bookvar assumption, not source measurement',
            'hardware_measurements': False,
        },
        'source_slide_arithmetic_findings': source_findings, 'tracks': results,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if output:
        Path(output).write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    main(parser.parse_args().output)
