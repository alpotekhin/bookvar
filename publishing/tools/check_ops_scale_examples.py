#!/usr/bin/env python3
"""Run reviewed Harvard arithmetic without importing Marimo or writing its ledger."""
import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace
from practice_runtime_baseline import before_observable_runtime
from serving_operations_baseline import before_serving_operations

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/mlops-scale-lab-2026-10-26'
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'
PRACTICE = ROOT / '06 Практика/17 Развернуть наблюдаемый ML сервис.md'
HELPERS = ['v2_12_track_packet', 'v2_12_period_minutes', 'v2_12_error_budget',
           'v2_12_canary_budget', 'v2_12_runbook_factor', 'v2_12_incident_budget',
           'v2_12_governance_score', 'v2_12_policy_eval']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_helpers():
    """Compile only the eight named pure functions, never their enclosing cells."""
    raw = (SOURCE / 'lab_12_ops_scale.py').read_text()
    tree = ast.parse(raw)
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    assert sorted(node.name for node in nodes) == sorted(HELPERS)
    assert all(not node.decorator_list for node in nodes)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 'harvard-ops-reviewed-pure-functions', 'exec'), namespace)
    return namespace, raw


def source_report():
    namespace, raw = extract_helpers()
    profile = SimpleNamespace(track_id='cloud_fleet', label='Cloud fleet',
                              narrative='Arithmetic fixture', source_policy='original lab')
    variant = SimpleNamespace(stakeholder='platform team', workload_summary='fixture',
                              hardware_ref='not measured', model_ref='not loaded')
    packet = namespace['v2_12_track_packet'](profile, variant)
    a = namespace['v2_12_error_budget'](
        packet, slo_pct=99.9, quality_floor_pct=91, incident_count=3,
        detection_min=25, impact_min=40)
    b = namespace['v2_12_canary_budget'](
        packet, canary_pct=5, stage_hours=2, sample_needed=10_000, traffic_multiplier=1)
    late = namespace['v2_12_canary_budget'](
        packet, canary_pct=1, stage_hours=5, sample_needed=10_000, traffic_multiplier=1)
    c = {key: namespace['v2_12_incident_budget'](
        packet, mttd_min=20, diagnosis_min=25, mitigation_pct=60,
        recovery_min=50, runbook_level=key)
        for key in ['ad_hoc', 'typed_runbook', 'drilled_automation']}
    policies = {'fast': (70, 45, 35, 'self_approved'),
                'balanced': (22, 75, 75, 'governed_window'),
                'conservative': (8, 88, 90, 'safety_board')}
    d = {key: namespace['v2_12_policy_eval'](
        packet, name=key, rollout=values[0], automation=values[1],
        telemetry=values[2], governance=values[3])
        for key, values in policies.items()}
    assert math.isclose(a['error_budget_minutes'], 43.2)
    assert a['impact_minutes'] == 195 and math.isclose(a['quality_spend_pp'], 1.825)
    assert a['binding_key'] == 'availability' and not a['ok']
    assert b['required_hours'] == .2 and b['samples_collected'] == 100_000
    assert b['evidence_ok'] and b['blast_ok']
    assert late['evidence_ok'] and late['blast_ok'] and 5 > packet['release_window_h']
    assert [c[key]['lost_work'] for key in c] == [45_600, 41_600, 36_800]
    assert [d[key]['feasible'] for key in d] == [False, True, False]
    assert not a['ok'] and d['balanced']['feasible']

    tracks = {}
    for track_id in ['cloud_fleet', 'iphone', 'oura_ring', 'robotaxi']:
        profile.track_id = track_id
        track_packet = namespace['v2_12_track_packet'](profile, variant)
        result = namespace['v2_12_canary_budget'](
            track_packet, canary_pct=track_packet['canary_default_pct'],
            stage_hours=track_packet['canary_default_h'],
            sample_needed=track_packet['sample_default'], traffic_multiplier=1)
        tracks[track_id] = {
            'traffic_unit': track_packet['traffic_unit'],
            'traffic_per_h': track_packet['traffic_per_h'],
            'sample_needed': track_packet['sample_default'],
            'exposure_budget': track_packet['blast_budget_units'],
            **result,
        }
    assert [tracks[key]['samples_collected'] for key in tracks] == [100_000, 20_000, 3_000, 120]
    assert [tracks[key]['actual'] for key in tracks] == ['balanced', 'balanced', 'aggressive_exposed', 'tiny_blind']
    assert tracks['robotaxi']['sample_needed'] > tracks['robotaxi']['exposure_budget']

    # Independent integer/rational accounting, not just rerunning the same formulas.
    rng = random.Random(20261026)
    for _ in range(200):
        rate = rng.randrange(1, 10_000) * 100
        fraction_pct, hours = rng.randrange(1, 101), rng.randrange(1, 10)
        samples = rng.randrange(1, 1_000_000)
        custom = {**packet, 'traffic_per_h': rate}
        result = namespace['v2_12_canary_budget'](
            custom, canary_pct=fraction_pct, stage_hours=hours,
            sample_needed=samples, traffic_multiplier=1)
        counted = (rate // 100) * fraction_pct * hours
        assert math.isclose(result['samples_collected'], counted, rel_tol=1e-12)
        assert result['evidence_ok'] == (counted >= samples)
        assert result['blast_ok'] == (counted <= packet['blast_budget_units'])
        diagnosis, detection, recovery, mitigation = [rng.randrange(101) for _ in range(4)]
        incident = namespace['v2_12_incident_budget'](
            packet, mttd_min=detection, diagnosis_min=diagnosis,
            mitigation_pct=mitigation, recovery_min=recovery,
            runbook_level='typed_runbook')
        expected = 640 * (detection + diagnosis) + Fraction(640 * (100 - mitigation) * recovery, 100)
        assert math.isclose(incident['lost_work'], float(expected))

    return namespace, raw, {
        'source_hashes': {
            str((SOURCE / name).relative_to(ROOT)): sha((SOURCE / name).read_bytes())
            for name in ['lab_12_ops_scale.py', '12_ops_scale.tex']},
        'source_helpers_executed': HELPERS,
        'cloud_packet': packet, 'availability': a, 'canary': b,
        'source_accepts_canary_past_deadline': late,
        'incidents': c, 'policies': d, 'source_track_scenarios': tracks,
        'independent_accounting_cases': 200,
    }


def section_and_previous(text, key):
    match = re.search(r'<!-- BEGIN REVIEW: ' + re.escape(key) + r' 2026-10-26 -->\n'
                      r'(.*?)<!-- END REVIEW: ' + re.escape(key) + r' 2026-10-26 -->\n\n',
                      text, re.S)
    assert match, key
    return match[1], text[:match.start()] + text[match.end():]


def main(output=None):
    namespace, raw, report = source_report()
    expected_hashes = ['ec9b37e07f0579cc9937656d65d886efd79a560576f8b49038deea48ad7e6046',
                       'ef63aa3bd042f2cc84218bd5c12032417fadfb980ad737a21cae4e396c1e04a1']
    assert list(report['source_hashes'].values()) == expected_hashes
    chapter, practice = CHAPTER.read_text(), PRACTICE.read_text()
    chapter_part, old_chapter = section_and_previous(
        before_serving_operations(chapter, CHAPTER), 'mlops fleet-alerts')
    practice_part, old_practice = section_and_previous(before_observable_runtime(practice), 'practice ops-scale')
    assert sha(old_chapter.encode()) == 'ac9a516bc886263afb6ebd7303dbd2d12be6533d167842b1fa6c713d26a370ee'
    assert sha(old_practice.encode()) == '87f6c48330ce879804236099e5f248e61cdf101505b8e94523af5f7035a913a0'
    printed = []
    for label, text in [('fleet', chapter_part), ('practice', practice_part)]:
        own = {}
        blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
        for number, block in enumerate(blocks):
            exec(compile(block, f'ops-scale-{label}-{number}', 'exec'), own)
        printed.append(len(blocks))
    assert printed == [1, 4]
    # The reproduced function is AST-identical to the pinned original.
    original = next(node for node in ast.walk(ast.parse(raw))
                    if isinstance(node, ast.FunctionDef) and node.name == 'v2_12_canary_budget')
    copied = next(node for node in ast.walk(ast.parse('\n'.join(
        re.findall(r'^```python\n(.*?)^```', practice_part, re.M | re.S))))
                  if isinstance(node, ast.FunctionDef) and node.name == original.name)
    assert ast.dump(original) == ast.dump(copied)
    # Source prose excerpts stay verbatim; do not paraphrase the licensed source silently.
    excerpts = re.findall(r'<!-- source-excerpt: harvard-ops-[^>]+ -->\n'
                          r'(.*?)\n<!-- /source-excerpt -->', practice_part, re.S)
    assert len(excerpts) == 3
    for excerpt in excerpts:
        plain = '\n'.join(re.sub(r'^> ?', '', line) for line in excerpt.splitlines())
        assert ' '.join(plain.split()) in ' '.join(raw.split())
    assert 'prometheus-architecture.svg]]' in chapter_part
    assert 'sre-canary-versions.webp]]' in practice_part
    assert '| Wearable syncs | 30,000 | 2,500 | 2,500 | 3,000 | No: too much exposure |' in practice_part
    assert '| Vehicle live miles | 1,200 | 3,000 | 120 | 120 | No: too little evidence |' in practice_part
    report.update({
        'chapter_sha256': sha(chapter.encode()), 'practice_sha256': sha(practice.encode()),
        'checker_sha256': sha(Path(__file__).read_bytes()),
        'accepted_2240_chapter_lines_preserved': True,
        'accepted_645_practice_lines_preserved': True,
        'new_python_blocks_executed': sum(printed),
        'verbatim_source_excerpts': len(excerpts),
        'scope': 'CPU arithmetic from eight AST-extracted pure functions and printed examples; '
                 'no Marimo UI, prediction gates, ledger write, model, traffic, Docker or deployment',
    })
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if output:
        Path(output).write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    main(parser.parse_args().output)
