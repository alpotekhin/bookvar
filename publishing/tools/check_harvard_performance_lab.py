"""Execute only the reviewed pure numeric helpers, never Marimo/bootstrap/cloud code."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/performance-practice-2026-10-26/harvard-lab09.py'
tree = ast.parse(SOURCE.read_text())
names = {
    'v2_09_track_packet', 'v2_09_best_lever', 'v2_09_profile_result',
    'v2_09_regression_result', 'v2_09_capacity_result', 'v2_09_tradeoff_candidates',
    'v2_09_bottleneck_label', 'v2_09_candidate_label',
}
definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in definitions} == names
assert all(not n.decorator_list for n in definitions)
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for d in definitions for n in ast.walk(d))
namespace = {'math': math}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), namespace)

packet = namespace['v2_09_track_packet'](SimpleNamespace(
    track_id='cloud_fleet', label='Cloud fleet', stakeholder='teaching fixture',
    source_policy='source constants, not measurements', hardware_ref='not a hardware run',
    system_ref=None,
))
profile = namespace['v2_09_profile_result'](packet)
assert profile['terms'] == {'data': 101.76, 'compute': 55.08, 'overhead': 26.52}
assert math.isclose(profile['p50_ms'], 128.28)
assert math.isclose(profile['p99_ms'], 173.178)
regression = namespace['v2_09_regression_result'](packet, 8, 6, 5)
assert math.isclose(regression['mde_pct'], 5.88)
assert regression['decision'] == 'hold'
print('Original fixture baseline:', profile)
print('Original fixture regression:', regression)

# A source-model discontinuity: crossing capacity reduces its queue multiplier.
capacity_fn = namespace['v2_09_capacity_result']
base_capacity = capacity_fn(packet, 1, 5, 'none')['capacity']
before = capacity_fn(packet, .999 * base_capacity / packet['base_demand'], 5, 'none')
at = capacity_fn(packet, base_capacity / packet['base_demand'], 5, 'none')
assert before['utilization'] < 1 and at['utilization'] == 1
assert before['queue_multiplier'] > at['queue_multiplier']
print('Source queue boundary (not a valid latency predictor):', {
    'rho_before': before['utilization'], 'multiplier_before': before['queue_multiplier'],
    'rho_at': at['utilization'], 'multiplier_at': at['queue_multiplier'],
})

rows, selected, rejected = namespace['v2_09_tradeoff_candidates'](packet, 1, 5, 2, 6615)
print('Original fixture candidates:')
for row in rows:
    print({k: row[k] for k in ['id', 'p99_ms', 'headroom', 'daily_cost', 'quality_loss_pct', 'feasible', 'failures']})
print('Source-selected candidate:', selected['id'], '; feasible:', selected['feasible'])

# Original archives and repaired readable wrappers must agree exactly.
lab_wrapper = (ROOT / '05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_09_perf_engineering.md').read_text()
assert lab_wrapper.split('## Complete original Marimo source\n\n```python\n', 1)[1].rsplit('\n```', 1)[0].strip() == SOURCE.read_text().strip()
deck_wrapper = (ROOT / '05 Источники/Courses/Harvard ML Systems/slides/vol2/09_performance_engineering.md').read_text()
deck = (ROOT / '05 Источники/Originals/performance-practice-2026-10-26/harvard-deck09.tex').read_text()
assert deck_wrapper.split('## Complete original Beamer source\n\n```tex\n', 1)[1].rsplit('\n```', 1)[0].strip() == deck.strip()
print('PASS reviewed numeric helper checks and archive/wrapper identity; no lab UI, GPU or production run')
