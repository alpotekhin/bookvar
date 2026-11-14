"""Compare published NumPy routing with the pinned, unmodified Gate on CPU.

The existing local source archive is required. No checkpoint, GPU kernel or
distributed collective is imported or run. This is an evidence check, not a
public-checkout dependency or a performance measurement.
"""
import ast
import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / '05 Источники/Originals/mla-2026-10-22/model.py'
expected_sha = 'fdb7995bc234badbf0011b5af8cf391be94343abc51305428cb9a04452116f02'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha
tree = ast.parse(source.read_text())
gate_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
scope = dict(torch=torch, nn=nn, Tuple=Tuple, ModelArgs=SimpleNamespace, linear=F.linear)
exec(compile(ast.Module(body=[gate_node], type_ignores=[]), str(source), 'exec'), scope)
Gate = scope['Gate']
ns = runpy.run_path(str(ROOT / 'publishing/tools/check_moe_examples.py'))['ns']
route = ns['node_limited_route']


def actual_gate(scores, k, groups, allowed, bias, scale=1.):
    # Constructor creates a bias only at dim=7168. Exercise that real branch;
    # use seven nonzero input coordinates to supply independently chosen logits.
    tokens, experts = scores.shape
    dim = 7168 if bias is not None else max(tokens, 8)
    args = SimpleNamespace(dim=dim, n_activated_experts=k,
                           n_expert_groups=groups, n_limited_groups=allowed,
                           score_func='sigmoid', route_scale=scale,
                           n_routed_experts=experts)
    gate = Gate(args).double()
    x = torch.zeros(tokens, dim, dtype=torch.float64)
    x[:, :tokens] = torch.eye(tokens, dtype=torch.float64)
    logits = np.log(scores / (1 - scores))
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[:, :tokens].copy_(torch.from_numpy(logits.T))
        if bias is not None:
            gate.bias.copy_(torch.from_numpy(bias))
        weights, ids = gate(x)
    return ids.numpy(), weights.numpy()


rng = np.random.default_rng(23)
count = 0
for groups, allowed, k in [(2, 1, 2), (4, 2, 4), (8, 4, 8),
                           (4, 2, 2), (4, 2, 3), (3, 2, 5), (4, 4, 4)]:
    scores = rng.uniform(.02, .98, (7, groups * 4))
    for bias in [None, np.zeros(groups * 4), rng.uniform(-.1, .1, groups * 4)]:
        top = 1 if bias is None else 2
        for scale in [.5, 1., 2.5]:
            selected, weights = route(scores, k, groups, allowed, bias, group_topk=top)
            ref_ids, ref_weights = actual_gate(scores, k, groups, allowed, bias, scale)
            np.testing.assert_array_equal(selected, ref_ids)
            np.testing.assert_allclose(weights * scale, ref_weights, atol=2e-15, rtol=2e-15)
            count += 1

counter = np.array([[.90, .02, .60, .59, .58, .57]])
paper_ids, _ = route(counter, 2, 3, 2)
released_ids, released_weights = route(counter, 2, 3, 2, np.zeros(6), group_topk=2)
ref_ids, ref_weights = actual_gate(counter, 2, 3, 2, np.zeros(6))
np.testing.assert_array_equal(paper_ids, [[0, 2]])
np.testing.assert_array_equal(released_ids, [[2, 3]])
np.testing.assert_array_equal(released_ids, ref_ids)
np.testing.assert_allclose(released_weights, ref_weights, atol=1e-15)
assert not np.array_equal(paper_ids, released_ids)
no_bias_ids, _ = actual_gate(counter, 2, 3, 2, None)
np.testing.assert_array_equal(no_bias_ids, paper_ids)
assert len(set(released_ids[0] // 2)) == 1  # selected experts need not fill each admitted group
print(f'PASS {count} pinned Gate comparisons, report/code counterexample, '
      'absent versus zero bias and non-quota selection; CPU only; source ' + expected_sha)
