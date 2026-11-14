"""Check chunkwise correction solves against the published token recurrence.

CPU NumPy only; supplied projections/gates, not trained models or GPU timing.
"""
from pathlib import Path
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(ROOT/'publishing/tools/check_sequence_examples.py'))['ns']
chunked, token = ns['gated_delta_chunked'], ns['gated_delta']
rng = np.random.default_rng(143)
count = 0
for dk, dv in [(1, 1), (2, 3), (4, 2)]:
    T = 7
    q, k = rng.normal(size=(2, T, dk))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.normal(size=(T, dv))
    initial = rng.normal(size=(dk, dv))
    for boundary in ['ordinary', 'zero_decay', 'zero_write', 'unit_write']:
        gamma = rng.uniform(.1, .99, T)
        beta = rng.uniform(.05, .95, T)
        if boundary == 'zero_decay':
            gamma[3] = 0.  # algebraic boundary; trained sigmoid gates are positive
        if boundary == 'zero_write':
            beta[2] = 0.
        if boundary == 'unit_write':
            beta[:] = 1.
        expected_y, expected_state = token(q, k, v, gamma, beta, initial)
        # Independent matrix-transition expansion, with ordered products.
        transitions = [
            g*(np.eye(dk)-b*np.outer(kt, kt))
            for g, b, kt in zip(gamma, beta, k)
        ]
        expanded = []
        for t in range(T):
            state = initial.copy()
            for j in range(t+1):
                state = transitions[j]@state + beta[j]*np.outer(k[j], v[j])
            expanded.append(q[t]@state)
        np.testing.assert_allclose(expanded, expected_y, atol=3e-14)
        for size in range(1, T+2):
            y, state = chunked(q, k, v, gamma, beta, size, initial)
            np.testing.assert_allclose(y, expected_y, atol=3e-14)
            np.testing.assert_allclose(state, expected_state, atol=3e-14)
            count += 1
        # All64 streaming partitions, not just uniform chunks.
        for mask in range(1 << (T-1)):
            ends = [i+1 for i in range(T-1) if mask & (1 << i)] + [T]
            starts = [0]+ends[:-1]
            carried, parts = initial.copy(), []
            for lo, hi in zip(starts, ends):
                y, carried = chunked(q[lo:hi], k[lo:hi], v[lo:hi],
                                     gamma[lo:hi], beta[lo:hi], 2, carried)
                parts.append(y)
            np.testing.assert_allclose(np.concatenate(parts), expected_y, atol=3e-14)
            np.testing.assert_allclose(carried, expected_state, atol=3e-14)
        # Future queries, keys and values must not change preceding outputs.
        for cut in range(1, T):
            changed_q, changed_k, changed_v = q.copy(), k.copy(), v.copy()
            changed_q[cut:] += 3
            changed_k[cut:] *= -2
            changed_v[cut:] += 100
            y, _ = chunked(changed_q, changed_k, changed_v, gamma, beta, 3, initial)
            np.testing.assert_allclose(y[:cut], expected_y[:cut], atol=3e-14)

# The printed two-key example: block-start-independent errors are wrong.
np.testing.assert_allclose(ns['corrections_block'][:, 0], [9., 16.2])
K = ns['keys_block']
D = ns['decay_matrix'](ns['g_block'])
lam = np.cumprod(ns['g_block'])
initial = ns['initial_block']
rhs = ns['b_block'][:, None]*(ns['values_block']-lam[:, None]*(K@initial))
wrong = lam[:, None]*(K@initial) + ((K@K.T)*D)@rhs
np.testing.assert_allclose(wrong[:, 0], [10., 22.7], atol=1e-14)
assert not np.allclose(wrong, ns['y_block'])
coupling = ns['b_block'][:, None]*np.tril((K@K.T)*D, -1)
np.testing.assert_allclose(coupling, [[0., 0.], [.3, 0.]])
wrong_sign = np.linalg.solve(np.eye(2)-coupling, rhs)
assert not np.allclose(wrong_sign, ns['corrections_block'])
try:
    chunked(K, K, ns['values_block'], ns['g_block'], ns['b_block'], 0)
    raise AssertionError('zero chunk size accepted')
except ValueError:
    pass
print(f'PASS {count} chunk-size comparisons, 768 stream partitions, 72 future controls, '
      'nonzero boundaries, zero gates, wrong-independent and wrong-sign controls; CPU only')
