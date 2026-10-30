#!/usr/bin/env python3
"""CPU checks for chapter 44f; no GPU, NCCL or distributed timing claims.

NumPy is the only dependency. The reference evaluates every expert on every
token; the routed implementation packs real assignments, groups by owner/expert,
splits the hidden width over virtual TP ranks, and inverts the permutation.
The simulation checks algebra, not a library's collective implementation.
"""

from itertools import product
import numpy as np


def close(actual, expected, label, atol=2e-11):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=atol,
                               err_msg=label)


def routing(x, router, k=2):
    logits = x @ router
    selected = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    chosen = np.take_along_axis(logits, selected, axis=1)
    exps = np.exp(chosen - chosen.max(axis=1, keepdims=True))
    return selected, exps / exps.sum(axis=1, keepdims=True)


def router_backward(x, router, selected, gates, dg):
    chosen_dz = gates * (dg - (gates * dg).sum(axis=1, keepdims=True))
    dz = np.zeros((len(x), router.shape[1]))
    np.put_along_axis(dz, selected, chosen_dz, axis=1)
    return dz @ router.T, x.T @ dz


def packed_order(selected, owners):
    """Value at packed position p is original assignment r = token*k + slot."""
    flat = selected.ravel()
    return np.array(sorted(range(len(flat)),
                           key=lambda r: (owners[flat[r]], flat[r], r)))


def routed_forward(x, w1, w2, selected, gates, owners, tp=1, corrupt=False):
    tokens, k = selected.shape
    width = w1.shape[-1]
    if width % tp:
        raise ValueError("expert hidden width must divide TP")
    order = packed_order(selected, owners)
    inverse = np.argsort(order)
    packed_x = x[order // k]
    packed_e = selected.ravel()[order]
    packed_outputs = np.zeros_like(packed_x)
    for expert in range(len(w1)):
        rows = np.flatnonzero(packed_e == expert)
        # Empty experts have a valid zero-row matrix, including under TP.
        for shard in np.array_split(np.arange(width), tp):
            hidden = np.tanh(packed_x[rows] @ w1[expert][:, shard])
            packed_outputs[rows] += hidden @ w2[expert][shard]
    if corrupt:
        # A shape-preserving bug: values lose their assignment identities.
        packed_outputs = np.roll(packed_outputs, 1, axis=0)
    by_assignment = packed_outputs[inverse].reshape(tokens, k, -1)
    y = (by_assignment * gates[..., None]).sum(axis=1)
    cache = (order, inverse, packed_x, packed_e, by_assignment, tp)
    return y, cache


def routed_backward(x, w1, w2, selected, gates, upstream, cache):
    order, inverse, packed_x, packed_e, by_assignment, tp = cache
    k = selected.shape[1]
    packed_dy = (upstream[:, None, :] * gates[..., None]).reshape(-1, x.shape[1])[order]
    packed_dx = np.zeros_like(packed_x)
    dw1, dw2 = np.zeros_like(w1), np.zeros_like(w2)
    for expert in range(len(w1)):
        rows = np.flatnonzero(packed_e == expert)
        for shard in np.array_split(np.arange(w1.shape[-1]), tp):
            a = packed_x[rows] @ w1[expert][:, shard]
            hidden = np.tanh(a)
            dh = packed_dy[rows] @ w2[expert][shard].T
            da = dh * (1.0 - hidden**2)
            dw1[expert][:, shard] = packed_x[rows].T @ da
            dw2[expert][shard] = hidden.T @ packed_dy[rows]
            packed_dx[rows] += da @ w1[expert][:, shard].T
    # Inverse dispatch restores assignment identities, then sums the k paths.
    dx = packed_dx[inverse].reshape(len(x), k, -1).sum(axis=1)
    dg = (by_assignment * upstream[:, None, :]).sum(axis=-1)
    return dx, dw1, dw2, dg


def dense_reference(x, w1, w2, router, target):
    """Independent dense all-expert evaluation with explicit scalar routes."""
    selected, gates = routing(x, router)
    hidden = np.tanh(np.einsum("td,edf->tef", x, w1))
    every_output = np.einsum("tef,efd->ted", hidden, w2)
    y = np.zeros_like(x)
    for token, choices in enumerate(selected):
        for slot, expert in enumerate(choices):
            y[token] += gates[token, slot] * every_output[token, expert]
    upstream = y - target
    dx, dw1, dw2 = np.zeros_like(x), np.zeros_like(w1), np.zeros_like(w2)
    dg = np.zeros_like(gates)
    for token, choices in enumerate(selected):
        for slot, expert in enumerate(choices):
            dy = gates[token, slot] * upstream[token]
            dg[token, slot] = upstream[token] @ every_output[token, expert]
            dw2[expert] += np.outer(hidden[token, expert], dy)
            da = (dy @ w2[expert].T) * (1 - hidden[token, expert]**2)
            dw1[expert] += np.outer(x[token], da)
            dx[token] += da @ w1[expert].T
    dx_router, dr = router_backward(x, router, selected, gates, dg)
    return y, (dx + dx_router, dw1, dw2, dr), (selected, gates)


def test_worked_example():
    x = np.array([[1., 0], [0, 1], [1, 1], [2, -1]])
    selected = np.array([[0, 2], [1, 0], [2, 1], [0, 2]])
    gates = np.array([[.75, .25], [.6, .4], [.5, .5], [.2, .8]])
    weights = np.array([np.eye(2), [[2, 0], [0, 1]], [[1, 1], [0, 1]], np.eye(2)])
    owners = np.array([0, 0, 1, 1])
    order = packed_order(selected, owners)
    assert order.tolist() == [0, 3, 6, 2, 5, 1, 4, 7]
    assert np.argsort(order).tolist() == [0, 5, 3, 1, 6, 4, 2, 7]
    packed = np.array([x[r // 2] @ weights[selected.ravel()[r]] for r in order])
    restored = packed[np.argsort(order)].reshape(4, 2, 2)
    y = (restored * gates[..., None]).sum(axis=1)
    close(y, [[1, .25], [0, 1], [1.5, 1.5], [2, .6]], "manual example")
    messages = np.zeros((2, 2), dtype=int)
    for token, choices in enumerate(selected):
        for expert in choices:
            messages[token // 2, owners[expert]] += 1
    assert messages.tolist() == [[3, 1], [2, 2]]
    assert np.bincount(selected.ravel(), minlength=4).tolist() == [3, 2, 3, 0]
    print("PASS worked outputs, assignment inverse, uneven dispatch, empty expert")


def numerical_gradient(array, loss, eps=1e-6):
    grad = np.empty_like(array)
    for idx in np.ndindex(array.shape):
        old = array[idx]
        array[idx] = old + eps
        plus = loss()
        array[idx] = old - eps
        minus = loss()
        array[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def test_outputs_and_gradients():
    rng = np.random.default_rng(44106)
    x = rng.uniform(.2, 1.5, (7, 3))
    w1 = rng.normal(scale=.4, size=(4, 3, 6))
    w2 = rng.normal(scale=.4, size=(4, 6, 3))
    router = rng.normal(size=(3, 4))
    router[:, 3] = -5  # Truly empty expert; positive x guarantees no top-2 use.
    owners = np.array([0, 0, 1, 1])
    target = rng.normal(size=x.shape)
    ref_y, ref_grads, (selected, gates) = dense_reference(x, w1, w2, router, target)
    assert 3 not in selected
    logits = np.sort(x @ router, axis=1)
    assert np.min(logits[:, -2] - logits[:, -3]) > 1e-3
    for tp in [1, 2, 3]:
        y, cache = routed_forward(x, w1, w2, selected, gates, owners, tp=tp)
        dx, dw1, dw2, dg = routed_backward(x, w1, w2, selected, gates, y-target, cache)
        dx_router, dr = router_backward(x, router, selected, gates, dg)
        close(y, ref_y, f"outputs TP={tp}")
        for name, actual, expected in zip(["x", "W1", "W2", "router"],
                                         [dx+dx_router, dw1, dw2, dr], ref_grads):
            close(actual, expected, f"{name} gradient TP={tp}")
        assert not np.any(dw1[3]) and not np.any(dw2[3])
    moved_y, _ = routed_forward(x, w1, w2, selected, gates, 1-owners, tp=2)
    close(moved_y, ref_y, "EP owner relabeling")
    wrong_y, _ = routed_forward(x, w1, w2, selected, gates, owners, corrupt=True)
    assert wrong_y.shape == ref_y.shape and not np.allclose(wrong_y, ref_y)
    try:
        routed_forward(x, w1, w2, selected, gates, owners, tp=4)
    except ValueError:
        pass
    else:
        raise AssertionError("non-divisible TP was accepted")
    print("PASS dense versus routed forward/backward, TP splits, owner relabeling, negative inverse")

    baseline_selection = selected.copy()
    def loss():
        choices, probs = routing(x, router)
        assert np.array_equal(choices, baseline_selection), "finite difference crossed top-k boundary"
        y, _ = routed_forward(x, w1, w2, choices, probs, owners, tp=2)
        return .5 * np.sum((y-target)**2)

    errors = []
    for array, expected in zip([x, w1, w2, router], ref_grads):
        numeric = numerical_gradient(array, loss)
        close(numeric, expected, "finite differences", atol=2e-8)
        errors.append(float(np.max(np.abs(numeric-expected))))
    # Do not silently omit the route-dependent input-gradient path.
    assert np.linalg.norm(dx_router) > 1e-5
    print("PASS finite differences x/W1/W2/router; max abs errors:", errors)


def test_load_and_capacity():
    counts = np.array([3, 2, 3, 0])
    rank_counts = np.array([counts[:2].sum(), counts[2:].sum()])
    assert counts.max()/counts.mean() == 1.5
    assert rank_counts.max()/rank_counts.mean() == 1.25
    loads = np.array([4, 4, 0, 0])
    assert max(loads[:2].sum(), loads[2:].sum()) == 8
    assert max(loads[[0, 2]].sum(), loads[[1, 3]].sum()) == 4
    c = int(np.ceil(1.5 * counts.sum()/len(counts)))
    assert c == 3 and (np.maximum(counts-c, 0)).sum() == 0
    assert len(counts)*c-counts.sum() == 4
    diagram = np.array([3, 6, 5, 2])
    retained = np.minimum(diagram, 4)
    assert diagram.sum() == 16 and retained.sum() == 13
    assert np.maximum(diagram-4, 0).sum() == 3
    assert 4*4-retained.sum() == 3
    print("PASS independent expert/rank imbalance; drop, padding and capacity arithmetic")


def rank(d, e, s, t):
    return (((d*8+e)*8+s)*8+t)


def test_process_mesh():
    coordinates = list(product(range(2), range(8), range(8), range(8)))
    assert sorted(rank(*coord) for coord in coordinates) == list(range(1024))
    assert rank(1, 3, 5, 6) == 750
    memberships = {name: {} for name in ["tp", "ep", "pp", "edp", "dense_dp"]}
    for d, e, s, t in coordinates:
        keys = dict(tp=(d, e, s), ep=(d, s, t), pp=(d, e, t), edp=(e, s, t), dense_dp=(s, t))
        for name, key in keys.items():
            memberships[name].setdefault(key, []).append(rank(d, e, s, t))
    sizes = dict(tp=8, ep=8, pp=8, edp=2, dense_dp=16)
    for name, groups in memberships.items():
        assert len(groups) == 1024//sizes[name]
        assert all(len(group) == sizes[name] for group in groups.values())
        assert sorted(r for group in groups.values() for r in group) == list(range(1024))
    for dense_group in memberships["dense_dp"].values():
        # At fixed stage and TP-shard, 16 replicas share the dense matrix.
        assert len(dense_group) == 2*8
    assert 8*1*16*64 == 8192
    assert 8*16*16*8 == 16384
    print("PASS 1024-rank bijection; TP/EP/PP/EDP/dense-DP partitions; Llama products")


def test_tp_sp_message_ledger():
    # One stage and one EDP replica: 8 source TP-groups x 8 TP ranks.
    # Deterministic routing balances each sender over all owners and experts.
    counts = np.zeros((8, 8, 8, 8), dtype=int)  # source_ep, tp, dest_ep, local_expert
    identities = set()
    for source, tp, local_token in product(range(8), range(8), range(128)):
        token = source*1024 + tp*128 + local_token
        for slot in range(2):
            expert = (2*local_token+slot) % 64
            counts[source, tp, expert//8, expert%8] += 1
            identities.add((token, slot))
    assert len(identities) == 8192*2
    assert np.all(counts.sum(axis=(2, 3)) == 256)  # source ranks
    assert np.all(counts.sum(axis=(0, 3)) == 256)  # after EP, fixed TP
    assert np.all(counts.sum(axis=(0, 1)) == 256)  # individual expert
    assert np.all(counts.sum(axis=(0, 1, 3)) == 2048)  # owner, after TP AG
    assert np.all(counts.sum(axis=3) == 32)  # every endpoint pair at fixed TP
    row_bytes = 4096*2
    packed = 256*row_bytes
    gathered = 2048*row_bytes
    remote = np.empty((8, 8), dtype=int)
    for source, tp in product(range(8), range(8)):
        remote[source, tp] = (counts[source, tp].sum()-counts[source, tp, source].sum())*row_bytes
    mib = 1024**2
    assert packed == 2*mib and gathered == 16*mib
    assert np.all(remote == 1.75*mib)
    tp_ring_sent = gathered*7//8
    assert tp_ring_sent == 14*mib
    assert 2*remote[0, 0]+2*tp_ring_sent == 31.5*mib
    assert counts.sum()*row_bytes == 128*mib
    assert remote[0].sum() == 14*mib  # Aggregate across TP, NOT single-rank EP.
    print("PASS exact route counts and BF16 ledger: EP 1.75, TP AG/RS 14, forward 31.5 MiB/rank")


def test_edp_gradient_normalization():
    # 16 dense-DP sources, equal token counts: a local mean from each source.
    rng = np.random.default_rng(44116)
    x = rng.normal(size=(2, 8, 5, 3))
    target = rng.normal(size=(2, 8, 5, 2))
    weight = rng.normal(size=(3, 2))
    prediction = x @ weight
    per_source = np.einsum("debi,debj->deij", x, prediction-target)/5
    global_reference = np.einsum("debi,debj->ij", x, prediction-target)/(2*8*5)
    per_edp = per_source.sum(axis=1)  # Expert already consumed eight sources.
    correct = per_edp.sum(axis=0)/16
    wrong = per_edp.mean(axis=0)
    close(correct, global_reference, "expert global mean")
    close(wrong, 8*global_reference, "naive EDP mean is eight times too large")
    close((per_edp*(2/16)).mean(axis=0), global_reference, "Megatron AVG prescale")
    close((per_edp/16).sum(axis=0), global_reference, "Megatron SUM prescale")
    assert not np.allclose(wrong, global_reference)
    print("PASS expert EDP SUM/AVG normalization versus dense global-loss reference")


def main():
    test_worked_example()
    test_outputs_and_gradients()
    test_load_and_capacity()
    test_process_mesh()
    test_tp_sp_message_ledger()
    test_edp_gradient_normalization()
    print("All CPU algebra checks passed. No distributed/GPU/NCCL execution or timing.")


if __name__ == "__main__":
    main()
