#!/usr/bin/env python3
"""Execute every chapter block and test independent numerical/mechanism cases."""
from collections import Counter
import ast
import hashlib
from math import exp, isclose, log, sqrt
from pathlib import Path
import re
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md"
text = CHAPTER.read_text()
assert chr(0xA4) not in text
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, flags=re.S)
assert len(blocks) == 7, len(blocks)
namespaces = []
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    namespaces.append(namespace)
    print("PASS published standalone block", number)

# Keep the five earlier examples in their original order, while executing
# and validating both source-derived data programs independently below.
window_namespace = next(n for n in namespaces if "get_centers_and_contexts" in n)
batch_namespace = next(n for n in namespaces if "batchify" in n)
namespaces = [n for n in namespaces
              if "get_centers_and_contexts" not in n and "batchify" not in n]

n = namespaces[0]
expected = np.array([[0, 0, 2, 0], [0, 0, 1, 1], [2, 1, 0, 0], [0, 1, 0, 0]])
np.testing.assert_equal(n["X"], expected)
assert Counter(n["pairs"]) == {(0, 2): 2, (2, 0): 2, (1, 2): 1, (2, 1): 1, (1, 3): 1, (3, 1): 1}
assert (2, 0) in n["pairs"] and (2, 1) in n["pairs"]
np.testing.assert_equal(n["X"].sum(0), [2, 2, 3, 1])
np.testing.assert_equal(n["X"].sum(1), [2, 2, 3, 1])
cos = lambda a, b: np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
assert isclose(cos(expected[0], expected[1]), 1/sqrt(2))
assert cos(expected[0], expected[2]) == 0 and expected[0, 2] == 2
assert isclose(n["ppmi"][0, 2], log(8/3))
assert isclose(n["ppmi"][1, 2], log(4/3))
assert isclose(n["ppmi"][1, 3], log(4))
assert np.all(n["ppmi"][expected == 0] == 0)
assert log((1*12)/(6*6)) < 0  # A nonzero pair can still have negative PMI.
np.testing.assert_allclose(n["singular"], [1.440847850045, 1.440847850045, .943693020269, .943693020269], atol=1e-9)
assert isclose(np.linalg.norm(n["ppmi"]-n["reconstruction"]), 1.3345834677327886)
assert np.linalg.matrix_rank(n["reconstruction"]) == 2
assert 100_000**2*4 == 40_000_000_000
print("PASS directional boundaries, co-occurrence vs similarity, PMI/PPMI, SVD dimensions/error")

# Full-softmax skip-gram: compare every row against the analytic gradient.
v = torch.tensor([1., .5], dtype=torch.float64, requires_grad=True)
u = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64, requires_grad=True)
scores = u @ v
probs = scores.softmax(0).detach()
loss = F.cross_entropy(scores[None], torch.tensor([2]))
dv, du = torch.autograd.grad(loss, (v, u))
g = probs.clone(); g[2] -= 1
torch.testing.assert_close(du, g[:, None]*v.detach()[None])
torch.testing.assert_close(dv, probs @ u.detach()-u.detach()[2])
assert torch.autograd.gradcheck(lambda x, y: F.cross_entropy((y @ x)[None], torch.tensor([2])), (v, u))
# CBOW uses the mean, not a position-dependent feature or an unscaled sum.
n = namespaces[1]
torch.testing.assert_close(n["logits"].detach(), torch.tensor([0., .5, 1., -.5], dtype=torch.float64))
assert not torch.allclose((n["U"] @ (2*n["h"])).softmax(0), n["logits"].softmax(0))
repeated = torch.tensor([0, 0])
e = n["E"].detach().clone().requires_grad_()
mean = e[repeated].mean(0)
mean.sum().backward()
torch.testing.assert_close(e.grad[0], torch.ones(2, dtype=e.dtype))
assert torch.count_nonzero(e.grad[1:]) == 0
print("PASS full-softmax gradients and CBOW permutation, scale and repeated-input accumulation")

n = namespaces[2]
assert isclose(n["before"].item(), 2.006408868078168, rel_tol=1e-12)
assert isclose(n["v"].item(), .9268941421369995, rel_tol=1e-12)
assert isclose(n["u"][0].item(), .05)
assert isclose(n["u"][1].item(), .9268941421369995, rel_tol=1e-12)
assert isclose(n["after"].item(), 1.8825151930666, rel_tol=1e-12)
def negative_loss(vector, table, negatives):
    return -F.logsigmoid(table[0] @ vector)-F.logsigmoid(-(table[negatives] @ vector)).sum()
v0 = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
u0 = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64, requires_grad=True)
single = torch.autograd.grad(negative_loss(v0, u0, torch.tensor([1])), u0)[0]
double = torch.autograd.grad(negative_loss(v0, u0, torch.tensor([1, 1])), u0)[0]
torch.testing.assert_close(double[1], 2*single[1])
collision = torch.autograd.grad(negative_loss(v0, u0, torch.tensor([0])), u0)[0]
torch.testing.assert_close(collision[0], torch.zeros(1, dtype=u0.dtype))
assert torch.autograd.gradcheck(lambda a, b: negative_loss(a,b,torch.tensor([1])), (v0,u0))
assert not isclose(torch.sigmoid(u0 @ v0).sum().item(), 1)
frequencies = np.array([2., 2., 3., 1.])
q = frequencies**.75 / np.sum(frequencies**.75)
np.testing.assert_allclose(q, [.25316414, .25316414, .34313943, .15053230], atol=1e-8)
assert q[3] > 1/8 and isclose(q.sum(), 1)
assert isclose(sqrt(1e-5/.01), .03162277660168379)
A, B = 1., 2*2*q[3]
optimum = log(A/B)
assert isclose(optimum, .5072832647094724)
assert isclose(A*(1-1/(1+exp(-optimum)))-B/(1+exp(-optimum)), 0, abs_tol=1e-15)
assert isclose(log(1/(2*2*(1/8))), log(4)-log(2))
print("PASS SGNS values, sparse rows, repeated/colliding negatives, q and PMI optimum")

# Hierarchical softmax: each leaf probability and total normalization.
left_root, left_child, right_child = .6, .5, .25
leaf = np.array([left_root*left_child, left_root*(1-left_child),
                 (1-left_root)*right_child, (1-left_root)*(1-right_child)])
np.testing.assert_allclose(leaf, [.3, .3, .1, .3])
assert isclose(leaf.sum(), 1)
for p in [.01, .2, .9]:
    assert isclose(p+(1-p), 1)
node_logits = torch.tensor([log(.6/.4), log(.5/.5), log(.25/.75)],
                           dtype=torch.float64, requires_grad=True)
path_loss = -F.logsigmoid(-node_logits[0])-F.logsigmoid(node_logits[2])
path_loss.backward()
assert isclose(path_loss.item(), log(10))
torch.testing.assert_close(node_logits.grad, torch.tensor([.6, 0., -.75], dtype=node_logits.dtype))
print("PASS tree branch products and normalized leaf mass")

n = namespaces[3]
weight = (.02)**.75
assert isclose(weight, .053182958969449884)
assert isclose(n["before"].item(), .02555191292596024)
for name in ("u", "b", "context_bias"):
    assert isclose(n[name].item(), .007372723612701888)
assert isclose(n["v"].item(), 1)
assert n["objective"]() < n["before"]
# Zero counts must be selected away before log, not repaired by multiplication.
counts = torch.tensor([0., 2., 100., 1000.], dtype=torch.float64)
weights = (counts/100).pow(.75).clamp(max=1)
assert torch.isnan((weights * counts.log().square())[0])
keep = counts > 0
assert torch.isfinite((weights[keep] * counts[keep].log().square()).sum())
assert weights[2] == weights[3] == 1 and counts[2].log() != counts[3].log()
print("PASS GloVe residual, gradients, simultaneous update, positive mask and cap")

# Pair scores survive inverse transformations, input cosines need not.
V = np.array([[1., 1.], [1., -1.]])
U = np.array([[.5, 1.], [2., -1.]])
transform = np.diag([2., 1.])
changed_v = V @ transform.T
changed_u = U @ np.linalg.inv(transform)
np.testing.assert_allclose(changed_v @ changed_u.T, V @ U.T)
assert isclose(cos(V[0], V[1]), 0, abs_tol=1e-15)
assert isclose(cos(changed_v[0], changed_v[1]), .6)
assert isclose(cos(np.array([1.,0.]), np.array([100.,1.])), 100/sqrt(10001))
assert 2*100_000*300 == 60_000_000
word = "<where>"
assert [word[i:i+3] for i in range(len(word)-2)] == ["<wh", "whe", "her", "ere", "re>"]
print("PASS geometry counterexample, parameter count, fastText overlapping fragments")

# The shared input/output table receives both gradient paths. No Transformer
# block is present in this deliberately isolated, three-token example.
n = namespaces[4]
initial = n["initial"]
probabilities = (initial @ initial[0]).softmax(0)
torch.testing.assert_close(probabilities,
    torch.tensor([.6652409557748218, .2447284710547976, .09003057317038045], dtype=initial.dtype))
assert isclose(n["loss"].item(), 1.4076059644443804)
residual = probabilities.clone(); residual[1] -= 1
direct = residual[:, None] * initial[0][None]
indirect = torch.zeros_like(initial); indirect[0] = residual @ initial
torch.testing.assert_close(n["output_table"].grad, direct)
torch.testing.assert_close(n["input_table"].grad, indirect)
torch.testing.assert_close(n["tied"].grad, direct + indirect)
torch.testing.assert_close(n["tied"].grad[0],
    torch.tensor([1.240451338379263, -.7552715289452023], dtype=initial.dtype))
torch.testing.assert_close(n["tied"].detach(), initial - .1*(direct + indirect))
assert isclose(n["after"].item(), 1.175168994008524)
assert torch.autograd.gradcheck(
    lambda e: F.cross_entropy((e @ e[0])[None], torch.tensor([1])),
    (initial.clone().requires_grad_(),))
assert torch.count_nonzero(direct[1:]) == 2
assert torch.count_nonzero(indirect[1:]) == 0
print("PASS tied embeddings: probabilities, two gradient paths, all rows, SGD and finite differences")

# Arithmetic of the chapter's assigned fastText components. This is not a
# call to the fastText binary and not evidence of learned semantic quality.
fragments = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.]])
whole_word = np.array([2., 2.])
np.testing.assert_equal(fragments.sum(0), [1., 1.])
summed = fragments.sum(0) + whole_word
averaged = summed / (len(fragments) + 1)
np.testing.assert_equal(summed, [3., 3.])
np.testing.assert_equal(averaged, [.5, .5])
assert isclose(np.linalg.norm(averaged), 1/sqrt(2))
assert isclose(cos(summed, averaged), 1)
print("PASS fastText worked arithmetic: sum, mean and separate L2 normalization")

archive = ROOT / "05 Источники/Originals/word-embeddings-2026-10-22"
for name, expected_sha in {
    "fasttext.cc": "593200ed00fe048eb48076829bfec18bd0c3f71241bcb12286ec64bb5f58dbf7",
    "dictionary.cc": "5cffc5985bf9267ab309f32335135527f8cfc09f19c3d305290408ef02836a3b",
    "LICENSE": "d9cfdfdf548c48712f05c67eb76c79e7c9f45cba78db50794b780e3564b85ef7",
}.items():
    assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == expected_sha
print("PASS pinned fastText v0.9.2 source and license hashes; source inspected, binary not executed")
# The two data programs retain the pinned course functions. Only backend-neutral
# d2l tensor/reshape calls in batchify are adapted to PyTorch, as disclosed.
data_source = ROOT / "05 Источники/Originals/word-embeddings-2026-11-03/d2l-word-embedding-dataset.md"
source_text = data_source.read_text()
assert hashlib.sha256(data_source.read_bytes()).hexdigest() == "106fd54974ae5a9dc5d37d71aee1b01def9aee6b1168df74bd01cae11a053ec4"
source_blocks = re.findall(r"\x60\x60\x60\{\.python \.input\}\n(.*?)\n\x60\x60\x60",
                           source_text, flags=re.S)

def source_fragment(name):
    for block in source_blocks:
        for node in ast.parse(block).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return ast.get_source_segment(block, node)
    raise AssertionError(name)

def published_fragment(name):
    for block in blocks:
        for node in ast.parse(block).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(block, node)
    raise AssertionError(name)

assert published_fragment("get_centers_and_contexts") == source_fragment("get_centers_and_contexts")
adapted_batch = source_fragment("batchify").replace(
    "d2l.reshape(d2l.tensor(centers), (-1, 1))",
    "torch.tensor(centers).reshape((-1, 1))").replace("d2l.tensor(", "torch.tensor(")
assert published_fragment("batchify") == adapted_batch

extract = window_namespace["get_centers_and_contexts"]
cases = 0
for maximum in range(1, 6):
    for size in range(10):
        line = list(range(size))
        corpus = [[], [99], line, [100, 101]]
        for radius in range(1, maximum + 1):
            with patch.object(window_namespace["random"], "randint", return_value=radius):
                centers, contexts = extract(corpus, maximum)
            expected = [(word, [other for j, other in enumerate(sentence)
                                if 0 < abs(i-j) <= radius])
                        for sentence in corpus if len(sentence) >= 2
                        for i, word in enumerate(sentence)]
            assert list(zip(centers, contexts)) == expected
            cases += 1
for maximum in range(1, 10):
    corpus = [list(range(2*maximum+1))]
    middle = maximum
    for distance in range(1, maximum+1):
        included = 0
        for radius in range(1, maximum+1):
            with patch.object(window_namespace["random"], "randint", return_value=radius):
                _, contexts = extract(corpus, maximum)
            included += middle + distance in contexts[middle]
        assert isclose(included/maximum, (maximum-distance+1)/maximum)
with patch.object(window_namespace["random"], "randint", return_value=1):
    assert extract([[5, 5]], 1) == ([5, 5], [[5], [5]])
print("PASS original context function;", cases, "boundary/radius cases; exact distance weights and repeated IDs")

# Run the original rejection code with a deterministic proposal fixture,
# avoiding a claim about Monte Carlo convergence or source PTB sample counts.
class TinyVocab:
    def __len__(self):
        return 5

    def to_tokens(self, i):
        return i

class Proposals:
    latest_weights = None

    def __init__(self, weights):
        type(self).latest_weights = weights
        self.index = 0

    def draw(self):
        stream = [2, 3, 1, 1, 4]
        result = stream[self.index % len(stream)]
        self.index += 1
        return result

original_noise = {"RandomGenerator": Proposals}
exec(source_fragment("get_negatives"), original_noise)
counter = {1: 16, 2: 9, 3: 4, 4: 1}
negative_rows = original_noise["get_negatives"]([[2, 3], [2, 2, 3]], TinyVocab(), counter, 2)
assert [len(row) for row in negative_rows] == [4, 6]
assert all(set(row) <= {1, 4} for row in negative_rows)
assert all(row.count(1) >= 2 for row in negative_rows)
np.testing.assert_allclose(Proposals.latest_weights, [counter[i]**.75 for i in range(1, 5)])
proposal = np.array([.4, .3, .2, .1])
eligible = np.array([True, False, False, True])
accepted = proposal * eligible
accepted /= accepted.sum()
np.testing.assert_allclose(accepted, [.8, 0, 0, .2])
assert not np.array_equal(accepted, proposal)
assert 1 - proposal[~eligible].sum() == .5
print("PASS original rejection code, all-context exclusion, repeated draws and conditional noise calculation")

batchify = batch_namespace["batchify"]
cases = 0
for count in range(1, 5):
    for positives in range(1, 5):
        for negatives in range(4):
            data = [(row+1, [2]*(positives+row), [3]*(negatives+row))
                    for row in range(count)]
            center_ids, pair_ids, valid, labels = batchify(data)
            width = max(len(c)+len(n) for _, c, n in data)
            assert tuple(pair_ids.shape) == (count, width)
            assert tuple(center_ids.shape) == (count, 1)
            for row, (center, context, noise) in enumerate(data):
                length = len(context)+len(noise)
                assert center_ids[row, 0].item() == center
                assert pair_ids[row, :length].tolist() == context + noise
                assert valid[row].sum().item() == length
                assert labels[row, :len(context)].tolist() == [1]*len(context)
                assert labels[row, len(context):].count_nonzero() == 0
                assert valid[row, length:].count_nonzero() == 0
            cases += 1
assigned = torch.tensor([[1., 1., 1., 1., 1., 1.],
                         [2., 2., 2., 2., 2., 0.]], dtype=torch.float64)
valid = batch_namespace["masks"].bool()
assert isclose(assigned[valid].mean().item(), 16/11)
assert isclose(((assigned*valid).sum(-1)/valid.sum(-1)).mean().item(), 1.5)
assert not isclose(16/11, 1.5)
assert isclose(batch_namespace["loss"].item(), log(2))
print("PASS source-faithful batching;", cases, "shape/label/mask cases; padding gradient and distinct reduction weights")
print("PASS all word embeddings examples; torch", torch.__version__)
print("Chapter SHA256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

