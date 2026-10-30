"""Run every printed ranking block, then independently check DLRM mechanisms.

CPU only. This does not import the legacy DLRM training script, TorchRec,
FBGEMM or a distributed backend, and does not measure recommendation quality.
"""
from pathlib import Path
import ast
import hashlib
import re
import runpy
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
result = runpy.run_path(str(ROOT / "publishing/tools/check_recommender_ranking_examples.py"))
scope = result["scope"]
chapter = result["content"].decode()
original = (ROOT / "05 Источники/Originals/dlrm-2026-10-26/dlrm_s_pytorch.py").read_text()
assert hashlib.sha256(original.encode()).hexdigest() == "9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5"
source_method = next(n for n in ast.walk(ast.parse(original))
                     if isinstance(n, ast.FunctionDef) and n.name == "interact_features")
printed_method = next(n for n in ast.walk(ast.parse(result["blocks"][15]))
                      if isinstance(n, ast.FunctionDef) and n.name == "interact_features")
assert ast.get_source_segment(original, source_method) == ast.get_source_segment(
    result["blocks"][15], printed_method)
import json
record = next(r for r in json.loads((ROOT / "docs/audit/2026-10-29-selection-gaps-additions.json").read_text())["files"] if r["marker"] == "cold-ids")
addition = re.search(r"<!-- cold-ids-addition: start -->.*?<!-- cold-ids-addition: end -->\n\n", chapter, re.S).group()
assert hashlib.sha256(addition.encode()).hexdigest() == record["block_sha256"]
previous = chapter.replace(addition, "").replace("last_updated: 2026-10-29", "last_updated: 2026-10-26")
assert hashlib.sha256(previous.encode()).hexdigest() == record["before_sha256"]
restored = re.sub(r"<!-- dlrm 2026-10-26 -->\n.*?<!-- /dlrm -->\n\n", "", previous, flags=re.S)
restored = restored.replace("last_updated: 2026-10-26", "last_updated: 2026-10-25")
assert hashlib.sha256(restored.encode()).hexdigest() == "ecec78266fddd5dcfa2463d9c448f013ae63e7101e8f50052fe4e3895c092f25"
print("PASS original source method byte-for-byte and all 839 accepted chapter lines preserved")

g = scope["dlrm_g"]
dtype = torch.float64
expected_user = torch.zeros((4, 2), dtype=dtype)
expected_user[2] = g * torch.tensor([-.1, .3], dtype=dtype)
torch.testing.assert_close(scope["dlrm_user"].weight.grad, expected_user)
torch.testing.assert_close(scope["dlrm_bottom"][0].weight.grad,
    (g * torch.tensor([[.4], [.5]], dtype=dtype)) @ scope["dlrm_dense_input"])
torch.testing.assert_close(scope["dlrm_bottom"][0].bias.grad,
    g * torch.tensor([.4, .5], dtype=dtype))
torch.testing.assert_close(scope["dlrm_top"].weight.grad, g * scope["dlrm_r"].detach())
torch.testing.assert_close(scope["dlrm_top"].bias.grad, g)
assert torch.count_nonzero(scope["dlrm_products"].grad[1:]) == 0
assert torch.count_nonzero(scope["dlrm_users"].grad[1:]) == 0
print("PASS every bottom/top parameter gradient, user rows and unused batch gradients")

generator = torch.Generator().manual_seed(20261026)
interaction_class = scope["DLRMInteraction"]
cases = 0
for batch in (1, 3):
    for fields in (1, 2, 4):
        for width in (1, 2, 5):
            for diagonal in (False, True):
                module = interaction_class()
                module.arch_interaction_itself = diagonal
                vectors = torch.randn(batch, fields + 1, width,
                    generator=generator, dtype=dtype, requires_grad=True)
                actual = module.interact_features(vectors[:, 0], list(vectors[:, 1:].unbind(1)))
                pairs = [(i, j) for i in range(fields + 1)
                         for j in range(i + int(diagonal))]
                expected = torch.cat([vectors[:, 0], torch.stack([
                    (vectors[:, i] * vectors[:, j]).sum(-1) for i, j in pairs], dim=1)], dim=1)
                torch.testing.assert_close(actual, expected)
                assert actual.shape == (batch, width + len(pairs))
                weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
                actual_grad, = torch.autograd.grad((actual * weights).sum(), vectors)
                manual = torch.zeros_like(vectors)
                manual[:, 0] += weights[:, :width]
                for k, (i, j) in enumerate(pairs):
                    coefficient = weights[:, width + k, None]
                    manual[:, i] += coefficient * vectors[:, j].detach()
                    manual[:, j] += coefficient * vectors[:, i].detach()
                torch.testing.assert_close(actual_grad, manual)
                module.arch_interaction_op = "cat"
                torch.testing.assert_close(module.interact_features(
                    vectors[:, 0], list(vectors[:, 1:].unbind(1))), vectors.flatten(1))
                cases += 1
module = interaction_class()
small = torch.randn(2, 3, 2, generator=generator, dtype=dtype, requires_grad=True)
assert torch.autograd.gradcheck(lambda x: module.interact_features(
    x[:, 0], list(x[:, 1:].unbind(1))), (small,))
print(f"PASS {cases} dot/self-dot/cat configurations, analytic gradients and finite-difference gradcheck")

for lengths in ([0, 0, 0], [2, 0, 2], [1, 4, 3], [5, 1, 0]):
    weight = torch.randn(7, 4, generator=generator, dtype=dtype, requires_grad=True)
    ids = torch.randint(7, (sum(lengths),), generator=generator)
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    pooled = F.embedding_bag(ids, weight, offsets, mode="sum", include_last_offset=True)
    upstream = torch.randn(3, 4, generator=generator, dtype=dtype)
    actual, = torch.autograd.grad((pooled * upstream).sum(), weight)
    expected = torch.zeros_like(weight)
    for b in range(3):
        bag = ids[offsets[b]:offsets[b + 1]]
        torch.testing.assert_close(pooled[b], weight[bag].sum(0))
        for index in bag:
            expected[index] += upstream[b]
    torch.testing.assert_close(actual, expected)
print("PASS empty/multihot/repeated-ID bag values and frequency-counted gradients")

for size in (3, 7):
    for width in (2, 5):
        table = torch.randn(size, width, generator=generator, dtype=dtype)
        for count in (0, 1, 8, 20):
            ids = torch.randint(size, (count,), generator=generator)
            selected = table[ids].sum(0)
            for split in range(1, size):
                left = table[:split][ids[ids < split]].sum(0)
                right = table[split:][ids[ids >= split] - split].sum(0)
                torch.testing.assert_close(left + right, selected)
                if count:
                    torch.testing.assert_close((left + right) / count, table[ids].mean(0))
            for split in range(1, width):
                torch.testing.assert_close(torch.cat([
                    table[:, :split][ids].sum(0), table[:, split:][ids].sum(0)]), selected)
print("PASS row/column reconstruction including empty bags, repeated IDs and uneven shard counts")
print("CPU correctness only: no CUDA/NCCL/TorchRec/Criteo run or performance claim")
