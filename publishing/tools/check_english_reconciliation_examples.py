#!/usr/bin/env python3
"""Execute EN examples from the actual Markdown and independently check mechanisms.

CPU only. The five source-native serving fragments describing CUDA/runtime
objects are explicitly not executed; they require their pinned GPU runtimes.
"""
from pathlib import Path
import math
import re
import torch

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "en/00 Textbook"
torch.set_num_threads(1)
chapters = {
    "self-attention": (EN / "05 Attention and Transformer/02 Self-Attention — Q, K, V.md", 4, [0, 1, 2, 3]),
    "scheduling": (EN / "14 Inference and optimization/55b Scheduling — continuous batching, chunked prefill, and prefix caching.md", 6, [0, 1, 2, 3, 4, 5]),
    "serving": (EN / "14 Inference and optimization/55c Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer.md", 7, [5, 6]),
    "flashattention": (EN / "14 Inference and optimization/56 FlashAttention.md", 2, [0]),
}
loaded = {}
for name, (path, count, selected) in chapters.items():
    blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", path.read_text(), re.S)
    assert len(blocks) == count, (name, len(blocks))
    loaded[name] = []
    for index in selected:
        ns = {}
        exec(compile(blocks[index], f"{path}:python-block-{index + 1}", "exec"), ns)
        loaded[name].append(ns)
        print(f"PASS {name} standalone block {index + 1}")
print("NOT RUN serving source fragments 1–5: pinned-runtime/GPU or tensor-layout illustrations.")
print("NOT RUN FlashAttention block 2: CUDA-only backend diagnostic, no GPU available.")

sa = loaded["self-attention"][2]
torch.testing.assert_close(sa["A"][-1], torch.tensor(
    [.24825507825772306, .24825507825772306, .5034898434845538], dtype=torch.float64))
assert math.isclose(sa["before"], 2.0465561246435064)
assert math.isclose(sa["after"], 1.8381402347944618)
assert sa["WQ"] is not sa["WK"]
torch.testing.assert_close(sa["X"].detach(), torch.eye(3, dtype=torch.float64))
# The prior 3x2 X cannot generate the displayed V by one shared linear map.
bad_x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
displayed_v = torch.tensor([[1., 0.], [0., 2.], [3., 1.]], dtype=torch.float64)
assert torch.equal(bad_x[2], bad_x[0] + bad_x[1])
assert not torch.equal(displayed_v[2], displayed_v[0] + displayed_v[1])
print("PASS shared projections, gradient assertions, numerical outputs and SGD step.")

blocked = loaded["flashattention"][0]["blocked_attention"]
torch.manual_seed(7)
cases = 0
for n in (2, 7, 11):
    for d in (1, 4):
        for qb, kb in ((1, 1), (4, 3), (16, 2)):
            for causal in (False, True):
                q, k, v = [torch.randn(n, d, dtype=torch.float64, requires_grad=True) for _ in range(3)]
                actual, lse = blocked(q, k, v, q_block=qb, k_block=kb, causal=causal)
                scores = q @ k.T / math.sqrt(d)
                if causal:
                    scores = scores.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), -torch.inf)
                expected = scores.softmax(-1) @ v
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(lse, scores.logsumexp(-1), atol=1e-12, rtol=1e-12)
                upstream = torch.randn_like(expected)
                got = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
                want = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
                for a, b in zip(got, want):
                    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
                cases += 1
print(f"PASS {cases} blocked attention cases: outputs, logsumexp, Q/K/V gradients, partial tiles and causal masks.")
assert 64 * 64 * 4 == 16 * 1024
assert 8 * (8192 // 128) == 512
assert 4 * 16 + 4 * 5 // 2 == 74
assert 4 * 16000 + 4 * 5 // 2 == 64010
print("PASS FA3 register arithmetic, FA2 task count and scheduling pair counts.")

