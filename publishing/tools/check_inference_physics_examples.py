#!/usr/bin/env python3
"""Check the actual 55a Markdown examples and independent arithmetic ledgers.

Pure Python accounting tests, not hardware traffic measurement or benchmark.
"""
from pathlib import Path
import hashlib
import math
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md"


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    text = CHAPTER.read_text()
    ns = {}
    for name in ("inference-mlp-traffic", "inference-llama-budget"):
        match = re.search(
            rf"<!-- example: {name} -->\n\x60\x60\x60python\n(.*?)\n\x60\x60\x60",
            text, re.S,
        )
        assert match, name
        exec(compile(match[1], f"{CHAPTER.name}:{name}", "exec"), ns)

    # Independently enumerate transfers, not just re-evaluate a copied formula.
    for rows in (1, 64, 1024):
        d, f, b = 5120, 13824, 2
        x, w, hidden, y = rows * d, d * f, rows * f, rows * d
        separate_ops = (
            (x + w + hidden),       # up
            (x + w + hidden),       # gate
            (hidden + hidden + hidden),  # read U/G, write H
            (hidden + w + y),       # down
        )
        fused_ops = (x + 2 * w + hidden, hidden + w + y)
        flops, separate, fused = ns["mlp_cost"](rows, d, f)
        assert flops == sum((2 * rows * d * f,) * 3)
        assert separate == b * sum(separate_ops)
        assert fused == b * sum(fused_ops)
        assert separate > fused > 0
        print("MLP", rows, "FLOPs", flops, "bytes", separate, fused,
              "intensities", flops / separate, flops / fused)

    # Allocation and compulsory reads differ: embedding lookup is not full scan.
    for hkv in (40, 8):
        d, f, layers, vocab, hq, dh, context = 5120, 13824, 40, 32000, 40, 128, 1024
        matrices_per_layer = (d * hq * dh, d * hkv * dh, d * hkv * dh,
                              hq * dh * d, d * f, d * f, f * d)
        body = layers * sum(matrices_per_layer)
        emb = vocab * d
        kv_one = layers * context * hkv * dh * 2 * 2
        for batch in (1, 64, 256):
            allocated, read, seconds, throughput = ns["decode_budget"](batch, hkv, context)
            assert allocated == 2 * (body + emb + emb) + batch * kv_one
            assert read == 2 * (body + emb) + batch * kv_one + 2 * batch * d
            assert allocated - read == 2 * emb - 2 * batch * d
            close(seconds, read / 3.35e12)
            close(throughput * seconds, batch)
            print("DECODE", hkv, batch, "GB allocated/read", allocated / 1e9,
                  read / 1e9, "ms", seconds * 1000, "tokens/s estimate", throughput)
    assert ns["decode_budget"](256)[0] > 80e9
    assert 79e9 < ns["decode_budget"](64)[0] < 80e9

    # MHA/GQA attention core: one read of Q/K/V and one output write.
    for batch, q_positions, kv_positions in ((1, 1, 1024), (64, 1, 1024), (1, 1024, 1024)):
        for hkv in (40, 8):
            d, hq, dh, b = 5120, 40, 128, 2
            q = batch * q_positions * hq * dh
            kv = batch * kv_positions * hkv * dh
            output = q
            read_write = b * (q + kv + kv + output)
            flops = 4 * batch * kv_positions * q_positions * d
            formula = 2 * q_positions * kv_positions * d / (
                b * (q_positions * d + kv_positions * hkv * dh)
            )
            close(flops / read_write, formula)
            if hkv == hq and q_positions == 1:
                assert flops / read_write < 1
            if hkv == 8 and q_positions == 1:
                assert 4.9 < flops / read_write < 5
    s, d = 1024, 5120
    causal_pairs = sum(range(1, s + 1))
    assert 4 * d * causal_pairs == 2 * s * (s + 1) * d
    close(989e12 / 3.35e12, 295.2238805970149)
    times = (0.30, 0.34, 0.38, 0.78)
    intervals = [b - a for a, b in zip(times, times[1:])]
    for actual, expected in zip(intervals, (0.04, 0.04, 0.40)):
        close(actual, expected)
    tpot = sum(intervals) / len(intervals)
    close(tpot, 0.16)
    close(times[0] + (len(times) - 1) * tpot, times[-1])
    print("PASS: 2 actual Markdown blocks; MLP/embedding/KV/attention/roofline/latency arithmetic.")
    print("NOT TESTED: actual GPU HBM transactions, fused kernel, throughput or serving SLOs.")
    print("chapter_sha256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
