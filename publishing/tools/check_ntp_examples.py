#!/usr/bin/env python3
"""CPU checks for the executable examples in the Russian NTP chapter.

Run from any directory with Python and torch installed. No model downloads,
GPU checks, training-throughput measurements, or site rendering are performed.
The loss function and worked example are loaded from the chapter itself.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md"


def close(actual, expected, *, atol=2e-6):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=2e-6,
        atol=atol,
    )


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
    assert len(blocks) == 3, "Review new/removed chapter code blocks before updating this check"
    namespace = {"__name__": "bookvar_ntp_chapter"}
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f"{CHAPTER}:python-block-{index}", "exec"), namespace)
    loss_fn = namespace["ntp_loss"]
    ids = namespace["input_ids"]
    logits = namespace["logits"]
    mask = namespace["mask"]
    short_mask = namespace["mask_without_eos"]

    close(namespace["loss"].detach(), 2 * math.log(2))
    close(namespace["loss"].detach().exp(), 4)
    close(namespace["loss_without_eos"].detach(), 1.5 * math.log(2))
    close(namespace["loss_without_eos"].detach().exp(), 2 ** 1.5)
    close((0.5 * 0.25 * 0.125) ** (-1 / 3), 4)
    close(6 * math.log(2) / (12 * math.log(2)), 0.5)  # 12 raw bytes: BPB.
    print("PASS: executed all three chapter code blocks; NLL, masks, PPL, BPB")

    # The actual chapter function agrees with explicitly pre-shifted targets.
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 4), ids[:, 1:].reshape(-1))
    close(loss_fn(logits, ids, mask).detach(), expected.detach())
    wrong_double_shift = F.cross_entropy(logits[:, :-2].reshape(-1, 4), ids[:, 2:].reshape(-1))
    assert abs(wrong_double_shift.item() - expected.item()) > 0.1
    for empty_mask in (torch.zeros_like(mask), torch.tensor([[True, False, False, False]])):
        try:
            loss_fn(logits, ids, empty_mask)
        except ValueError as error:
            assert "No target tokens" in str(error)
        else:
            raise AssertionError("An empty effective target set must fail explicitly")
    try:
        loss_fn(logits, ids, mask.to(torch.int64))
    except ValueError:
        pass
    else:
        raise AssertionError("Nonboolean mask must be rejected")
    print("PASS: one shift only; double-shift negative control; empty/type guards")

    uniform = torch.zeros_like(logits, requires_grad=True)
    close(loss_fn(uniform, ids, mask).detach(), math.log(4))
    close(loss_fn(logits + 1000, ids, mask).detach(), expected.detach(), atol=6e-5)
    stable = F.cross_entropy(torch.tensor([[1000.0, 1001.0, 1002.0]]), torch.tensor([2]))
    close(stable, math.log(1 + math.exp(-1) + math.exp(-2)))
    p = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float64)
    z = p.log().requires_grad_()
    F.cross_entropy(z, torch.tensor([1])).backward()
    close(z.grad, [[0.1, -0.3, 0.2]])

    z = logits.detach().clone().requires_grad_()
    loss_fn(z, ids, short_mask).backward()
    close(z.grad[:, 2:], torch.zeros_like(z.grad[:, 2:]))
    changed = logits.detach().clone()
    changed[:, 2:] = float("nan")
    close(loss_fn(changed, ids, short_mask), namespace["loss_without_eos"].detach())
    # EOS and PAD may share ID=3; only length-aware masking keeps the real EOS.
    padded_ids = torch.tensor([[0, 1, 3, 3]])
    real_targets = torch.tensor([[False, True, True, False]])
    id_mask = padded_ids != 3
    assert real_targets[:, 1:].sum() == 2 and id_mask[:, 1:].sum() == 1
    print("PASS: stable CE and gradients; excluded rows; EOS/PAD role distinction")

    # Deterministic one-head attention fixture: not a GPT/model benchmark.
    x = torch.randn(1, 1, 6, 4, dtype=torch.float64)
    changed_x = x.clone()
    changed_x[:, :, 3:] += 10
    causal = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    changed_causal = F.scaled_dot_product_attention(changed_x, changed_x, changed_x, is_causal=True)
    close(causal[:, :, :3], changed_causal[:, :, :3], atol=1e-12)
    noncausal = F.scaled_dot_product_attention(x, x, x, is_causal=False)
    changed_noncausal = F.scaled_dot_product_attention(changed_x, changed_x, changed_x, is_causal=False)
    assert not torch.allclose(noncausal[:, :, :3], changed_noncausal[:, :, :3])
    print("PASS: causal prefix invariance; unmasked-attention negative control")

    close((3 * 1 + 1 * 3) / 4, 1.5)
    close((1 + 3) / 2, 2)
    batch_ids = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
    batch_mask = torch.tensor([[False, True, True, True], [False, True, False, False]])
    base = torch.randn(2, 4, 4)
    direction = torch.randn_like(base)
    scalar = torch.tensor(0.3, requires_grad=True)
    all_logits = base + scalar * direction
    full_loss = loss_fn(all_logits, batch_ids, batch_mask)
    pieces = [loss_fn(all_logits[i:i+1], batch_ids[i:i+1], batch_mask[i:i+1]) for i in range(2)]
    weighted = (3 * pieces[0] + pieces[1]) / 4
    equal_weighted = sum(pieces) / 2
    full_grad = torch.autograd.grad(full_loss, scalar, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted, scalar, retain_graph=True)[0]
    wrong_grad = torch.autograd.grad(equal_weighted, scalar)[0]
    close(full_loss.detach(), weighted.detach())
    close(full_grad, weighted_grad)
    assert abs(full_grad.item() - wrong_grad.item()) > 1e-4
    print("PASS: token-weighted microbatch losses AND shared-parameter gradients")

    windows = namespace["scoring_windows"]
    for length in range(2, 20):
        for context in range(2, 10):
            for stride in range(1, context):
                scored, counts = [], []
                for begin, end, target_mask, count in windows(length, context, stride):
                    assert len(target_mask) == end - begin <= context
                    actual = [begin + j for j in range(1, end - begin) if target_mask[j]]
                    assert count == len(actual) > 0
                    scored.extend(actual)
                    counts.append(count)
                assert scored == list(range(1, length)) and sum(counts) == length - 1
    for args in ((1, 4, 2), (7, 1, 1), (7, 4, 0), (7, 4, 4), (7, 4, 5)):
        try:
            list(windows(*args))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid scoring-window arguments accepted: {args}")

    archive = ROOT / "05 Источники/Originals/ntp-2026-10-23"
    source = (archive / "hf-perplexity.md").read_text()
    quote = re.search(r"^> (With our sliding window approach,.*?)$", text, re.M).group(1)
    assert " ".join(quote.split()) in " ".join(source.split())
    source_blocks = re.findall(r"^```python\n(.*?)^```", source, re.M | re.S)
    source_tree = ast.parse(source_blocks[-1])
    counter = next(node for node in ast.walk(source_tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "num_loss_tokens" for t in node.targets))
    counter_code = compile(ast.Expression(counter.value), "pinned-hf-counter", "eval")
    assert ast.unparse(counter.value) == "num_valid_tokens - batch_size"
    counts, wrong_counts, losses = [], [], []
    for index, (begin, end, target_mask, count) in enumerate(windows(7, 4, 2)):
        # Synthetic constant probabilities per window, not GPT-2 evaluation.
        local_ids = torch.arange(begin, end).unsqueeze(0) % 7
        keep = torch.tensor([target_mask])
        targets = local_ids.clone().masked_fill(~keep, -100)
        p_correct = 2.0 ** -(index + 1)
        probs = torch.full((1, end - begin, 7), (1 - p_correct) / 6)
        for row in range(end - begin - 1):
            probs[0, row, local_ids[0, row + 1]] = p_correct
        local_loss = loss_fn(probs.log(), local_ids, keep)
        library_loss = F.cross_entropy(probs[:, :-1].log().reshape(-1, 7),
                                       targets[:, 1:].reshape(-1))
        close(local_loss, library_loss)
        close(local_loss, (index + 1) * math.log(2))
        counts.append((targets[:, 1:] != -100).sum().item())
        wrong_counts.append(eval(counter_code, {}, {
            "num_valid_tokens": (targets != -100).sum().item(),
            "batch_size": targets.shape[0],
        }))
        losses.append(local_loss.item())
        assert counts[-1] == count
    assert counts == [3, 2, 1] and wrong_counts == [3, 1, 0]
    correct_ppl = math.exp(sum(n * loss for n, loss in zip(counts, losses)) / sum(counts))
    wrong_ppl = math.exp(sum(n * loss for n, loss in zip(wrong_counts, losses)) / sum(wrong_counts))
    close(correct_ppl, 2 ** (10 / 6))
    close(wrong_ppl, 2 ** (5 / 4))
    print(f"PASS: actual window function; pinned HF counter regression {counts} vs {wrong_counts}; "
          f"PPL {correct_ppl:.4f} vs {wrong_ppl:.4f}")

    head_counts = [len(range(5 - k)) for k in (1, 2, 3)]
    assert head_counts == [4, 3, 2] and sum(head_counts) == 9
    close(sum((1, 2, 3)) / 3, 2)
    close(sum(n * mean for n, mean in zip(head_counts, (1, 2, 3))) / 9, 16 / 9)
    assert [(t, t + 2) for t in range(4)] == [(0, 2), (1, 3), (2, 4), (3, 5)]
    # A one-head synthetic MTP fixture exercises the depth-specific causal
    # boundary; it is not an implementation or execution of DeepSeek-V3.
    emb = torch.randn(1, 1, 6, 4, dtype=torch.float64)
    projected = torch.randn(4, 8, dtype=torch.float64)
    def mtp_fixture(tokens, causal):
        base = F.scaled_dot_product_attention(tokens, tokens, tokens, is_causal=True)
        a = F.rms_norm(base[:, :, :-1], (4,))
        b = F.rms_norm(tokens[:, :, 1:], (4,))
        fused = F.linear(torch.cat((a, b), dim=-1), projected)
        return F.scaled_dot_product_attention(fused, fused, fused, is_causal=causal)
    changed_emb = emb.clone()
    changed_emb[:, :, 3:] += torch.tensor([10., -7., 3., 5.])
    close(mtp_fixture(emb, True)[:, :, :2], mtp_fixture(changed_emb, True)[:, :, :2], atol=1e-12)
    assert not torch.allclose(mtp_fixture(emb, False)[:, :, :2], mtp_fixture(changed_emb, False)[:, :, :2])
    print("PASS: MTP valid-pair counts, 2 vs 16/9 weighting; shifted-input causal boundary")

    hashes = {
        "ppl-full.gif": "8e4421a0b180d9a5b5d8edee4705da19775ce52f80d297623cbbd29bdc406c9d",
        "ppl-chunked.gif": "3d8b260868c278a01e6b291cb0a939065fc5182ba01dbb08bbc1ba4849ce402a",
        "ppl-sliding.gif": "c85f2334f8f447a1dc314f697a27074f586a5f305b666472a3e7d9297297df15",
    }
    for name, digest in hashes.items():
        published = ROOT / "00 Учебник/Assets/Figures/curated/ntp-2026-10-23" / name
        assert published.read_bytes() == (archive / name).read_bytes()
        assert hashlib.sha256(published.read_bytes()).hexdigest() == digest
        assert name in text
    assert "license: cc-by-nc-sa-4.0" in (archive / "hf-images-README.md").read_text()
    assert (ROOT / "site/public/licenses/transformers-e7b03417-Apache-2.0.txt").read_bytes() == (archive / "transformers-LICENSE.txt").read_bytes()
    print("PASS: unchanged original GIF bytes, dataset license and complete Apache text license")

    # Very small bigram model; verifies the chapter loss in an optimizer loop.
    table = torch.nn.Embedding(4, 4)
    optimizer = torch.optim.Adam(table.parameters(), lr=0.1)
    initial = loss_fn(table(ids), ids, mask).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(table(ids), ids, mask)
        loss.backward()
        optimizer.step()
    final = loss_fn(table(ids), ids, mask).item()
    assert final < 0.01 and final < initial / 100
    print(f"PASS: one-sequence CPU overfit fixture, NLL {initial:.6f} -> {final:.6f}")
    print(f"All NTP checks passed on CPU with torch {torch.__version__}; no GPU/benchmark claims.")
    print(f"Chapter SHA-256: {hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
