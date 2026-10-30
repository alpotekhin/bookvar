#!/usr/bin/env python3
"""Execute chapter 55 examples on CPU; no serving or GPU performance claims."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import re
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md'


def main():
    ns = {}
    text = CHAPTER.read_text()
    assert importlib.metadata.version('transformers') == '4.57.1'
    blocks = re.findall(r'<!-- example: ([a-z-]+) -->\n```python\n(.*?)\n```', text, re.S)
    assert [name for name, _ in blocks] == [
        'cached-attention-equivalence', 'cached-llama-equivalence',
        'cached-greedy-loop', 'cached-left-padding', 'paged-slot-address',
    ]
    assert len(re.findall(r'^```python$', text, re.M)) == len(blocks)
    for name, code in blocks:
        exec(compile(code, f'{CHAPTER.name}:{name}', 'exec'), ns)
    x = ns['x']
    step = ns['attention_chunk']
    ref, _ = step(x)
    for chunks in ((6,), (1, 1, 1, 1, 1, 1), (2, 4), (3, 2, 1), (5, 1)):
        past, parts, offset = None, [], 0
        for size in chunks:
            out, past = step(x[offset:offset+size], past)
            parts.append(out)
            offset += size
            # Every cached prefix is compared, not only the final concatenation.
            expected, cache = step(x[:offset])
            torch.testing.assert_close(out, expected[-size:], rtol=1e-12, atol=1e-12)
            for actual, wanted in zip(past, cache):
                torch.testing.assert_close(actual, wanted, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.cat(parts), ref, rtol=1e-12, atol=1e-12)

    # Independent reference through SDPA; verify the documented non-square trap.
    q, k, v = (x @ ns[name] for name in ('wq', 'wk', 'wv'))
    q4, k4, v4 = (t[None, None] for t in (q, k, v))
    torch.testing.assert_close(F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0, 0], ref)
    last = F.scaled_dot_product_attention(q4[:, :, -1:], k4, v4)
    torch.testing.assert_close(last[0, 0, 0], ref[-1])
    incorrect = F.scaled_dot_product_attention(q4[:, :, -1:], k4, v4, is_causal=True)
    torch.testing.assert_close(incorrect[0, 0, 0], v[0])
    assert not torch.allclose(incorrect, last)

    # A genuine two-layer decoder, not only the isolated attention equation.
    from transformers import DynamicCache
    model, ids = ns['lm'], ns['token_ids']
    reference = ns['full_output']
    maximum_error = 0.0
    with torch.inference_mode():
        for chunks in ((6,), (1, 1, 1, 1, 1, 1), (2, 4), (3, 2, 1), (5, 1)):
            cache, offset = DynamicCache(config=model.config), 0
            for size in chunks:
                end = offset + size
                output = model(ids[:, offset:end],
                    attention_mask=torch.ones((1, end), dtype=torch.long),
                    cache_position=torch.arange(offset, end),
                    past_key_values=cache, use_cache=True)
                expected = reference.logits[:, offset:end]
                maximum_error = max(maximum_error, (output.logits-expected).abs().max().item())
                torch.testing.assert_close(output.logits, expected, rtol=2e-6, atol=1e-8)
                for actual, complete in zip(cache.layers, reference.past_key_values.layers):
                    for name in ('keys', 'values'):
                        torch.testing.assert_close(getattr(actual, name), getattr(complete, name)[:, :, :end])
                offset = end
        greedy = model.generate(ids[:, :3], attention_mask=torch.ones_like(ids[:, :3]),
            max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
        assert torch.equal(greedy, ns['sequence'])

        bad_rope_cache = DynamicCache(config=model.config)
        model(ids[:, :3], past_key_values=bad_rope_cache, use_cache=True)
        bad_rope = model(ids[:, 3:], past_key_values=bad_rope_cache, use_cache=True,
            cache_position=torch.arange(3, 6), position_ids=torch.arange(3)[None, :]).logits
        rope_error = (bad_rope-reference.logits[:, 3:]).abs().max().item()
        assert rope_error > 1e-8

        duplicated_cache = DynamicCache(config=model.config)
        model(ids[:, :3], past_key_values=duplicated_cache, use_cache=True)
        duplicated = model(ids, past_key_values=duplicated_cache, use_cache=True)
        assert duplicated_cache.get_seq_length() == 9
        assert (duplicated.logits-reference.logits).abs().max().item() > 1e-8

        # Wrong next position after a correctly zero-based, left-padded prefill.
        bad_padding_cache = DynamicCache(config=model.config)
        model(ns['padded_ids'], attention_mask=ns['padding_mask'],
            position_ids=ns['positions'], cache_position=torch.arange(5),
            past_key_values=bad_padding_cache, use_cache=True)
        bad_padding = model(ns['next_ids'], attention_mask=ns['next_mask'],
            cache_position=torch.tensor([5]), past_key_values=bad_padding_cache,
            use_cache=True).logits[:, -1, :]
        padding_error = (bad_padding[0]-ns['padded_next'][0]).abs().max().item()
        assert padding_error > 1e-8
        torch.testing.assert_close(bad_padding[1], ns['padded_next'][1])

    stored_bytes = sum(t.numel()*t.element_size() for layer in ns['layer_cache'].layers
                       for t in (layer.keys, layer.values))
    assert model.dtype == torch.float32
    assert stored_bytes == 2*2*1*6*2*4*4 == 768
    assert len(ns['layer_cache'].layers) == 2
    assert torch.isfinite(ns['padded_next']).all()
    for layer in ns['padding_cache'].layers:
        assert torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()
    # Numerical boundary discovered while testing the FP64/eager variant:
    # a finite FP64 all-masked row overflows to -inf in an FP32 softmax.
    finite64 = torch.full((2,), torch.finfo(torch.float64).min, dtype=torch.float64)
    assert torch.isfinite(finite64).all()
    assert torch.isnan(torch.softmax(finite64, -1, dtype=torch.float32)).all()

    # Paged reads and blockwise stable softmax; unused slots are adversarial.
    generator = torch.Generator().manual_seed(41)
    keys = torch.randn(9, 4, dtype=torch.float64, generator=generator)
    values = torch.randn(9, 4, dtype=torch.float64, generator=generator)
    query = torch.randn(4, dtype=torch.float64, generator=generator)
    dense_scores = keys @ query / 2
    dense = dense_scores.softmax(0) @ values
    pool_k = torch.full((9*4, 4), 1e6, dtype=torch.float64)
    pool_v = torch.full_like(pool_k, 1e6)
    for t in range(9):
        slot = ns['physical_slot'](t)
        pool_k[slot], pool_v[slot] = keys[t], values[t]
    triples = []
    for logical, physical in enumerate(ns['block_table']):
        valid = min(4, 9 - logical * 4)
        kb = pool_k[physical*4:physical*4+valid]
        vb = pool_v[physical*4:physical*4+valid]
        scores = kb @ query / 2
        maximum = scores.max()
        exp = (scores - maximum).exp()
        triples.append((maximum, exp.sum(), exp @ vb))
    m = max(t[0] for t in triples)
    combined = sum((mb-m).exp()*u for mb, z, u in triples) / sum((mb-m).exp()*z for mb, z, u in triples)
    torch.testing.assert_close(combined, dense, rtol=1e-12, atol=1e-12)
    scores = torch.tensor([0, 0, math.log(3), math.log(3)], dtype=torch.float64)
    torch.testing.assert_close(scores.softmax(0)[:2].sum(), torch.tensor(0.25, dtype=torch.float64))
    correct_mean = scores.softmax(0) @ torch.tensor([1, 1, 3, 3], dtype=torch.float64)
    torch.testing.assert_close(correct_mean, torch.tensor(2.5, dtype=torch.float64))

    # Independent accounting: enumerate valid/padded slots for each request.
    slot_bytes = 2 * 32 * 8 * 128 * 2
    assert slot_bytes == 128 * 1024
    assert 8192 * slot_bytes == 1024**3
    for size, expected_slots, expected_empty, table_bytes in ((16, 112, 24, 28), (32, 128, 40, 16)):
        allocated = sum(len(range(0, length, size)) * size for length in (17, 31, 40))
        assert allocated == expected_slots
        assert allocated - sum((17, 31, 40)) == expected_empty
        assert allocated // size * 4 == table_bytes
        for length in range(1, 130):
            padding = math.ceil(length / size) * size - length
            assert 0 <= padding < size
        assert size * slot_bytes == (2 if size == 16 else 4) * 1024**2

    # COW reference count invariants for two branches, one incomplete tail.
    pool = {7: [0, 1, 2, 3], 1: [4, 5, 6, None]}
    refs = {7: 2, 1: 2}
    a, b = [7, 1], [7, 1]
    pool[3] = pool[a[-1]].copy()
    refs[a[-1]] -= 1
    a[-1], refs[3] = 3, 1
    pool[a[-1]][3], pool[b[-1]][3] = 7, 8
    assert pool[1] == [4, 5, 6, 8] and pool[3] == [4, 5, 6, 7]
    assert refs == {7: 2, 1: 1, 3: 1}
    for block in a:
        refs[block] -= 1
    assert refs[7] == refs[1] == 1 and refs[3] == 0
    print(json.dumps({'actual_python_fences': len(blocks),
        'random_model_parameters': sum(p.numel() for p in model.parameters()),
        'full_model_max_logit_error': maximum_error,
        'wrong_rope_position_max_error': rope_error,
        'wrong_padding_position_max_error': padding_error,
        'kv_bytes_fp32': stored_bytes, 'manual_greedy_matches_generate': True}, indent=2))
    print('PASS: 5 actual Markdown blocks; isolated and genuine two-layer cached decoder; GQA/RoPE/padding; SDPA mask negative control; paged reads/global softmax; byte/rounding and COW arithmetic.')
    print('torch', torch.__version__, 'device CPU')
    print('NOT TESTED: pretrained-model quality, GPU kernels, real allocator concurrency, serving performance.')
    print('chapter_sha256', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
