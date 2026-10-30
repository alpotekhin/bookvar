#!/usr/bin/env python3
"""Verify chapter code, a separate functional forward/backward, masks and arithmetic."""
from pathlib import Path
import copy
import hashlib
import inspect
import itertools
import math
import re
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5 import modeling_t5
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / '05 Источники/Originals/t5-2026-10-22'
CHAPTER = ROOT / '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md'
assert transformers.__version__ == '4.57.6'
assert Path(inspect.getfile(modeling_t5)).read_bytes() == (ARC / 'modeling_t5.py').read_bytes()
assert hashlib.sha256((ARC / 'modeling_t5.py').read_bytes()).hexdigest() == 'ee1a043f5c03c684a2bbb0213ea3b3f8be272a03ab884bdca57c31e7315e7b7e'
blocks = re.findall(r'~~~python\n(.*?)\n~~~', CHAPTER.read_text(), re.S)
assert len(blocks) == 2
envs = []
for i, block in enumerate(blocks):
    env = {}
    exec(compile(block, str(CHAPTER) + ':' + str(i + 1), 'exec'), env)
    envs.append(env)
collapse = envs[0]['replace_runs']
e = envs[1]
model = e['model'].eval()
inputs, labels = e['inputs'], e['labels']
src_mask, dec_mask, dec = e['source_mask'], e['decoder_mask'], e['decoder_ids']
assert e['after_loss'] < e['out'].loss
assert int(labels.ne(-100).sum()) == 11
assert dec[0, 0] == 0 and dec_mask[0, 0] == 1

# Derive the actual published batch from both original sequences, not just
# arbitrary invertible masks. Each clean suffix closes the sentinel sequence.
for row, original, noise in [
        (0, [2, 3, 4, 5, 6, 7, 8], [False, True, True, False, False, True, False]),
        (1, [3, 9, 4], [False, True, False])]:
    source = collapse(original, noise) + [1]
    target = collapse(original, [not x for x in noise]) + [1]
    assert inputs[row, :len(source)].tolist() == source
    assert labels[row, labels[row].ne(-100)].tolist() == target
    sentinel_ids = [x for x in target if x >= 12]
    assert sentinel_ids == list(range(14, 14-len(sentinel_ids), -1))
    spans, current = {}, None
    for token in target[:-1]:
        if token >= 12:
            current = token
            spans[current] = []
        else:
            spans[current].append(token)
    recovered = []
    for token in source[:-1]:
        recovered.extend(spans[token] if token >= 12 else [token])
    assert recovered == original
print('PASS both published source/target pairs, closing sentinels and reconstruction')

# Invert every mask starting with clean text, including clean/noise endings.
for n in range(2, 9):
    original = list(range(2, n + 2))
    for tail in itertools.product([False, True], repeat=n - 1):
        noise = [False, *tail]
        if not any(noise):
            continue
        source = collapse(original, noise, 1000)
        target = collapse(original, [not v for v in noise], 1000)
        replacements, current = {}, None
        for token in target:
            if token >= 900:
                current = token
                replacements[current] = []
            else:
                replacements[current].append(token)
        recovered = []
        for token in source:
            recovered.extend(replacements[token] if token >= 900 else [token])
        assert recovered == original

tok = T5Tokenizer(vocab_file=str(ARC / 'sentencepiece.model'), extra_ids=100, legacy=True)
assert tok.vocab_size == 32000 and len(tok) == 32100
assert tok('not acceptable').input_ids == [59, 9961, 1]
assert [tok.convert_tokens_to_ids('<extra_id_%s>' % i) for i in (0, 1, 99)] == [32099, 32098, 32000]
assert tok('Thank you <extra_id_0> me to your party <extra_id_1> week.').input_ids == [1562, 25, 32099, 140, 12, 39, 1088, 32098, 471, 5, 1]
assert tok('<extra_id_0> for inviting <extra_id_1> last <extra_id_2>').input_ids == [32099, 21, 14256, 32098, 336, 32097, 1]
print('PASS actual tokenizer IDs and invertible sentinel examples')

def bucket(r, bidirectional):
    count = 16 if bidirectional else 32
    direction = 16 if bidirectional and r > 0 else 0
    distance = abs(r) if bidirectional else max(-r, 0)
    exact = count // 2
    if distance < exact:
        return direction + distance
    large = exact + int(math.log(distance / exact) / math.log(128 / exact) * (count - exact))
    return direction + min(large, count - 1)

offsets = torch.arange(-1000, 1001)
for bidi in (False, True):
    expected = torch.tensor([bucket(int(r), bidi) for r in offsets])
    actual = modeling_t5.T5Attention._relative_position_bucket(offsets, bidi)
    torch.testing.assert_close(actual, expected)
assert [bucket(r, True) for r in (0, 1, 2, 8, 16, 64, 128)] == [0, 17, 18, 24, 26, 30, 31]

def norm(x, module):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.variance_epsilon) * module.weight

def attn(x, module, allowed, memory=None, bias=None):
    batch, length, _ = x.shape
    memory = x if memory is None else memory
    heads, dk = module.n_heads, module.key_value_proj_dim
    def split(z):
        return z.reshape(batch, -1, heads, dk).transpose(1, 2)
    q = split(F.linear(x, module.q.weight))
    k = split(F.linear(memory, module.k.weight))
    v = split(F.linear(memory, module.v.weight))
    if bias is None:
        if module.has_relative_attention_bias:
            ids = torch.tensor([[bucket(j-i, not module.is_decoder) for j in range(k.shape[2])] for i in range(length)])
            bias = F.embedding(ids, module.relative_attention_bias.weight).permute(2, 0, 1)[None]
        else:
            bias = x.new_zeros(1, heads, length, k.shape[2])
    scores = q @ k.transpose(-1, -2) + bias
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = scores.float().softmax(-1).to(scores.dtype)
    result = (weights @ v).transpose(1, 2).reshape(batch, length, heads * dk)
    return F.linear(result, module.o.weight), bias

def ff(x, layer):
    a = norm(x, layer.layer_norm)
    return x + F.linear(F.relu(F.linear(a, layer.DenseReluDense.wi.weight)), layer.DenseReluDense.wo.weight)

def functional_forward():
    memory = F.embedding(inputs, model.shared.weight)
    bias = None
    enc_allowed = src_mask.bool()[:, None, None, :]
    for block in model.encoder.block:
        layer = block.layer[0]
        added, bias = attn(norm(memory, layer.layer_norm), layer.SelfAttention, enc_allowed, bias=bias)
        memory = ff(memory + added, block.layer[1])
    memory = norm(memory, model.encoder.final_layer_norm)
    hidden = F.embedding(dec, model.shared.weight)
    causal = torch.ones(dec.shape[1], dec.shape[1], dtype=torch.bool).tril()
    dec_allowed = causal[None, None] & dec_mask.bool()[:, None, None, :]
    bias = None
    for block in model.decoder.block:
        layer = block.layer[0]
        added, bias = attn(norm(hidden, layer.layer_norm), layer.SelfAttention, dec_allowed, bias=bias)
        hidden = hidden + added
        layer = block.layer[1]
        added, _ = attn(norm(hidden, layer.layer_norm), layer.EncDecAttention, enc_allowed, memory=memory)
        hidden = ff(hidden + added, block.layer[2])
    hidden = norm(hidden, model.decoder.final_layer_norm)
    logits = F.linear(hidden * model.config.d_model**-0.5, model.shared.weight)
    return memory, logits

kw = dict(input_ids=inputs, attention_mask=src_mask, decoder_input_ids=dec,
          decoder_attention_mask=dec_mask, labels=labels, use_cache=False)
model.zero_grad()
reference = model(**kw, output_attentions=True)
reference.loss.backward()
ref_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
model.zero_grad()
memory, logits = functional_forward()
torch.testing.assert_close(memory, reference.encoder_last_hidden_state, atol=2e-6, rtol=2e-5)
torch.testing.assert_close(logits, reference.logits, atol=2e-6, rtol=2e-5)
loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
loss.backward()
max_grad_error = 0.
for name, parameter in model.named_parameters():
    torch.testing.assert_close(parameter.grad, ref_grads[name], atol=3e-6, rtol=3e-4)
    max_grad_error = max(max_grad_error, float((parameter.grad - ref_grads[name]).abs().max()))
assert reference.encoder_attentions[0].shape == (2, 2, 7, 7)
assert reference.decoder_attentions[0].shape == (2, 2, 7, 7)
assert reference.cross_attentions[0].shape == (2, 2, 7, 7)
assert reference.decoder_attentions[0].triu(1).abs().max() == 0
assert reference.cross_attentions[0][1, :, :, 4:].abs().max() == 0
print('PASS independent two-stack outputs and all gradients; max error', max_grad_error)

with torch.no_grad():
    changed = dec.clone()
    changed[:, 3:] = 10
    out = model(**{**kw, 'decoder_input_ids': changed})
    torch.testing.assert_close(out.logits[:, :3], reference.logits[:, :3])
    changed = inputs.clone()
    changed[1, 4:] = 10
    out = model(**{**kw, 'input_ids': changed})
    torch.testing.assert_close(out.logits, reference.logits)
    # Full pass vs sequential cached decoding for one unpadded example.
    encoded = BaseModelOutput(last_hidden_state=reference.encoder_last_hidden_state[:1])
    full = model(encoder_outputs=encoded, attention_mask=src_mask[:1],
                 decoder_input_ids=dec[:1], use_cache=False).logits
    counts = {}
    handles = []
    for layer, block in enumerate(model.decoder.block):
        for kind, attention in [('self', block.layer[0].SelfAttention),
                                ('cross', block.layer[1].EncDecAttention)]:
            for projection in ('k', 'v'):
                key = (layer, kind, projection)
                counts[key] = 0
                def increment(_module, _args, _output, key=key):
                    counts[key] += 1
                handles.append(getattr(attention, projection).register_forward_hook(increment))
    past, parts = None, []
    for pos in range(dec.shape[1]):
        one = model(encoder_outputs=encoded, attention_mask=src_mask[:1],
                    decoder_input_ids=dec[:1, pos:pos+1], past_key_values=past, use_cache=True)
        past = one.past_key_values
        parts.append(one.logits)
        for layer in range(model.config.num_decoder_layers):
            self_layer = past.self_attention_cache.layers[layer]
            cross_layer = past.cross_attention_cache.layers[layer]
            assert self_layer.keys.shape == (1, 2, pos+1, 4)
            assert cross_layer.keys.shape == (1, 2, inputs.shape[1], 4)
    for handle in handles:
        handle.remove()
    for (layer, kind, projection), count in counts.items():
        assert count == (1 if kind == 'cross' else dec.shape[1])
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-6, rtol=2e-5)
print('PASS causal visibility, source padding and sequential self/cross-KV')

with torch.no_grad():
    a = torch.tensor([[2, 3, 4, 1]])
    b = torch.tensor([[2, 3, 9, 1]])
    mask = torch.ones_like(a)
    memory_a = model.encoder(input_ids=a, attention_mask=mask)
    memory_b = model.encoder(input_ids=b, attention_mask=mask)
    prefix_error = (memory_a.last_hidden_state[:, :2] - memory_b.last_hidden_state[:, :2]).abs().max().item()
    assert prefix_error > 1e-4  # Same raw prefix, different completed encoder memory.
    start = torch.tensor([[0]])
    next_token = torch.tensor([[14]])
    def start_cache(memory):
        return model(encoder_outputs=memory, attention_mask=mask,
                     decoder_input_ids=start, use_cache=True).past_key_values
    cache_a, cache_b = start_cache(memory_a), start_cache(memory_b)
    def continue_with(memory, cache):
        return model(encoder_outputs=memory, attention_mask=mask,
                     decoder_input_ids=next_token,
                     past_key_values=copy.deepcopy(cache), use_cache=True).logits
    old = continue_with(memory_a, cache_a)
    stale = continue_with(memory_b, cache_a)
    fresh = continue_with(memory_b, cache_b)
    # Negative control: replacing encoder_outputs alone leaves old cross-KV.
    torch.testing.assert_close(stale, old, atol=0, rtol=0)
    stale_error = (stale-fresh).abs().max().item()
    assert stale_error > 1e-4
    higher_self_error = (cache_a.self_attention_cache.layers[1].keys -
                         cache_b.self_attention_cache.layers[1].keys).abs().max().item()
    assert higher_self_error > 1e-4
print('PASS changed-source cache negative control; encoder-prefix/stale-logit/higher-self deltas',
      prefix_error, stale_error, higher_self_error)

base = T5Config(vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
                num_heads=12, num_layers=12, num_decoder_layers=12)
with torch.device('meta'):
    count = sum(p.numel() for p in T5ForConditionalGeneration(base).parameters())
d, a, f, h, v, layers = 768, 768, 3072, 12, 32128, 12
expected = v*d + layers*(4*d*a+2*d*f+2*d) + layers*(8*d*a+2*d*f+3*d) + 2*d+64*h
assert count == expected == 222_903_552
assert 2*12*128*768*2 == 4.5*2**20
assert 2*12*512*768*2 == 18*2**20
assert 512*768*2 == .75*2**20
assert 568-85+28+1 == 512 and 85+28+1 == 114
assert round(568*.15) == 85 and round(85/3) == 28
assert 2**19*128*512 == 2**35 and 1_000_000*2**11*512 == 1_048_576_000_000
assert 2**35//2**23 == 4096
assert math.isclose(-math.log(.4*.5*.8)/3, .6108604879161033)
sizes = np.array([100_000_000., 1_000_000., 10_000.])
capped = np.minimum(sizes, 1_000_000)
np.testing.assert_allclose(capped/capped.sum(), [100/201, 100/201, 1/201])
heated = np.sqrt(capped)
np.testing.assert_allclose(heated/heated.sum(), [10/21, 10/21, 1/21])
assert 2*768*32 == 49152 and 2*768*512 == 786432
print('PASS parameters, context lengths, resource estimates, loss, mixtures and adapters')
print('No released-weight quality or TPU/production performance claim.')
print('CHAPTER_SHA256', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
