"""Neural correctness check; no pretrained weights, tokenizer download or timing.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_edls_kv_reference.py
"""
from __future__ import annotations

import importlib.util
import itertools
import contextlib
import io
from pathlib import Path
import random
import re
import sys
from types import ModuleType

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / '06 Практика/Code/edls_kv_reference.py'
spec = importlib.util.spec_from_file_location('edls_kv_reference', MODULE)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
Request, Engine = reference.Request, reference.KVGreedyEngine
torch.set_num_threads(1)
torch.manual_seed(8)
config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8,
                     max_position_embeddings=128, attention_dropout=0.,
                     bos_token_id=1, eos_token_id=2, pad_token_id=0,
                     use_sliding_window=False)
config._attn_implementation = 'eager'
model = Qwen3ForCausalLM(config).eval()
engine = Engine(model)
oracle = reference.uncached_greedy
cases = 0
max_cache_error = 0.

def check_requests(requests):
    global cases, max_cache_error
    for r in requests:
        assert r.generated_tokens == oracle(model, r.prompt_ids, len(r.generated_tokens))
        if r.past_key_values is not None:
            expected_len = len(r.prompt_ids) + len(r.generated_tokens) - 1
            assert r.past_key_values.get_seq_length() == expected_len
            prefix = (*r.prompt_ids, *r.generated_tokens[:-1])
            with torch.no_grad():
                expected_cache = model(torch.tensor([prefix]), use_cache=True).past_key_values
            for (key, value), (expected_key, expected_value) in zip(r.past_key_values, expected_cache):
                assert key.shape == value.shape == (1, 2, expected_len, 8)
                for actual, expected in [(key, expected_key), (value, expected_value)]:
                    max_cache_error = max(max_cache_error, float((actual - expected).abs().max()))
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        cases += 1

def drain(requests, reorder=False):
    while any(not r.is_finished for r in requests):
        engine.decode(requests[::-1] if reorder else requests)
        check_requests(requests)

prompts = [(1, 4, 5), (1, 7, 8, 9, 10), (1, 12)]
limits = [1, 5, 3]
for order in itertools.permutations(range(3)):
    requests = [Request(str(i), prompts[i], limits[i]) for i in order]
    first = engine.prefill(requests)
    assert first.request_ids == [r.request_id for r in requests]
    assert first.finished == [r.max_new_tokens == 1 for r in requests]
    check_requests(requests)
    drain(requests, reorder=True)

# Admission after another request is already decoding; lengths are unequal.
a = Request('A', (1, 17, 18), 6)
engine.prefill([a])
engine.decode([a])
b = Request('B', (1, 23), 3)
engine.prefill([b])
drain([b, a], reorder=True)
assert a.generated_tokens == oracle(model, a.prompt_ids, 6)
assert b.generated_tokens == oracle(model, b.prompt_ids, 3)

# Different arriving batches, permutations, context lengths and stop lengths.
rng = random.Random(26)
for batch_size in [1, 2, 4]:
    for _ in range(8):
        requests = [Request(str(i), tuple(rng.randrange(1, 64) for _ in range(rng.randrange(1, 10))),
                            rng.randrange(1, 7)) for i in range(batch_size)]
        engine.prefill(requests)
        drain(requests, reorder=True)

# Record actual forwards; inactive requests may not launch another model pass.
calls = []
handle = model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
zero = Request('zero', (1, 3), 0)
assert engine.prefill([zero]).request_ids == [] and zero.stop_reason == 'length'
assert calls == []
one = Request('one', (1, 3), 1)
engine.prefill([one])
assert len(calls) == 1 and len(one.generated_tokens) == 1
before = list(one.generated_tokens)
for _ in range(3):
    assert engine.decode([one, zero]).request_ids == []
assert len(calls) == 1 and one.generated_tokens == before

# EOS at prefill, and EOS after the first decode.
expected = oracle(model, prompts[1], 2)
assert expected[0] != expected[1]
eos_first = Request('eos-first', prompts[1], 10, eos_token_id=expected[0])
engine.prefill([eos_first])
assert eos_first.stop_reason == 'eos' and len(eos_first.generated_tokens) == 1
eos_second = Request('eos-second', prompts[1], 10, eos_token_id=expected[1])
engine.prefill([eos_second])
engine.decode([eos_second])
assert eos_second.stop_reason == 'eos' and eos_second.generated_tokens == expected

cancelled = Request('cancelled', prompts[0], 8)
survivor = Request('survivor', prompts[2], 4)
engine.prefill([cancelled, survivor])
cancelled.cancel()
saved = list(cancelled.generated_tokens)
engine.decode([cancelled, survivor])
assert cancelled.stop_reason == 'cancelled' and cancelled.generated_tokens == saved
drain([survivor])

# Context limit means no processing beyond the configured valid positions.
limited = Engine(model, max_context=5)
at_limit = Request('limit', (1, 3, 4, 5, 6), 9)
limited.prefill([at_limit])
assert at_limit.stop_reason == 'context' and len(at_limit.generated_tokens) == 1
assert limited.decode([at_limit]).request_ids == []

# Finishing one row cannot retain aliases to another row's compact cache.
left, right = Request('left', prompts[0], 1), Request('right', prompts[2], 1)
engine.prefill([left, right])
right_copy = right.past_key_values[0][0].clone()
assert left.past_key_values[0][0].untyped_storage().data_ptr() != right.past_key_values[0][0].untyped_storage().data_ptr()
left.past_key_values[0][0].fill_(123)
torch.testing.assert_close(right.past_key_values[0][0], right_copy)
handle.remove()

def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError('invalid input accepted')

rejects(Engine, model, max_context=0)
rejects(Engine, model, max_context=129)
rejects(Engine, model, pad_token_id=64)
rejects(engine.prefill, [Request('empty', (), 1)])
rejects(engine.prefill, [Request('bad-token', (64,), 1)])
rejects(engine.prefill, [Request('negative', (1,), -1)])
rejects(engine.prefill, [Request('fraction', (1,), .5)])
rejects(engine.prefill, [Request('eos', (1,), 2, eos_token_id=64)])
rejects(engine.prefill, [Request('dup', (1,), 1), Request('dup', (1,), 1)])
rejects(engine.decode, [Request('not-prefilled', (1,), 1)])
active = Request('already', (1,), 3)
engine.prefill([active])
rejects(engine.prefill, [active])
assert engine.prefill([]).request_ids == engine.decode([]).request_ids == []

# Execute the actual published module and neural smoke fences in a clean module.
chapter = ROOT / '06 Практика/14 Собрать минимальный inference engine.md'
text = chapter.read_text()
solution = text.split('<!-- BEGIN SOLUTION: EDLS KV module -->')[1].split('<!-- END SOLUTION: EDLS KV module -->')[0].strip()
code = re.fullmatch(r'```python\n(.*)\n```', solution, re.S).group(1)
assert code == MODULE.read_text().rstrip()
smoke = re.search(r'<!-- example: neural-kv-smoke -->\n```python\n(.*?)\n```', text, re.S).group(1)
published = ModuleType('bookvar_published_edls_kv')
sys.modules[published.__name__] = published
with contextlib.redirect_stdout(io.StringIO()) as captured:
    exec(compile(code, 'practice14:module-fence', 'exec'), published.__dict__)
    exec(compile(smoke, 'practice14:neural-smoke-fence', 'exec'), published.__dict__)
assert captured.getvalue().splitlines() == [
    'A [50] 3', 'B [32, 27, 34, 28, 63] 9', 'C [63, 59, 24] 4']
print('PASS neural Qwen3 prefill/decode vs uncached greedy;', cases,
      'prefix/cache cases; EOS, 0/1-token budgets, removal/reorder, admission, '
      'cancellation, context limit and independent KV storage')
print('parameters=', sum(p.numel() for p in model.parameters()))
print('torch=', torch.__version__, 'transformers=', transformers.__version__)
print('max_cache_abs_error=', max_cache_error)
print('actual published module and smoke fences: PASS')
print('CPU random weights only; no language-quality, GPU or speedup claim')
