---
title: "Seminar: Prefill vs Decode, KV-cache, Inference Engines"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week08_inference_software/seminar.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/seminar.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



Install all libs

```python
pip install transformers==4.53.0
```

```python
pip install flash-attn --no-build-isolation
```

### Check GPUs

```python
import torch
import torch.nn as nn
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from collections import defaultdict

print(f"Number of available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  Memory: {total_mem:.2f} GB")
    available, total = torch.cuda.mem_get_info(i)
    print(f"  Free memory: {available / 1e9:.2f} GB")
```

## Download base model - Qwen3-4B

```python
MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=device,
    attn_implementation="eager",
)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")
```

```python
model
```

### What is KV cache?

```python
prompt = "Pick up the phone, baby\nI know you want to call me today"
tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

output = model(tokens) # What happened here?
```

```python
output
```

```python
print(f"number of tokens: {tokens.shape}")
print(f"logits shape: {output.logits.shape}")
```

```python
output.past_key_values
```

!!! important: this class has changed in transformers 5+

```python
# Let's see how key and value cache are stored
print(f"number of layers from past_key_values: {len(output.past_key_values)}")
print(f"number of layers from key_cache: {len(output.past_key_values.key_cache)}")
print(f"number of layers from value_cache: {len(output.past_key_values.value_cache)}")

print("\n")
# Let's see shapes
idx = 17
print(f"shape of key cache for layer {idx}: {output.past_key_values.key_cache[idx].shape}")
print(f"shape of value cache for layer {idx}: {output.past_key_values.value_cache[idx].shape}")

print("\n")
# Alternative way to get key and value cache
print(f"shape of key cache for layer {idx}: {output.past_key_values[idx][0].shape}")
print(f"shape of value cache for layer {idx}: {output.past_key_values[idx][1].shape}")
```

```python
### If you want you can turn off KV cache
output = model(tokens, use_cache=False)
output.past_key_values is None
```

If you already have some KV cache, you can pass it into model generation to avoid dublicate computations

```python
first_part = "Pick up the phone, baby\n"
second_part = "I know you want to call me today"

tokens_first_part = tokenizer.encode(first_part, return_tensors="pt").to(model.device)
tokens_second_part = tokenizer.encode(second_part, return_tensors="pt").to(model.device)
print(f"length of first part: {len(tokens_first_part[0])}")
print(f"length of second part: {len(tokens_second_part[0])}")

assert torch.allclose(torch.cat([tokens_first_part, tokens_second_part], dim=1), tokens)
```

```python
# We can reuse KV cache from first part
output_first_part = model(tokens_first_part, use_cache=True)
kv_cache = output_first_part.past_key_values

seq_len_first = tokens_first_part.shape[1]   # 9
seq_len_second = tokens_second_part.shape[1]  # 11

output_second_part = model(tokens_second_part, use_cache=True, past_key_values=kv_cache)

assert torch.allclose(output_second_part.logits, output.logits[:, -seq_len_second:, :])
```

```python
output_second_part.logits, output.logits[:, -seq_len_second:, :]
```

```python
diff = (output_second_part.logits - output.logits[:, -seq_len_second:, :]).abs()
print(f"Max diff: {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
```

```python
assert torch.allclose(output_second_part.logits, output.logits[:, -seq_len_second:, :], atol=1e-7)

# The main result: predicted token is the same
assert output_second_part.logits[0, -1, :].argmax() == output.logits[0, -1, :].argmax(), "predicted tokens differ!"
```

### Is padding side important?

```python
prompt_list = [
    "The left side is the right side",
    "No, the right side is the right side",
    "You can pad sequneces in batch whatever you want",
    "Just tell us what is the truth here!",
]
```

```python
# Padding to the left
tokenizer.padding_side = "left"
tokens_left = tokenizer(
    prompt_list, 
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True,
)
```

```python
output_left = model(
    input_ids=tokens_left["input_ids"].to(model.device),
    attention_mask=tokens_left["attention_mask"].to(model.device),
    use_cache=True,
)

print(tokens_left["attention_mask"])
kv_left = output_left.past_key_values
```

```python
kv_left_second_layer_key = kv_left.key_cache[2][0, ...]
kv_left_second_layer_value = kv_left.value_cache[2][0, ...]

print(kv_left_second_layer_key.shape)
print(kv_left_second_layer_value.shape)
```

```python
# Is KV cache the same?
prompt_first = prompt_list[0]
prompt_first_tokens = tokenizer(prompt_first, return_tensors="pt").to(model.device)
output_first = model(
    input_ids=prompt_first_tokens["input_ids"],
    use_cache=True,
)

print(prompt_first_tokens["attention_mask"])
kv_first = output_first.past_key_values
```

```python
kv_first_second_layer_key = kv_first.key_cache[2][0, ...]
kv_first_second_layer_value = kv_first.value_cache[2][0, ...]

print(kv_first_second_layer_key.shape)
print(kv_first_second_layer_value.shape)
```

```python
# check if it same
kv_first_len = prompt_first_tokens["attention_mask"].shape[-1]

print(torch.allclose(kv_left_second_layer_key[:, -kv_first_len:, :], kv_first_second_layer_key, atol=1e-1))
print(torch.allclose(kv_left_second_layer_value[:, -kv_first_len:, :], kv_first_second_layer_value, atol=1e-3))
```

Our keys are different however all math is the same! Let's see this differnce

```python
diff = (kv_left_second_layer_key[:, -kv_first_len:, :] - kv_first_second_layer_key).abs()
print(f"Max diff: {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
```

What about right side?

```python
# Now pad to the right
tokenizer.padding_side = "right"
tokens_right = tokenizer(
    prompt_list, 
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True,
)
```

```python
output_right = model(
    input_ids=tokens_right["input_ids"].to(model.device),
    attention_mask=tokens_right["attention_mask"].to(model.device),
    use_cache=True,
)

print(tokens_right["attention_mask"])
kv_right = output_right.past_key_values
```

```python
kv_right_second_layer_key = kv_right.key_cache[2][0, ...]
kv_right_second_layer_value = kv_right.value_cache[2][0, ...]

print(kv_right_second_layer_key.shape)
print(kv_right_second_layer_value.shape)
```

```python
print(torch.allclose(kv_right.key_cache[2][0, :, :kv_first_len, :], kv_first_second_layer_key, atol=1e-1))
print(torch.allclose(kv_right.value_cache[2][0, :, :kv_first_len, :], kv_first_second_layer_value, atol=1e-3))
```

```python
diff = (kv_right_second_layer_key[:, :kv_first_len, :] - kv_first_second_layer_key).abs()
print(f"Max diff: {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
```

So we see that since all math are the same, the difference between left and right padding sides is important. Also due to numerical instabilities, with padding we don't get the same KV vector as for single request

## Now let's measure prefill/decode stages

What is prefill?

You got prompt tokens and calculate KV-cache for all tokens in this prompt

```python
import random
import string

def random_string(length):
    alphabet = string.ascii_letters + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + string.digits + " "
    return random.choice(alphabet[:52]) + ''.join(random.choices(alphabet, k=length - 1))

random_string(64)
```

```python
# Create random prompts
prompt_lengths = [1, 4, 8, 16, 32, 64, 128]

prompts = [
    [random_string(64) for _ in range(length)]
    for length in prompt_lengths
]

prompts[1]
```

```python
# Let's measure prefill performance

@torch.no_grad()
def measure_prefill_performance(model, tokenizer, prompt: str, num_gpus: int = 1, num_warmup: int = 2, num_runs: int = 4):

    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    
    # warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
    
    del outputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    model_memory = torch.cuda.memory_allocated()

    times = []

    for _ in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        torch.cuda.synchronize()
        end_time = time.time()   
        times.append(end_time - start_time)
        del outputs


    prefill_time = np.mean(times)
    num_tokens = input_ids.shape[1] * input_ids.shape[0]
    tokens_per_second = num_tokens / prefill_time if prefill_time > 0 else 0
    
    memory_used = (torch.cuda.max_memory_allocated() - model_memory) / 1e9
    
    return {
        'time': prefill_time,
        'tokens': num_tokens,
        'tokens_per_second': tokens_per_second,
        'memory_gb': memory_used
    }
```

```python
prefill_results = []
for i, prompt_batch in enumerate(prompts):
    print(f"Prompt {i+1} ({len(tokenizer.encode(prompt_batch[0])) * len(prompt_batch)} tokens):")
    result = measure_prefill_performance(model, tokenizer, prompt_batch)
    prefill_results.append(result)
    print(f"  Number of tokens: {result['tokens']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Memory per token: {result['memory_gb'] * 1024**2 / result['tokens']:.6f} KB/token")
    print()
```

Let's on decode stage

```python
@torch.no_grad()
def measure_decode_performance(model, tokenizer, prompt: str, num_tokens_to_generate: int = 50, num_warmup: int = 2):
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = model.device
    input_ids = inputs["input_ids"].to(device)

    for _ in range(num_warmup):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    model_memory = torch.cuda.memory_allocated()

    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=num_tokens_to_generate,
            do_sample=False,
            eos_token_id=None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    # Approximately: prefill time + decode time
    # For accuracy, we can measure separately, but for demonstration we use total time
    
    memory_used = (torch.cuda.max_memory_allocated() - model_memory) / 1e9

    total_tokens_generated = num_tokens_to_generate * input_ids.shape[0]

    return {
        'time': total_time,
        'tokens_generated': total_tokens_generated,
        'tokens_per_second': total_tokens_generated / total_time if total_time > 0 else 0,
        'memory_gb': memory_used,
        'output': tokenizer.decode(outputs[0], skip_special_tokens=True)
    }


decode_results = []
for prompt in prompts:
    print(f"Generation for {len(prompt)} prompts:")
    result = measure_decode_performance(model, tokenizer, prompt, num_tokens_to_generate=256)
    decode_results.append(result)
    print(f"  Total tokens: {result['tokens_generated']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Generated text (first 100 characters): {result['output'][:100]}...")
    print()
```

Decode is musch slower than prefill

## Kernels for Prefill/decode?

Let's see if optimzed kernels can help us improve perfomance for prefill/decode stages?

```python
model_with_flash_attn = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="flex_attention",
)
```

```python
# Create random prompts
prompt_lengths = [1, 4, 8, 16, 32]

prompts = [
    [random_string(2048) for _ in range(length)]
    for length in prompt_lengths
]
```

```python
prefill_results = []
for i, prompt_batch in enumerate(prompts):
    print(f"Prompt {i+1} ({len(tokenizer.encode(prompt_batch[0])) * len(prompt_batch)} tokens):")
    result = measure_prefill_performance(model, tokenizer, prompt_batch)
    prefill_results.append(result)
    print(f"  Number of tokens: {result['tokens']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Memory per token: {result['memory_gb'] * 1024**2 / result['tokens']:.6f} KB/token")
    print()
```

```python
prefill_results_with_flash_attn = []
for i, prompt_batch in enumerate(prompts):
    print(f"Prompt {i+1} ({len(tokenizer.encode(prompt_batch[0])) * len(prompt_batch)} tokens):")
    result = measure_prefill_performance(model_with_flash_attn, tokenizer, prompt_batch)
    prefill_results_with_flash_attn.append(result)
    print(f"  Number of tokens: {result['tokens']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Memory per token: {result['memory_gb'] * 1024**2 / result['tokens']:.6f} KB/token")
    print()
```

So, we can see that for long sequnces our prefill was memory bound since flash attention reduces memory footprint. But what about decode? Will flash attention help us there?

```python
decode_results = []
for prompt in prompts:
    print(f"Generation for {len(prompt)} prompts:")
    result = measure_decode_performance(model, tokenizer, prompt, num_tokens_to_generate=256)
    decode_results.append(result)
    print(f"  Total tokens: {result['tokens_generated']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Generated text (first 100 characters): {result['output'][:100]}...")
    print()
```

```python
decode_results_with_flash_attn = []
for prompt in prompts:
    print(f"Generation for {len(prompt)} prompts:")
    result = measure_decode_performance(model_with_flash_attn, tokenizer, prompt, num_tokens_to_generate=256)
    decode_results_with_flash_attn.append(result)
    print(f"  Total tokens: {result['tokens_generated']}")
    print(f"  Time: {result['time']:.4f} sec")
    print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
    print(f"  Generated text (first 100 characters): {result['output'][:100]}...")
    print()
```

So for decode stage kernel optimizations don't give the same bost as for prefill stage

## Let's see what is Continuos Batching

```python
from dataclasses import dataclass
from transformers.cache_utils import DynamicCache
from typing import List

@dataclass
class Request:
    prompt: str
    max_tokens: int
    generated_tokens: List[int] = None
    finished: bool = False
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []
```

```python
@torch.no_grad()
def simple_continuous_batching(model, tokenizer, requests: List[Request]):
    active_requests = requests.copy()
    
    # Prefill
    prompts = [req.prompt for req in active_requests]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    
    # First Step - Get KV 
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    for i, req in enumerate(active_requests):
        req.generated_tokens = [next_tokens[i].item()]
    
    print(f"Handling {len(active_requests)} requests")
    
    # Decode
    iteration = 0
    while active_requests:
        unfinished = [i for i, req in enumerate(active_requests) if not req.finished]
        
        if not unfinished:
            break

        # use only unfinished requests
        current_tokens = next_tokens[unfinished]
        current_attn = attention_mask[unfinished]
        current_attn = torch.cat([
            current_attn,
            torch.ones(len(unfinished), 1, device=model.device, dtype=current_attn.dtype)
        ], dim=1)
        
        active_kv = DynamicCache()
        for layer_idx in range(len(past_key_values.key_cache)):
            active_kv.update(
                past_key_values.key_cache[layer_idx][unfinished],
                past_key_values.value_cache[layer_idx][unfinished],
                layer_idx
            )
        active_past_kv = active_kv

        outputs = model(
            input_ids=current_tokens,
            attention_mask=current_attn,
            past_key_values=active_past_kv,
            use_cache=True
        )
        
        next_tokens_batch = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        
        for batch_idx, orig_idx in enumerate(unfinished):
            req = active_requests[orig_idx]
            token = next_tokens_batch[batch_idx].item()
            req.generated_tokens.append(token)
            
            if len(req.generated_tokens) >= req.max_tokens:
                req.finished = True
        
        # Update state
        next_tokens = next_tokens_batch
        attention_mask = current_attn
        past_key_values = outputs.past_key_values
        
        # Remove completed requests from the active list
        active_requests = [req for req in active_requests if not req.finished]
        
        iteration += 1
        if len(active_requests) > 0:
            print(f"Iteration {iteration}: active requests = {len(active_requests)}")
    
    print(f"Generation finished")
    return requests
```

```python
requests = [
    Request("How does continuos batching work?\n", max_tokens=32),
    Request("What inference engine for LLM do you know\n", max_tokens=35),
    Request("Tell ma all about YSDA.\n", max_tokens=40),
]

results = simple_continuous_batching(model, tokenizer, requests)

for i, req in enumerate(results, 1):
    text = tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
    print(f"\n[{i}] {req.prompt}")
    print(f"Generated text: {text}")
    print(f"Tokens: {len(req.generated_tokens)}")
```

```python

```
