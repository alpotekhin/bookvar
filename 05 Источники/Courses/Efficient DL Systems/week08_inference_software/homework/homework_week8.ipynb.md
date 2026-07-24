---
title: "Mini-edlang"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week08_inference_software/homework/homework_week8.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/homework_week8.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



This week, your task is to implement an inference server for LLMs. Let's call this server **__mini-edlang__**.

Nowadays, popular inference servers (SGLang, vLLM, etc.) provide the user with a variety of features, allowing for maximum acceleration of LLM inference in a production environment. You don't need to create a production‑ready server, just implement the basic features.

The purpose of the inference server is to process the incoming flow of requests to the LLM. Requests come at random times, so it is necessary to implement **Continuous batching**. It involves more than just dividing each request into prefill and decode stages, but also keeping the current state of the request queue, making decisions about switching from prefill to decode and vice versa, collecting metrics, and providing a user‑friendly asynchronous server interface.

### Project Structure

In our implementation we are gonna divide our code in the following folders:

- __Entrypoints__:
        Here we have 2 files: __engine.py__ and __config.py__. In __engine.py__ you can find class InferenceEngine - backbone of our LLM engine implementation based upon transformers. Within config.py you will find basic configuration. Your task here will be to implement all missing methods of InferenceEngine and check if they work correctly

- __Managers__:
        This folder contains **scheduler_manager.py** and **metric_manager.py**. **scheduler_manager.py** is responsible for schedule our requests, so we can say it is the main part of our server. In **metric_manager.py** you will find drafts for MetricManager class, which you need to complete to collect metrics to profile our inference

- __Server__:
        Here you will find implementation for asynchronious server. You don't need to rewrite it (unless you have idea how to make it more efficient :))

### Grades

Your homework will be divided into four parts. The points are distributed among the parts as follows:

* Engine - 2 points + 1 bonus
* Scheduler - 2 points
* Metrics - 2 points + 1 bonus
* Scheduler policy and benchmarking - 4 points

### Environment

For your convenience, this directory contains `requirements.txt`. To easily install the dependencies with uv, you can run

```uv pip install -r requirements.txt```

Also, we recommend working inside a virtual environment, which you can also create with uv:

```
uv venv
source .venv/bin/activate 
```

Let's start!

## Part 1. Engine (2 points + 1 bonus)

In this task you need to complete all unfinished methods of InferenceEngine from __inference_server/entrypoints/engine.py__. 

Below are the assertion statements for each method; to get the full grade, you should pass all of them. Remember that passing asserts is a necessary but not a sufficient condition! Your points will be awarded based on your code implementation.

Since we manually divide our generation into prefill and decode stages, we can't use model.generate method and pass Sampling Params into it. The only way use Sampling Params is to apply them ourselves. Here, you should implement just greedy decoding.

```python
import torch

from edlang.entrypoints.engine import InferenceEngine, Request
from edlang.entrypoints.config import EngineConfig, ModelConfig
```

```python
model_config = ModelConfig(model_name="Qwen/Qwen3-4B", device="cuda:0", torch_dtype=torch.float16, max_prompt_length=512)
engine_config = EngineConfig(model_config=model_config)

engine = InferenceEngine(engine_config)
```

```python
prompt_list = [
    "Why do we need inference servers for LLMs?\n",
    "How does Continuous Batching improve LLM inference throughput?\n",
    "What is Paged Attention in the context of LLMs?\n",
    "How can we reduce the latency of LLM inference for real-time applications?\n",
    "What are the main challenges of serving large LLMs in production?\n",
]

MAX_NEW_TOKENS = 59
```

```python
request_list = [
    Request(request_id=i, prompt=prompt, max_new_tokens=MAX_NEW_TOKENS) for i, prompt in enumerate(prompt_list)
]
token_list = [
    engine.tokenizer.encode(prompt) for prompt in prompt_list
]
batch_prefill_result = engine.prefill(request_list)
```

```python
# check prefill - 1 point

# Batch‑level checks
assert len(batch_prefill_result.request_ids) == len(request_list)
assert len(batch_prefill_result.new_tokens) == len(request_list)
assert len(batch_prefill_result.finished) == len(request_list)
assert set(batch_prefill_result.request_ids) == {r.request_id for r in request_list}

for request, tokens in zip(request_list, token_list):
    idx = batch_prefill_result.request_ids.index(request.request_id)
    new_tokens = batch_prefill_result.new_tokens[idx]
    finished_flag = batch_prefill_result.finished[idx]

    # check if attention mask is calculated correctly
    assert len(tokens) == int(request.attention_mask.sum())

    # check if KV cache stores correctly
    num_layers = engine.model.config.num_hidden_layers
    assert num_layers == len(request.past_key_values.key_cache)

    num_kv_heads = engine.model.config.num_key_value_heads
    head_dim = engine.model.config.head_dim
    kv_shape = request.past_key_values.key_cache[0].shape
    assert kv_shape[0] == 1 and kv_shape[1] == num_kv_heads and kv_shape[3] == head_dim
```

```python
# check decode and text generation - 2 point

answer_to_prompts = []

for request in request_list:
    for _ in range(request.max_new_tokens):
        batch_result = engine.decode([request])

        assert batch_result.request_ids == [request.request_id]
        assert len(batch_result.new_tokens) == 1
        assert len(batch_result.finished) == 1

        if request.is_finished:
            assert batch_result.finished[0] is True
            assert batch_result.finished[0] == request.is_finished

    assert request.is_finished

    # check if request is truly finished
    number_of_tokens = list(request.generated_tokens)
    for _ in range(request.max_new_tokens):
        engine.decode([request])

    assert number_of_tokens == request.generated_tokens

    print(f"{'='*20} Request #{request.request_id} {'='*20}")
    print(f"Prompt: {request.prompt!r}\n")
    print("Generated:")
    answer_to_prompts.append(engine.get_generated_text(request))
    print(answer_to_prompts[-1])
    print("="*54 + '\n')
```

### Bonus part - sampling parameters (1 Point)

As you saw before, our engine cannot use sampling parameters. Your bonus task is to complete the method **engine._sample** to correctly work with the following sampling params:

```python
sampling_params_list = [    
    "temperature",
    "top_k",
    "top_p",
    "do_sample",
    "eos_token_id",
    "ignore_eos_token",
]
```

Below are some tests that you should pass:

```python
def make_request(sampling_params):
    return Request(
        request_id=0,
        prompt="",
        max_new_tokens=1,
        sampling_params=sampling_params,
    )

logits = torch.tensor([0.1, 2.0, 0.3], device=engine.model_config.device)
```

```python
# We have to get the same argmax
token = engine._sample(logits, make_request(None))
assert token == 1, f"Expected greedy argmax token 1, got {token}"
```

```python
# Check top_k
token = engine._sample(
    logits,
    make_request({"top_k": 1, "do_sample": True}),
)
assert token == 1, f"Expected token 1 with top_k=1, got {token}"
```

```python
# Check top_p
logits = torch.tensor([10.0, 0.0, 0.0], device=engine.model_config.device)
token = engine._sample(
    logits,
    make_request({"top_p": 0.9, "do_sample": True}),
)
assert token == 0, f"Expected token 0 with top_p=0.9, got {token}"
```

```python
# check eos_token_id and ignore tokens
logits = torch.tensor([5.0, 1.0, 0.0], device=engine.model_config.device)

token = engine._sample(
    logits,
    make_request({"do_sample": True, "top_k": 1, "eos_token_id": 0}),
)
assert token == 0, f"Expected eos token 0 to be sampled, got {token}"

token = engine._sample(
    logits,
    make_request({"do_sample": True, "top_k": 1, "eos_token_id": 0, "ignore_eos_token": True}),
)
assert token == 1, f"Expected token 1 when ignore_eos_token=True, got {token}"
```

## Part 2.1. Scheduler (2 Points)

Now you need to implement EDLangScheduler — the scheduler that will handle our request flow. You can find the interface of this class in **inference_server/managers/scheduler_manager.py**. Again, your task is to complete all methods. If you want, you can create additional private methods for scheduler, but describe their purpose here in this notebook as part of your report

```python
from edlang.managers.scheduler_manager import EDLangScheduler
```

```python
scheduler = EDLangScheduler(engine=engine)
```

```python
# check request registration
for i, prompt in enumerate(prompt_list):
    scheduler.add_request(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert len(scheduler.waiting_queue) == i + 1, f"Len of your request queue is {len(scheduler.waiting_queue)} for {i} requests"
    assert len(scheduler.active_requests) == 0
```

```python
# .clear should remove all requests and reset all counters and queues
scheduler.clear()
```

```python
scheduler.active_requests
```

```python
# check prefill step with current prefill policy
scheduler.add_request("Hello, world!", max_new_tokens=MAX_NEW_TOKENS)
scheduler._prefill_step()

for i, prompt in enumerate(prompt_list):
    scheduler.add_request(prompt, max_new_tokens=MAX_NEW_TOKENS)
    scheduler._prefill_step()
    assert len(scheduler.waiting_queue) == i + 1, f"Prefill schould transform request from waiting to active if there are no active requests, but you got {len(scheduler.waiting_queue)} waiting requests"
    assert len(scheduler.active_requests) == 1
```

```python
scheduler.clear()
```

```python
# check decode step
error_list = []

for i, prompt in enumerate(prompt_list):
    scheduler.add_request(prompt, max_new_tokens=MAX_NEW_TOKENS)
    request = scheduler.waiting_queue[-1]
    scheduler._prefill_step()

    for _ in range(request.max_new_tokens):
        scheduler._decode_step()
    
    finished_request = scheduler.get_finished_requests()[0]

    assert finished_request.is_finished
    assert finished_request == request

    print(f"{'='*20} Request #{finished_request.request_id} {'='*20}")
    print(f"Prompt: {finished_request.prompt!r}\n")
    print("Generated:")
    scheduler_answer = scheduler.engine.get_generated_text(finished_request)

    if scheduler_answer != answer_to_prompts[i]:
        error_list.append(f"Expected: {answer_to_prompts[i]}, \n\nGot: {scheduler_answer}")
    print(scheduler_answer)
    print("="*54 + '\n')
```

Below you can see how many generations differs from different approaches

```python
assert len(error_list) < len(prompt_list) / 4, "You have too many errors in your code"

for error in error_list:
    print(error)
```

If you passed all assert statements above, then (probably) your code is working and we can benchmark it like a real production inference server! But firstly, we need to implement the metrics

## Part 2.2. Metrics (2 point + 1 Bonus)

To properly evaluate and benchmark our inference server, we need to understand what metrics to measure. When talking about LLM inference, there are plenty of metrics we can collect. Your task is to implement the calculation of the following basic metrics:

### Core Metrics

* **Generation Throughput** (Gen throughput) - measured in tokens per second. This metric quantifies the overall generation speed of the inference server. It is calculated as the total number of tokens generated across all requests in one second during the decode step. Higher throughput indicates better server efficiency and capacity to handle multiple concurrent requests.

* **Time To First Token (TTFT)** - measures the latency between when a request is submitted to the inference server and when the first generated token is returned to the client. This is a critical metric for user experience, especially in interactive applications like chatbots, as it determines the perceived responsiveness of the system.

* **Time Per Output Token (TPOT)** or **Inter-token Latency** - the average time between generating consecutive tokens during the decode phase. It is calculated as the decode time divided by the number of generated tokens. This metric helps understand the steady-state generation speed.

* **Request Throughput** - measured in requests per second (RPS). This indicates how many complete requests the server can process per unit of time, which is important for understanding the server's capacity under load.

The main code for the metrics lies in **inference_server/managers/metric_manager.py**. You have some requirements for this class's interface, but the inner implementaion of the methods is up to you

```python
# To enable metrics we need to pass enable_metrics=True to EDLangScheduler constructor
from edlang.managers.scheduler_manager import SchedulerConfig

scheduler_config = SchedulerConfig(enable_metrics=True)

scheduler = EDLangScheduler(engine=engine, config=scheduler_config)

# check metrics
for i, prompt in enumerate(prompt_list):
    scheduler.add_request(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert scheduler.metrics_manager.waiting_queue_num == 1

    # first step is prefill step
    scheduler.step()

    for i in range(MAX_NEW_TOKENS):
        scheduler.step()
        assert scheduler.metrics_manager.waiting_queue_num == 0
        assert scheduler.metrics_manager.active_requests_num == 1
        assert scheduler.metrics_manager.throughput_tokens_per_second > 0
    
    finished_request = scheduler.get_finished_requests()[0]
```

```python
scheduler.clear()
MAX_NEW_TOKENS = 256

# let's collect requests
for i, prompt in enumerate(prompt_list):
    scheduler.add_request(prompt, max_new_tokens=MAX_NEW_TOKENS)
    # assert scheduler.metrics_manager.waiting_queue_num == 1

    scheduler.step()

    assert scheduler.metrics_manager.active_requests_num > 0
    assert scheduler.metrics_manager.throughput_tokens_per_second > 0

print(f"all requests have been added")

for _ in range(MAX_NEW_TOKENS - len(prompt_list) - 1):
    scheduler.step()
    assert scheduler.metrics_manager.active_requests_num > 0
    assert scheduler.metrics_manager.throughput_tokens_per_second > 0

for i in range(len(prompt_list)):
    scheduler.step()
    assert scheduler.metrics_manager.throughput_tokens_per_second > 0

    # get_finished_requests should be the only way to remove a finished request from active requests
    scheduler.get_finished_requests()
```

### BONUS: Additional Important Metrics in LLM Inference (1 point)

While you only need to implement the metrics above, here are other commonly used metrics in production LLM inference systems.Adding each one will grant you extra 0.25 points:

* **End-to-End Latency (Total Time)** - the total time from request submission to completion (when the last token is generated or the request finishes). This is the sum of TTFT and the time to generate all remaining tokens.

* **GPU Utilization** - the percentage of time the GPU is actively performing computations. High utilization indicates efficient use of hardware resources, while low utilization may suggest bottlenecks elsewhere (e.g., memory bandwidth, CPU preprocessing).

* **Memory Usage** - tracks GPU memory consumption, including model weights, KV cache, and intermediate activations. This is crucial for understanding resource constraints and optimizing batch sizes.

* **Queue Wait Time** - the time a request spends waiting in the queue before being processed. This metric is important for understanding server load and scheduling efficiency, especially under high traffic conditions.

```python
# Show how your additional metrics work here
```

## Part 3. Scheduler Policy (4 points)

Now we have a functioning engine, scheduler, and metrics system, but our inference framework is still far from optimal. The main limitation lies in our scheduler, specifically in the `_decide_prefill_batch_size` function, which currently makes very inefficient decisions regarding prefill and decode handling.

Your task is to iteratively experiment and develop better strategies to optimize this function. Design and implement at least **THREE** different policies for deciding the prefill batch size, and benchmark their performance.

For each policy you try, analyze the impact on throughput, latency, and overall efficiency. Compare the results and determine which approach works best in practice. This is an open-ended, exploratory exercise: be creative and systematic in your experimentation, and justify your conclusions based on your findings.

Below you can see an example of a benchmarking function. It takes RPS, number of requests and mode for testing
* hard_prefill - huge prompt and small max_new_tokens
* hard_decode - small prompt and huge max_new_tokens
* medium - same size for prompt and max_new_tokens

To start your server, you can open an additional terminal and run

```
python -m edlang.server.launch
```

Note: see what additional arguments you can pass with parser.

Example of a benchmark call:

```python
from edlang.test_benchmark import benchmark
import json

result = benchmark(mode="hard_decode", rps=2.0, num_requests=10)

print(json.dumps(result, indent=2))
```

Now, you can change the `_decide_prefill_batch_size` function, collect the results, and write a report summarizing what you did and how it helped to imporve performance

* Analysis and visualization: You can add any functions to analyze results: build plots (matplotlib, seaborn), compute statistics, compare policies. Use the cells below for your report.

* Changing the benchmark: If you have a strong reason to modify the benchmark code (test_benchmark.py), you may do so, but you must justify the changes in your report (why, what exactly you changed, and how it affects the results).

```python
# Your analysis here
```

## Comments

Leave any comments about the homework here: what you liked, what was difficult, suggestions for improvement, remarks, etc

```python
# Your comments here
```
