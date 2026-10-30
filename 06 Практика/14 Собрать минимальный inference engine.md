---
title: Собрать минимальный inference engine
type: practice
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-29
source_language: mixed
---

# Собрать минимальный inference engine

Начните с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|EDLS Week 8 seminar]] и выполните [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|оригинальный homework notebook]].

Минимальный engine должен отдельно представлять request state, prefill, decode,
KV-cache и scheduler. Сначала добейтесь совпадения greedy generation с
эталонной реализацией при batch=1. Затем добавьте несколько запросов разной
длины и continuous batching.

Измеряйте TTFT, общую задержку запроса, число выходных токенов в секунду и
пиковую память KV. Межтокенная задержка (ITL) — каждый отдельный интервал
$t_i-t_{i-1}$ между поступлениями соседних токенов. TPOT для запроса с $N>1$
выходными токенами — средний интервал $(t_N-t_1)/(N-1)$; при $N=1$ он не
определён. Хвост распределения ITL может содержать длинные паузы, скрытые
в среднем TPOT, поэтому покажите оба распределения и единицу усреднения.
При объединении нескольких токенов в один сетевой chunk время его получения
не позволяет восстановить отдельные серверные интервалы: измеряйте их на
сервере или явно называйте клиентскую метрику межпакетной.
На timeline должны различаться prefill и decode. Проведите sweep
concurrency и покажите точку насыщения, где throughput перестаёт расти, а
queueing и tail latency увеличиваются.

Сдайте код, correctness tests, workload description и графики. Одно число
tokens/s без распределения длин, arrival process и SLO не считается результатом.

<!-- BEGIN REVIEW: EDLS neural KV 2026-10-26 -->
## Neural reference: prefill, decode and request-owned KV

A prefill pass produces both a cache of the prompt and the first output token.
That token has been selected from logits but has not yet passed through the
model itself. The next decode pass consumes it, appends its K/V and selects
another token. Keeping the selected token separate from the cached prefix
prevents an extra forward pass at the end of generation.

The EDLS homework states the implementation constraint as follows:

<!-- BEGIN SOURCE: EDLS manual greedy contract -->
> Since we manually divide our generation into prefill and decode stages, we can't use model.generate method and pass Sampling Params into it. The only way use Sampling Params is to apply them ourselves. Here, you should implement just greedy decoding.
<!-- END SOURCE: EDLS manual greedy contract -->

*Efficient DL Systems, Week 8 homework, Part 1, commit
`e632aa89ca9e6638d52e1b686095e7442faffbb0`;
[original notebook](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/homework_week8.ipynb),
[MIT license](/licenses/edls-e632aa89-MIT.txt). The English source is unchanged.*

The reference below completes the core prefill/decode contract from
[the course engine interface](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/edlang/entrypoints/engine.py).
It accepts token IDs so that it can be tested without downloading a tokenizer
or pretrained weights. It uses a dense full-attention Qwen3 with GQA and a
DynamicCache; it does not implement an HTTP service, paging, shared prefixes
or the homework's sampling bonus.

### The cache ends one token before the returned sequence

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png]]

*Rémi Ouazan Reboul, Arthur Zucker and Luc Georges, Hugging Face,
[Continuous batching from first principles — KV-cache](https://huggingface.co/blog/continuous_batching#kv-cache),
CC BY-NC-SA 4.0. Blue columns are cached states; the white column is the current
input token. The predicted next token on the right is not in that cache yet.
Head axes are omitted in the source diagram.*

For a prompt of three tokens and a three-token answer:

| Operation | Tokens passed to the model | KV length after the pass | Newly returned token |
|---|---|---:|---|
| prefill | prompt positions 0, 1, 2 | 3 | y1 |
| decode 1 | y1 | 4 | y2 |
| decode 2 | y2 | 5 | y3 |

After the last row, the request is finished. Feeding y3 into the model would
compute an unwanted fourth output token. At ordinary completion the cache
contains $P+N-1$ positions for a prompt of length $P$ and $N>0$ emitted tokens.
For $N=0$, no model pass is needed. If a later conversation turn reuses this
cache, the last emitted token must be accounted for before adding new text.

The same boundary applies to EOS. If prefill selects EOS or already reaches a
one-token generation budget, mark the request finished immediately. A finished
or cancelled request must not run through the model again.

### Padding columns and token positions are separate coordinates

The course interface uses right-padded temporary KV batches. Suppose the next
decode batch contains C with two cached positions and B with five. Packing
both caches gives physical width 5. The new K/V column is appended at physical
column 5 for both rows, even though C's new token has logical position 2.

| Request | Valid old KV columns | Temporary padding | New physical KV column | position_ids | Attention mask |
|---|---|---|---:|---:|---|
| C | 0, 1 | 2, 3, 4 | 5 | 2 | 1, 1, 0, 0, 0, 1 |
| B | 0, 1, 2, 3, 4 | none | 5 | 5 | 1, 1, 1, 1, 1, 1 |

The mask excludes C's padding columns. Its logical position is used for RoPE;
`cache_position` locates the appended column in this temporary batch.
After the pass, save columns `[0, 1, 5]` for C and all six columns for B.
C's compact cache then has length 3. Slicing the first three physical columns
would save a padding column and discard the new K/V.

Each request owns its compact cache. Every decode call packs the currently
active requests in the caller's current order, runs the model, then saves each
row back to that same request ID. Removing A or swapping B and C therefore
does not reinterpret an old batch row as another request's history. The
[original HF diagrams below](#c-trace-a-scheduler-before-timing-a-model)
show how padding and cached tokens appear in the attention mask.

The code is pinned to Transformers 4.57.1. The course seminar installs 4.53.0
and accesses `key_cache`; this reference uses the supported `cache[layer]`
interface of the checked version. See the
[Qwen3 forward implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)
and [cache implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py).
This is not a claim of compatibility with arbitrary Transformers versions,
sliding-window caches or other model families.

### Complete reference implementation

The implementation is visible here and available as a
[downloadable Python file](/examples/edls_kv_reference.py).
Its repository path is `06 Практика/Code/edls_kv_reference.py`.
It right-pads only the temporary batch; `_save` selects valid columns into
new per-request tensors. Finished caches are retained for inspection in this
exercise. A server must release them when they are no longer needed.

<!-- BEGIN SOLUTION: EDLS KV module -->
```python
"""CPU teaching solution for the EDLS Week 8 prefill/decode cache contract.

Adapted from the MIT-licensed course interface at commit
e632aa89ca9e6638d52e1b686095e7442faffbb0. Copyright (c) 2021 course authors.
See /licenses/edls-e632aa89-MIT.txt. This is not an HTTP server or a benchmark.
Requires transformers==4.57.1; full-attention DynamicCache only.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class Request:
    request_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    past_key_values: DynamicCache | None = None
    stop_reason: str | None = None

    @property
    def is_finished(self):
        return self.stop_reason is not None

    def cancel(self):
        if not self.is_finished:
            self.stop_reason = "cancelled"


@dataclass
class BatchResult:
    request_ids: list[str]
    new_tokens: list[list[int]]
    finished: list[bool]


class KVGreedyEngine:
    """Right-padded temporary batches; compact, independently owned request KV."""

    def __init__(self, model, max_context=128, pad_token_id=0):
        if not isinstance(max_context, int) or not 1 <= max_context <= model.config.max_position_embeddings:
            raise ValueError("max_context must fit the configured position limit")
        if not isinstance(pad_token_id, int) or not 0 <= pad_token_id < model.config.vocab_size:
            raise ValueError("pad token outside vocabulary")
        self.model = model.eval()
        self.max_context = max_context
        self.pad_token_id = pad_token_id
        self.device = next(model.parameters()).device

    def _validate(self, requests):
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("duplicate request IDs in one batch")
        for r in requests:
            if not isinstance(r.max_new_tokens, int) or r.max_new_tokens < 0:
                raise ValueError("max_new_tokens must be a nonnegative integer")
            if not r.prompt_ids or len(r.prompt_ids) > self.max_context:
                raise ValueError("prompt must fit the context and contain a token")
            if any(not isinstance(t, int) or not 0 <= t < self.model.config.vocab_size
                   for t in r.prompt_ids):
                raise ValueError("prompt token outside vocabulary")
            if r.eos_token_id is not None and not 0 <= r.eos_token_id < self.model.config.vocab_size:
                raise ValueError("EOS token outside vocabulary")

    def _save(self, request, cache, row, positions):
        # index_select copies the selected columns; no view pins another row.
        owned = DynamicCache()
        for layer_idx, (key, value) in enumerate(cache):
            owned.update(key[row:row+1].index_select(2, positions),
                         value[row:row+1].index_select(2, positions), layer_idx)
        request.past_key_values = owned

    def _emit(self, request, token):
        request.generated_tokens.append(int(token))
        if request.eos_token_id is not None and token == request.eos_token_id:
            request.stop_reason = "eos"
        elif len(request.generated_tokens) >= request.max_new_tokens:
            request.stop_reason = "length"
        elif request.past_key_values.get_seq_length() >= self.max_context:
            request.stop_reason = "context"

    @staticmethod
    def _result(requests, tokens):
        return BatchResult([r.request_id for r in requests],
                           [[int(t)] for t in tokens],
                           [r.is_finished for r in requests])

    @torch.no_grad()
    def prefill(self, requests):
        self._validate(requests)
        if any(r.past_key_values is not None and not r.is_finished for r in requests):
            raise ValueError("active request was already prefilled")
        for r in requests:
            if not r.is_finished and r.max_new_tokens == 0:
                r.stop_reason = "length"
        active = [r for r in requests if not r.is_finished]
        if not active:
            return BatchResult([], [], [])
        lengths = torch.tensor([len(r.prompt_ids) for r in active], device=self.device)
        width = int(lengths.max())
        ids = torch.full((len(active), width), self.pad_token_id,
                         dtype=torch.long, device=self.device)
        for row, r in enumerate(active):
            ids[row, :len(r.prompt_ids)] = torch.tensor(r.prompt_ids, device=self.device)
        columns = torch.arange(width, device=self.device)
        mask = columns[None, :] < lengths[:, None]
        output = self.model(input_ids=ids, attention_mask=mask.long(),
                            position_ids=columns[None, :].expand(len(active), -1),
                            use_cache=True)
        rows = torch.arange(len(active), device=self.device)
        # The final physical column may be padding: gather each valid last row.
        tokens = output.logits[rows, lengths - 1].argmax(-1).tolist()
        for row, (r, token) in enumerate(zip(active, tokens)):
            positions = torch.arange(len(r.prompt_ids), device=self.device)
            self._save(r, output.past_key_values, row, positions)
            self._emit(r, token)
        return self._result(active, tokens)

    @torch.no_grad()
    def decode(self, requests):
        self._validate(requests)
        active = [r for r in requests if not r.is_finished]
        if not active:
            return BatchResult([], [], [])
        if any(r.past_key_values is None or not r.generated_tokens for r in active):
            raise ValueError("decode requires prefilled request state")
        lengths = [r.past_key_values.get_seq_length() for r in active]
        width = max(lengths)
        packed = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            keys, values = [], []
            for r, length in zip(active, lengths):
                key, value = r.past_key_values[layer_idx]
                keys.append(F.pad(key, (0, 0, 0, width - length)))
                values.append(F.pad(value, (0, 0, 0, width - length)))
            packed.update(torch.cat(keys), torch.cat(values), layer_idx)
        mask = torch.zeros((len(active), width + 1), dtype=torch.long, device=self.device)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        mask[:, width] = 1
        ids = torch.tensor([[r.generated_tokens[-1]] for r in active], device=self.device)
        output = self.model(input_ids=ids, attention_mask=mask,
                            position_ids=torch.tensor(lengths, device=self.device)[:, None],
                            cache_position=torch.tensor([width], device=self.device),
                            past_key_values=packed, use_cache=True)
        tokens = output.logits[:, -1].argmax(-1).tolist()
        for row, (r, length, token) in enumerate(zip(active, lengths, tokens)):
            # New KV was appended at `width`, beyond temporary right-padding.
            positions = torch.tensor([*range(length), width], device=self.device)
            self._save(r, output.past_key_values, row, positions)
            self._emit(r, token)
        return self._result(active, tokens)


@torch.no_grad()
def uncached_greedy(model, prompt_ids, max_new_tokens):
    """Independent oracle: recompute the whole valid prefix, with no cache."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = []
    for _ in range(max_new_tokens):
        token = int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
        generated.append(token)
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return generated
```
<!-- END SOLUTION: EDLS KV module -->

### Run a neural comparison without downloading weights

This model has two layers, width 32, four query heads, two KV heads, head
dimension 8 and a vocabulary of 64 token IDs. It has 22,720 randomly initialized
parameters. Its output IDs test execution equivalence; they are not meaningful
language responses and do not measure the quality of Qwen3 checkpoints.

Run the implementation block and this block in the same Python process.
Validated dependencies are PyTorch 2.14.1, NumPy 2.3.5 and Transformers 4.57.1.

<!-- example: neural-kv-smoke -->
```python
from transformers import Qwen3Config, Qwen3ForCausalLM

torch.set_num_threads(1)
torch.manual_seed(8)
config = Qwen3Config(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8,
    max_position_embeddings=128, attention_dropout=0.,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
    use_sliding_window=False,
)
config._attn_implementation = "eager"
model = Qwen3ForCausalLM(config).eval()
engine = KVGreedyEngine(model)

requests = [
    Request("A", (1, 4, 5), 1),
    Request("B", (1, 7, 8, 9, 10), 5),
    Request("C", (1, 12), 3),
]
engine.prefill(requests)
while any(not r.is_finished for r in requests):
    engine.decode(requests[::-1])  # exercise a different row order

for request in requests:
    expected = uncached_greedy(model, request.prompt_ids, request.max_new_tokens)
    assert request.generated_tokens == expected
    print(request.request_id, request.generated_tokens,
          request.past_key_values.get_seq_length())
```
<!-- end example: neural-kv-smoke -->

The checked run emits:

| Request | Output IDs | Final KV length |
|---|---|---:|
| A | 50 | 3 |
| B | 32, 27, 34, 28, 63 | 9 |
| C | 63, 59, 24 | 4 |

A consumes its entire output budget in prefill. C finishes before B; reversing
the list on subsequent iterations must not change B's result. The separate
checker also compares K/V tensors with an uncached full-prefix pass, checks
all six initial request orders, admits a new request after decoding has begun,
and tests EOS, cancellation, zero/one-token budgets and the context boundary.

To run the repository check:

```bash
uv run --with torch==2.14.1 --with numpy==2.3.5 --with transformers==4.57.1 \
  python publishing/tools/check_edls_kv_reference.py
```

The original seminar's small batching example removes completed Request objects
without consistently compacting the remaining token/cache rows. Once a request
disappears from the middle or front, old row numbers no longer identify the
same surviving requests. It also does not stop a one-token request immediately
after prefill. The reference fixes those state transitions; it is an attributed
solution, not an unchanged copy of that function.

The original GPU exercises remain useful for performance work, but their
generation timer includes prefill and must not be relabeled as pure decode
latency. Their padding experiments also require explicit attention masks and
consistent position IDs before differences can be attributed to numerical
rounding. This CPU test does not run the course's Qwen3-4B checkpoint, CUDA
kernels, FlashAttention, an asynchronous server or a throughput benchmark.
<!-- END REVIEW: EDLS neural KV 2026-10-26 -->

<!-- BEGIN REVIEW: text checkpoint runtime 2026-10-29 -->
## Text input and a saved checkpoint

The token-ID tests above do not exercise a tokenizer or model loading.
`Code/text_checkpoint_lab.py` in the
[runtime lab archive](/examples/runtime-labs-2026-10-29.zip) adds that boundary.
From the extracted archive root, run:

```bash
uv run --with torch==2.14.1 --with numpy==2.3.5 --with transformers==4.57.1 \
  python Code/text_checkpoint_lab.py --device cpu --output Results/text-cpu.json
```

The program builds an 18-token WordLevel vocabulary from four declared English
and Russian sentences. It saves the tokenizer, special-token mapping, Qwen3
configuration and random weights to a temporary checkpoint directory, then
reloads both model and tokenizer with `local_files_only=True`. The complete
checkpoint occupies 84,944 bytes in the checked run. No pretrained weights or
text datasets are downloaded.

For `the cat reads a book`, tokenization produces `[4,8,13,5,6]`. The restored
tokenizer must produce exactly those IDs. The CPU logits before saving and after
loading must also match exactly. The saved model then generates five tokens
through the existing KV engine, alongside three other prompts of different
lengths. After every request finishes, the cache length must be input length
plus four; the fifth emitted token has not been consumed by a model pass.
The script compares each generated sequence with uncached greedy decoding.

The model has 19,776 random parameters. For the first prompt it emits
`[PAD] [PAD] [PAD] book [PAD]`. That output is expected to be meaningless:
this test checks artifact loading, token mapping and cache execution, not
language quality. It also checks an unseen word maps to `[UNK]`; the small
WordLevel tokenizer is not the Qwen3 tokenizer distributed with pretrained
checkpoints. EOS is disabled for this fixed-length comparison; EOS behavior
remains covered by the separate engine checks above.

The same four requests passed on CPU and on the available Apple MPS device.
Use `--device mps` to repeat the latter. MPS reports its allocation and driver
memory at the end of the run; those snapshots are not physical peak memory.
`--device cuda` fails before creating an output report if CUDA is unavailable.
Testing a pretrained checkpoint, CUDA peak memory, or GPU capacity at a chosen
SLO requires a separate workload and hardware run.
<!-- END REVIEW: text checkpoint runtime 2026-10-29 -->

<!-- BEGIN REVIEW: EDLS metric receipts 2026-10-26 -->
## Measure the request, the token stream and the experiment separately

The course's `/generate` endpoint waits for a completed `GenerateResponse`.
It returns the output IDs and text in one JSON response. A client can measure
the response delay and count output IDs, but that response contains no first-
or intermediate-token timestamps. Dividing its total latency by output length
does not recover TTFT or the intervals during generation.

Record a server event when a request enters the queue, when it is admitted,
when each output token becomes available, and when it terminates. Use one
monotonic clock, such as `time.perf_counter()`, for those server intervals.
Client timestamps measure a different path, including transport and client
processing; do not subtract timestamps from unrelated clock origins. A server
token timestamp also differs from a GPU kernel duration, which needs the
device's timing/synchronization procedure.

For request A, submission at 0.00 s, admission at 0.10 s and output tokens at
0.30, 0.50 and 0.90 s give a queue wait of 0.10 s and TTFT of 0.30 s. The two
ITLs are 0.20 and 0.40 s. Request TPOT is their mean, 0.30 s; it conceals the
0.40 s pause. For one emitted token there is no adjacent-token interval and
TPOT is undefined. A rejected request has neither TTFT nor TPOT; it still
belongs in the offered-request and failure counts.

### An executable receipt calculation

The following records describe a fixed one-second teaching fixture. Times are
assigned inputs, not observations from a GPU or the neural smoke test above.
`finished` means the recorded termination time, including cancellation or
rejection. `tokens` contains server availability timestamps, not token IDs.

<!-- example: edls-metric-receipts -->
```python
from math import isfinite
from statistics import fmean

def request_report(records, start, end):
    if not isfinite(start) or not isfinite(end) or end <= start:
        raise ValueError("positive finite experiment interval required")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate request IDs")
    rows = []
    for r in records:
        status = r["status"]
        if status not in {"completed", "cancelled", "rejected", "error"}:
            raise ValueError("every receipt must have a terminal outcome")
        submitted, admitted, finished = r["submitted"], r["admitted"], r["finished"]
        tokens = list(r["tokens"])
        if admitted is None and tokens:
            raise ValueError("tokens require admission")
        if status == "rejected" and (admitted is not None or tokens):
            raise ValueError("this fixture defines rejection before admission")
        events = [submitted, *([] if admitted is None else [admitted]), *tokens, finished]
        if (not all(isfinite(t) for t in events)
                or events != sorted(events) or events[0] < start or events[-1] > end):
            raise ValueError("events must be ordered and inside the experiment")
        itls = [b - a for a, b in zip(tokens, tokens[1:])]
        rows.append(dict(
            id=r["id"], status=status, output_tokens=len(tokens),
            queue_wait=None if admitted is None else admitted - submitted,
            ttft=None if not tokens else tokens[0] - submitted,
            itls=itls, tpot=None if not itls else fmean(itls),
            latency=finished - submitted,
        ))
    completed = [r for r in rows if r["status"] == "completed"]
    tpots = [r["tpot"] for r in completed if r["tpot"] is not None]
    intervals = [dt for r in completed for dt in r["itls"]]
    emitted = sum(r["output_tokens"] for r in rows)
    return rows, dict(
        offered=len(rows), completed=len(completed),
        outcomes={s: sum(r["status"] == s for r in rows)
                  for s in ["completed", "cancelled", "rejected", "error"]},
        emitted_tokens=emitted,
        completed_tokens=sum(r["output_tokens"] for r in completed),
        output_tokens_per_second=emitted / (end - start),
        completed_requests_per_second=len(completed) / (end - start),
        mean_request_tpot=None if not tpots else fmean(tpots),
        pooled_mean_itl=None if not intervals else fmean(intervals),
    )

receipts = [
    dict(id="A", submitted=0., admitted=.10, tokens=[.30, .50, .90], finished=.90, status="completed"),
    dict(id="B", submitted=.10, admitted=.60, tokens=[.80], finished=.80, status="completed"),
    dict(id="C", submitted=.20, admitted=None, tokens=[], finished=.40, status="rejected"),
    dict(id="D", submitted=.20, admitted=.70, tokens=[.95], finished=1., status="cancelled"),
    dict(id="E", submitted=0., admitted=0., tokens=[.20, .30], finished=.30, status="completed"),
]
rows, totals = request_report(receipts, start=0., end=1.)
for row in rows:
    print(row)
print(totals)
```
<!-- end example: edls-metric-receipts -->

| Request | Outcome | TTFT, s | TPOT, s | Individual ITLs, s |
|---|---|---:|---:|---|
| A | completed | 0.30 | 0.30 | 0.20, 0.40 |
| B | completed | 0.70 | undefined | none |
| C | rejected | undefined | undefined | none |
| D | cancelled after one token | 0.75 | undefined | none |
| E | completed | 0.20 | 0.10 | 0.10 |

All five requests were offered. Three completed, one was rejected and one was
cancelled. Seven tokens became available, including D's partial output; six
belong to completed requests. Over the declared one-second interval these
give 7 emitted tokens/s and 3 completed requests/s. A report that counts only
completed-request tokens would give 6 tokens/s and must name that convention.
Do not divide by the sum of per-request latencies: concurrent requests overlap.

Among completed requests with at least two tokens, mean request TPOT is
`(0.30 + 0.10) / 2 = 0.20 s`. Pooling their three ITLs gives
`(0.20 + 0.40 + 0.10) / 3 ≈ 0.2333 s`. Both calculations are valid for their
named populations; they answer different questions because the second weights
a request by its number of intervals. B's undefined TPOT is not a zero.

If the assigned SLO is TTFT ≤ 0.40 s and request TPOT ≤ 0.35 s where defined,
A and E pass: goodput is 2 completed requests/s. If it instead requires every
ITL ≤ 0.35 s, A fails because of its 0.40 s pause; only E passes. State which
constraint defines goodput and retain the rejected/cancelled counts beside it.
This finite fixture does not estimate a production p99.

### What the course benchmark actually observes

Read the [complete original benchmark](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/edlang/test_benchmark.py)
together with the [server](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/edlang/server/api_server.py).
Both are preserved under the course's MIT license. The benchmark captures
`latency` when response headers arrive, before `await resp.json()` reads and
parses the body. Because this endpoint waits for generation to finish, that
value is a response-header delay after generation, not first-token latency.
To measure client completion time, take an additional timestamp after reading
the response body. To measure client TTFT/ITL, add a streaming protocol with
token events; assigning token timestamps to one completed JSON body cannot
reconstruct the missing events.

`hard_prefill` requests use 2000 prompt characters and a maximum of 20 output
tokens; `hard_decode` uses 50 characters and a maximum of 2000 output tokens.
The tokenizer determines the actual input length, and EOS determines actual
output length. Save both counts rather than treating character limits and
generation caps as measured token work. The load generator sleeps between
submissions; prompt creation and event-loop work can shift the achieved rate.
Record actual submission times, use the same seeded prompt/arrival manifest
for every policy, and distinguish target RPS from achieved submission RPS.

The original statistics combine successful and failed request latencies in
one percentile list. Keep outcome-specific latency distributions and overall
outcome counts. Failed requests must not silently disappear, but a fast error
response must not improve a graph labeled successful-generation latency.
The finite experiment denominator includes request submission and drain time;
a steady-state window needs its own explicit boundaries.

### Three policies to compare on the same offered requests

The homework's Part 3 requires:

<!-- BEGIN SOURCE: EDLS three policies -->
> Your task is to iteratively experiment and develop better strategies to optimize this function. Design and implement at least **THREE** different policies for deciding the prefill batch size, and benchmark their performance.
<!-- END SOURCE: EDLS three policies -->

*Efficient DL Systems, Week 8 homework, Part 3,
[original assignment](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/homework_week8.ipynb),
commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`, MIT.*

The provided `_decide_prefill_batch_size` returns zero while any request is
active and one otherwise. Use that exact single-admission policy as a baseline.
A second policy can fill every available request slot before decoding. It
admits waiting work earlier, but a full new prompt makes existing decoders
wait for that prefill pass. A third can defer admission while there is active
decode work, then force admission when the oldest waiting request exceeds an
explicit queue-wait threshold. That changes which requests wait, so both queue
time/TTFT and existing-request ITLs must appear in the comparison.

All three policies must respect the available request slots and memory budget.
These are admission decisions for the separate prefill/decode calls above.
The engine does not expose partial-prompt state: a token-budgeted, chunked
prefill policy needs an additional interface and tests. Do not call a batch
size limit a prefill-token budget. The existing
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|scheduling chapter]]
explains that extension. The following section runs a bounded neural CPU
comparison; the assigned receipts and equal-cost trace are not its measurements.

At the API boundary, retain one request ID from submission to completion.
The supplied server generates an ID in the API process but the scheduler
allocates a second counter instead of accepting that ID. Their agreement
depends on the same successful arrival sequence. Propagate the original ID
explicitly before adding rejection, retries or cancellation. Also remove
resolved futures, handle cancelled futures before calling `set_result`, and
bound readiness waits if the worker fails. The included code has no such
end-to-end failure handling; passing its local engine assertions does not
establish a reliable asynchronous service.
<!-- END REVIEW: EDLS metric receipts 2026-10-26 -->

<!-- BEGIN REVIEW: EDLS neural admission 2026-10-26 -->
## Run three admission policies around the neural engine

The three-policy assignment above can now be run against the same tiny Qwen3,
rather than only the equal-cost state machine. The scheduler below accepts
the caller's request ID, keeps it through every event, and records the time
at which each synchronous engine call makes an output token available.

### Admission changes who waits for prefill

Suppose A is active, one of two request slots is free, and the remaining KV
reservation budget is six positions. B has a three-token prompt and a
three-token generation budget, so its maximum cache reservation is five
positions. B has waited 5 ms; the configured wait threshold is 20 ms.

| Decision state | single | refill | wait_bound |
|---|---:|---:|---:|
| A active; B age 5 ms; one free slot and six free KV positions | admit 0 | admit B | admit 0 |
| Same state; B age 20 ms | admit 0 | admit B | admit B |
| No free request slot | admit 0 | admit 0 | admit 0 |
| One free slot, but only four free KV positions | admit 0 | admit 0 | admit 0 |

`single` retains the course baseline: prefill one request only when no request
is active. `refill` fills available slots from the FIFO queue before decode.
`wait_bound` defers that prefill while decode is active until the oldest
waiting request reaches the threshold. The threshold is an admission trigger,
not a queue-latency guarantee: a running engine call, occupied slots or memory
pressure can delay admission further.

The reservation covers the largest cache the request can reach under its
output and context limits. Early EOS releases that reservation. The counter
does not measure physical peak memory: the engine also allocates a padded
temporary batch and copies compact caches. Those allocations require separate
measurement before turning this logical bound into a memory-capacity claim.

### Complete scheduler and terminal-event handling

Save the following module beside `edls_kv_reference.py`, or download
[the complete scheduler](/examples/edls_admission_reference.py) and
[the neural core](/examples/edls_kv_reference.py) into the same directory.
The repository file is `06 Практика/Code/edls_admission_reference.py`.

A step can contain a prefill call followed by a decode call. Requests that
finish in prefill are removed before decode, including a one-token budget.
Unfinished new requests join that decode call. Arrivals are consumed at the
next step boundary; no engine call is interrupted to admit a request.

<!-- BEGIN SOLUTION: EDLS admission module -->
```python
"""EDLS Week 8 teaching scheduler around the separate KVGreedyEngine.

Adapted from the MIT course's scheduler/metric contract, commit
e632aa89ca9e6638d52e1b686095e7442faffbb0, Copyright (c) 2021 course authors.
See /licenses/edls-e632aa89-MIT.txt. Single consumer; no HTTP/paging/chunking.
The KV budget counts reserved logical positions, not peak physical bytes.
"""
from collections import deque
from math import isfinite
from time import perf_counter

from edls_kv_reference import Request


POLICIES = ('single', 'refill', 'wait_bound')


class AdmissionScheduler:
    def __init__(self, engine, policy='single', slots=2, kv_positions=256,
                 max_waiting=100, max_wait_s=.02, clock=perf_counter):
        if policy not in POLICIES:
            raise ValueError('unknown admission policy')
        if any(not isinstance(n, int) or n < 1 for n in [slots, kv_positions, max_waiting]):
            raise ValueError('positive integer capacities required')
        if not isfinite(max_wait_s) or max_wait_s < 0:
            raise ValueError('finite nonnegative wait bound required')
        self.engine, self.policy, self.clock = engine, policy, clock
        self.slots, self.kv_positions = slots, kv_positions
        self.max_waiting, self.max_wait_s = max_waiting, max_wait_s
        self.waiting, self.active = deque(), []
        self.requests, self.receipts, self.reservations = {}, {}, {}
        self.phases = []

    @property
    def has_work(self):
        return bool(self.waiting or self.active)

    def _reservation(self, request):
        return min(self.engine.max_context,
                   len(request.prompt_ids) + request.max_new_tokens - 1)

    def submit(self, request, submitted=None, scheduled=None):
        now = self.clock()
        submitted = now if submitted is None else submitted
        scheduled = submitted if scheduled is None else scheduled
        if not all(isfinite(t) for t in [now, submitted, scheduled]) or not scheduled <= submitted <= now:
            raise ValueError('scheduled <= submitted <= consumer time required')
        if request.request_id in self.requests:
            raise ValueError('duplicate external request ID')
        self.engine._validate([request])
        if request.is_finished or request.generated_tokens or request.past_key_values is not None:
            raise ValueError('fresh request state required')
        rid = request.request_id
        self.requests[rid] = request
        self.receipts[rid] = dict(id=rid, scheduled=scheduled, submitted=submitted,
                                 observed=now, admitted=None, finished=None,
                                 tokens=[], output_ids=[], status='waiting', reason=None)
        if request.max_new_tokens == 0:
            request.stop_reason = 'length'
            self._finish(request, 'completed', now, 'length')
        elif self._reservation(request) > self.kv_positions or len(self.waiting) >= self.max_waiting:
            self._finish(request, 'rejected', now, 'capacity')
        else:
            self.waiting.append(request)
        return rid

    def _finish(self, request, status, now, reason):
        receipt = self.receipts[request.request_id]
        if receipt['finished'] is not None:
            return
        receipt.update(status=status, finished=now, reason=reason)
        request.stop_reason = reason
        request.past_key_values = None
        self.reservations.pop(request.request_id, None)

    def cancel(self, request_id):
        request = self.requests[request_id]
        if self.receipts[request_id]['finished'] is not None:
            return False
        self._finish(request, 'cancelled', self.clock(), 'cancelled')
        self.waiting = deque(r for r in self.waiting if r is not request)
        self.active = [r for r in self.active if r is not request]
        return True

    def _decide_prefill_batch_size(self):
        free_slots = self.slots - len(self.active)
        if not self.waiting or free_slots <= 0:
            return 0
        if self.policy == 'single' and self.active:
            return 0
        age = self.clock() - self.receipts[self.waiting[0].request_id]['submitted']
        if self.policy == 'wait_bound' and self.active and age < self.max_wait_s:
            return 0
        limit = 1 if self.policy == 'single' else free_slots
        free_positions = self.kv_positions - sum(self.reservations.values())
        count = 0
        for request in self.waiting:
            cost = self._reservation(request)
            if count == limit or cost > free_positions:
                break  # FIFO: do not bypass a large head request
            count += 1
            free_positions -= cost
        return count

    def _phase(self, stage, batch):
        if not batch:
            return
        start = self.clock()
        event = dict(stage=stage, ids=[r.request_id for r in batch], start=start,
                     reserved_positions=sum(self.reservations.values()))
        try:
            result = getattr(self.engine, stage)(batch)
            expected = {r.request_id for r in batch}
            if (set(result.request_ids) != expected or len(result.request_ids) != len(expected)
                    or not len(result.request_ids) == len(result.new_tokens) == len(result.finished)
                    or any(len(tokens) != 1 for tokens in result.new_tokens)):
                raise ValueError('one token per participating request required')
            ended = self.clock()
            for rid, tokens in zip(result.request_ids, result.new_tokens):
                self.receipts[rid]['tokens'].append(ended)
                self.receipts[rid]['output_ids'].extend(tokens)
            for request in batch:
                if request.is_finished:
                    self._finish(request, 'completed', ended, request.stop_reason)
            event.update(end=ended, emitted=len(result.request_ids), error=None)
        except Exception as error:
            ended = self.clock()
            for request in batch:
                # Internal candidate tokens from an incomplete call were not published.
                request.generated_tokens[:] = self.receipts[request.request_id]['output_ids']
                self._finish(request, 'error', ended, type(error).__name__)
            event.update(end=ended, emitted=0, error=type(error).__name__)
        self.phases.append(event)
        self.active = [r for r in self.active if not r.is_finished]

    def step(self):
        count = self._decide_prefill_batch_size()
        batch = [self.waiting.popleft() for _ in range(count)]
        now = self.clock()
        for request in batch:
            self.receipts[request.request_id].update(admitted=now, status='active')
            self.reservations[request.request_id] = self._reservation(request)
        self.active.extend(batch)
        self._phase('prefill', batch)
        # Prefill-finished requests were removed; new unfinished requests can decode.
        self._phase('decode', self.active.copy())
```
<!-- END SOLUTION: EDLS admission module -->

Cancellation is processed at a step boundary. It removes the request from
the waiting or active set, marks its receipt once, releases its reservation
and drops its cache reference. Cancelling an already finished request does
not change its outcome or produce another token.

If an engine call raises, all participating requests receive an error outcome.
Internal candidate tokens from that incomplete call are not published as
successful output. Their caches are discarded. The source request ID remains
the receipt key; no second counter has to stay synchronized with the caller.
This is a single-consumer reference, not a thread-safe HTTP service.

### Run the fixed-workload CPU experiment

```bash
uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 \
  python publishing/tools/check_edls_admission_reference.py --experiment
```

Without `--experiment`, the command runs deterministic correctness checks.
The experiment additionally checks neural EOS, context limits, zero output
and cancellation, then runs 54 measured cases: two workloads, three slot
limits, three policies and three repeats. The policy order is shuffled with
a recorded seed; the model weights and request manifest stay fixed.

Each workload has twelve requests at planned 2 ms intervals. The prefill-heavy
one uses prompts of 24/32/40 token IDs and outputs of one/two tokens, for
18 emitted tokens per run. The decode-heavy one uses prompts of two/three/four
IDs and output budgets of 10/13/16, for 156 emitted tokens. These are explicit
token-ID fixtures, not the original benchmark's character-based prompts or
a language-quality test.

EOS stopping is disabled in these timed fixtures so every policy receives the
same output work. First/second-token EOS is checked separately before timing;
the exact token totals do not assume that arbitrary checkpoints ignore EOS.

A separate standard-library process produces arrivals. Each receipt retains
`scheduled` (the target offset), `submitted` (the producer's actual timestamp),
`observed` (when the consumer registers it), admission, token events and
termination. Thus producer lateness and consumer/ingress delay remain visible
instead of being removed from TTFT. The local processes use the same
[monotonic performance clock](https://docs.python.org/3.14/library/time.html#time.perf_counter);
this would not justify subtracting unrelated hosts' clock readings.

### Recorded CPU trace and finite-run results

![[02 Areas/ML & DL/06 Практика/Results/edls-admission-2026-10-26/phase-timeline.png]]

*Bookvar CPU measurement, 26 October 2026: decode-heavy, two slots, repeat 0.
Hatched bars are prefill, dotted bars are decode; triangles mark actual producer
submissions. Each bar covers a synchronous engine call, including KV packing/
copying, and wrapper validation.*

The checked run used Darwin arm64, Python 3.14.3, PyTorch 2.14.1,
Transformers 4.57.1, CPU FP32, one PyTorch thread and 22,720 random parameters.
Warmup and independent uncached-oracle checks were outside the measured
windows. Every output in all 648 request instances matched that oracle.

The following values are median emitted tokens/s over three repeats,
rounded to whole tokens/s. The denominator is the whole finite run from its
clock origin until the producer and requests have drained.

| Workload | Policy | 1 slot | 2 slots | 4 slots |
|---|---|---:|---:|---:|
| prefill-heavy | single | 663 | 603 | 685 |
| prefill-heavy | refill | 626 | 706 | 698 |
| prefill-heavy | wait_bound | 604 | 572 | 696 |
| decode-heavy | single | 787 | 887 | 731 |
| decode-heavy | refill | 739 | 1222 | 2295 |
| decode-heavy | wait_bound | 876 | 1466 | 2068 |

`single` uses only one active request even when the configured slot limit is
larger. Differences between its columns are run variation, not a scaling
effect. The prefill-heavy workload emits only 18 tokens over an arrival span
of 22 ms, so its finite-run rate also depends strongly on the offered pace
and drain overhead; it is not a measurement of maximum model throughput.

The largest observed producer lateness was 5.00 ms despite the 2 ms target
spacing. Actual timestamps are saved for every run. Three repeats on a shared
CPU, with different realized arrival jitter, do not establish a policy ranking
or a production p99. Inspect TTFT, individual ITLs, phase durations and arrival
delays together before attributing a difference to admission.

The [complete recorded JSON](/examples/edls-admission-cpu-2026-10-26.json)
includes all manifests, receipts, phases, runtime settings and source hashes.
Re-running should preserve request IDs, budgets and greedy outputs; wall-clock
numbers will differ. Plot reproduction uses
`publishing/tools/plot_edls_admission.py` with Matplotlib 3.10.7.
An asynchronous streaming API, tokenizer/checkpoint workload, physical-memory
measurement and GPU/SLO-capacity experiment remain separate work.

<!-- BEGIN REVIEW: HTTP followup 2026-10-26 -->
Рабочий HTTP/streaming слой и его жизненный цикл вынесены в
[[02 Areas/ML & DL/06 Практика/14a HTTP streaming и жизненный цикл запроса|продолжение 14a]]:
доставка токенов, ID, отмена, timeout, readiness и завершение при ошибке.
<!-- END REVIEW: HTTP followup 2026-10-26 -->
<!-- END REVIEW: EDLS neural admission 2026-10-26 -->

<!-- BEGIN REVIEW: inference practice 2026-10-26 -->
## Serving experiment: cost, KV capacity, scheduling and release constraints

A server can load its weights successfully and still run out of memory when
requests arrive. It can also complete many tokens per second while making
new users wait behind long generations. Keep three separate records in your
experiment: bytes allocated to live requests, work scheduled in each iteration,
and timestamps observed by each request.

The worked investigation below follows the four parts of
[Harvard Lab V2-10, The Inference Economy](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/labs/vol2/lab_10_inference.py)
and its [Inference at Scale slides](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/slides/vol2/10_inference/10_inference.tex).
Source passages remain in English. The numerical fixtures are stated below;
they are teaching inputs, not measurements of a deployed model. The original
Marimo lab contains heuristic latency and scheduling scores, so its computed
“winner” cannot substitute for a load test.

### A. Recurring cost and the deployment horizon

Assume a one-time training/setup cost of USD 2 million, one million daily users,
50 requests per user per day, and an assigned serving cost of USD 0.001 per
request. Traffic is 50 million requests per day. At that rate, serving costs
USD 50,000 per day and reaches the setup cost after 40 days. Over a 365-day year
it costs USD 18.25 million. A sustained 10% reduction in cost per request saves
USD 1.825 million over that year, if demand and quality remain unchanged.

The slide's rounded USD 18 million example uses the same reasoning. This is a
conditional comparison, not a law that serving always costs more than training:
a rarely used model, short deployment horizon or expensive recurring training
changes the result. The USD 0.001 is an assumed cost, not a current API price.

<!-- BEGIN SOURCE: Harvard inference cost equations -->
```
C_total(t) = C_setup + C_event * QPS * seconds * t
C_serving_day = QPS * 86400 * C_event * (1 - optimization_pct)
t_crossover = C_setup / C_serving_day

TTFT proxy ~= f(prompt/input amount)
TPOT proxy ~= model_bytes / memory_bandwidth
decode_loop ~= output_steps * TPOT
```
<!-- END SOURCE: Harvard inference cost equations -->

*Original Math Peek equations, Harvard Lab V2-10, Part A; CC BY-NC-SA 4.0.
In the daily-cost equation, `optimization_pct` must mean a fraction such as
0.10, not the slider's integer 10. In the first equation, `seconds * t` must
be the duration in seconds. The phase equations are explicitly proxies.*

For the calculation, keep duration and rate units explicit:

$$
\begin{aligned}
C_{\text{day}}&=86400\,\lambda c(1-r),\\
t_{\text{cross,days}}&=\frac{C_0}{C_{\text{day}}}.
\end{aligned}
$$

Here $\lambda$ is requests per second, $c$ is cost per request and
$r\in[0,1]$ is the cost reduction fraction. With no traffic or zero serving
cost there is no finite crossover for positive $C_0$.

<!-- example: inference-practice-cost -->
```python
from fractions import Fraction

setup = 2_000_000
requests_day = 1_000_000 * 50
cost_request = Fraction(1, 1000)
reduction = Fraction(1, 10)
daily = requests_day * cost_request
optimized_daily = daily * (1 - reduction)
print("baseline/day", daily)
print("crossover/days", setup / daily)
print("optimized crossover/days", round(float(setup / optimized_daily), 2))
print("annual saving", 365 * (daily - optimized_daily))
```

The optimization delays crossover to 44.44 days because each day now costs
USD 45,000. It does not change the original setup bill. If a request becomes
longer or is routed to a larger model, re-estimate $c$ rather than holding it
constant while changing the workload.

### B. Fit live KV beside the weights

<!-- BEGIN SOURCE: Harvard inference state explanation -->
For transformer serving, the state term is KV cache. For device tracks, the same
slot is runtime state such as activation buffers, sensor windows, or local cache.
The lesson is identical: **weights are only one memory term**.
<!-- END SOURCE: Harvard inference state explanation -->

*Original explanation, Harvard Lab V2-10, Part B; CC BY-NC-SA 4.0.*

Use the slide's GQA exercise: 80 layers, eight KV heads, head dimension 128,
FP16 K/V, context length 4096. There are separate key and value tensors, so
one sequence needs

$$
\begin{aligned}
M_{KV,1}&=2\cdot80\cdot8\cdot128\cdot2\cdot4096\\
&=1\,342\,177\,280\ \text{bytes}.
\end{aligned}
$$

This is 1.25 GiB or about 1.34 decimal GB. It uses eight **KV** heads,
not all query heads. Assume exactly 35 decimal GB of packed weights and
80 decimal GB available on one serving device. The 35 GB omits quantization
metadata in this exercise; an actual deployment must include that metadata.
Reserve another 8 GB for non-KV runtime allocations and headroom. This reserve
is an explicit fixture, not a universal engine constant.

| Batch | Weights + KV, decimal GB | Including 8 GB reserve | Fits the 80 GB budget? |
|---:|---:|---:|---|
| 16 | 56.47 | 64.47 | yes, in this accounting |
| 32 | 77.95 | 85.95 | no |

The second row explains why “weights plus KV barely fit” is not enough.
The allocator also needs space for temporary tensors, graph pools and other
engine state. Under this fixed-reserve model, the largest batch at 4096 tokens
is 27; at 8192 tokens it is 13. These are memory bounds, not recommended batch
sizes or measured capacities.

<!-- example: inference-practice-memory -->
```python
weights = 35_000_000_000
device = 80_000_000_000
reserve = 8_000_000_000
kv_token = 2 * 80 * 8 * 128 * 2
for context in [4096, 8192]:
    per_request = context * kv_token
    max_batch = max(0, (device - weights - reserve) // per_request)
    print(context, per_request, max_batch)
for batch in [16, 32]:
    total = weights + batch * 4096 * kv_token + reserve
    print(batch, f"{total / 1e9:.2f}", total <= device)
```

When you connect this budget to an engine, record the allocated blocks per
request and release them on EOS, cancellation and errors. Reserve room for
the next decode write before scheduling it. For variable contexts, sum actual
per-request cache lengths; using a historical workload mean for every live request
can hide a group of long requests that exhausts memory. Prefix sharing changes
physical ownership, so shared blocks need reference counts or equivalent
ownership tracking. Do not free a shared prefix when only one user finishes.

The full MHA/GQA and paging explanation is in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|the KV-cache chapter]].
The memory bound above assumes one-device placement; adding the nominal memory
of TP or PP ranks does not prove every rank fits its own shard and buffers.

### C. Trace a scheduler before timing a model

In a closed cohort, unfinished requests continue together until the cohort
has completed; new requests cannot fill its finished slots. An iteration-level
scheduler can remove a finished request and admit another at the next iteration.
This is the scheduling distinction introduced by
[Orca](https://www.usenix.org/conference/osdi22/presentation/yu).
The implementation still has to budget new prefill work, decode work and memory.

The two original Hugging Face diagrams below show why changing the active
request set also changes the input tensor and attention mask. In the first,
padding remains in a rectangular batch. In the second, current decode tokens
and chunks of new prompts share an iteration with a new-token budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/hf-dynamic-batching.png]]

*Rémi Ouazan Reboul, Arthur Zucker and Luc Georges, Hugging Face,
[Continuous batching](https://huggingface.co/blog/continuous_batching#continuous-batching).
Complete original dynamic-batching PNG, CC BY-NC-SA 4.0; no crop or relabeling.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/hf-continuous-batching.png]]

*Same authors and source: complete original continuous-batching PNG,
CC BY-NC-SA 4.0. White token boxes denote new work; blue tokens are cached;
green entries mark allowed attention. The diagram is schematic and does not
define the timing of the executable trace below.*

Start with four requests. The arrival column uses integer ticks; each tick
can emit one token for every active request. All prompts are assumed already
prefilled when eligible for admission, so prefill time is omitted here.

| Request | Arrival tick | Tokens to emit |
|---|---:|---:|
| A | 0 | 2 |
| B | 0 | 5 |
| C | 1 | 1 |
| D | 2 | 2 |

With two slots, both policies run A and B at ticks 0 and 1. A finishes at
time 2. The iteration-level policy admits C at tick 2, then D at tick 3.
B and D finish at time 5. The closed-cohort policy keeps A's slot unavailable
until B finishes at time 5; C and D then form the next cohort, ending at time 7.
Completed tokens can be returned immediately in either policy; the restriction
here concerns reuse of the slot, not an obligation to withhold completed output.

<!-- example: inference-practice-scheduler -->
```python
from collections import deque

def schedule(jobs, slots, refill):
    if not isinstance(slots, int) or slots < 1 or len({j[0] for j in jobs}) != len(jobs):
        raise ValueError("positive slots and unique request IDs required")
    if any(not isinstance(a, int) or not isinstance(n, int) or a < 0 or n < 1
           for _, a, n in jobs):
        raise ValueError("integer nonnegative arrivals and positive lengths required")
    waiting = deque(sorted(jobs, key=lambda job: job[1]))
    remaining = {name: length for name, _, length in jobs}
    emitted = {name: [] for name, _, _ in jobs}
    active, trace, tick = [], [], 0
    while waiting or active:
        if refill:
            active = [name for name in active if remaining[name] > 0]
        elif active and all(remaining[name] == 0 for name in active):
            active = []
        if not active and waiting:
            tick = max(tick, waiting[0][1])
        if refill or not active:
            while waiting and waiting[0][1] <= tick and len(active) < slots:
                name, _, _ = waiting.popleft()
                active.append(name)
        if not active:
            continue
        useful = [name for name in active if remaining[name] > 0]
        if not useful:  # the final cohort has just completed
            continue
        trace.append((tick, tuple(useful)))
        for name in useful:
            remaining[name] -= 1
            emitted[name].append(tick + 1)
        tick += 1
    return trace, emitted

jobs = [("A", 0, 2), ("B", 0, 5), ("C", 1, 1), ("D", 2, 2)]
for refill in [False, True]:
    trace, emitted = schedule(jobs, slots=2, refill=refill)
    print("refill", refill, "ticks", len(trace))
    print({name: times[-1] for name, times in emitted.items()})
```

| Tick | Closed cohort: useful rows | Iteration-level: useful rows |
|---:|---|---|
| 0 | A, B | A, B |
| 1 | A, B | A, B |
| 2 | B | B, C |
| 3 | B | B, D |
| 4 | B | B, D |
| 5 | C, D | completed |
| 6 | D | completed |

Both emit exactly ten useful tokens. Fourteen two-slot opportunities in the
closed-cohort run versus ten in the refilling run give 71.43% versus 100%
logical slot occupancy for this fixture. These percentages are not GPU
utilization, MFU or MBU. The assumed tick costs are equal; neither prefill nor
scheduler overhead is charged. A real trace can have different results.

For a single request with no queued work, refilling cannot create more work.
For a long new prompt, admitting it can increase the next iteration's duration.
Extend the trace with a prefill-token budget before comparing end-to-end
performance. The complete mechanism, including chunking, is in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|the scheduling chapter]].

### D. Select a feasible configuration, then compare cost

<!-- BEGIN SOURCE: Harvard inference feasibility model -->
```
feasible = memory_ok and slo_ok and quality_ok and cost_ok
QPS_per_replica = max_batch * base_qps * batching_multiplier
replicas_needed = ceil(target_qps / QPS_per_replica)
daily_cost = replicas * devices_per_replica * cost_per_device_hour * 24
```

Source model: `serving_plan()` combines `state_capacity()` with track-specific
QPS slots and recurring cost. The notebook adds local release-review indicators:
SLO margin, quality guardrail, routing risk, and binding constraint.
<!-- END SOURCE: Harvard inference feasibility model -->

*Original Math Peek, Harvard Lab V2-10, Part D; CC BY-NC-SA 4.0.*

In the source, `batching_multiplier` is an assigned coefficient. The memory
bound `max_batch` is not a measured service rate. In your implementation,
replace the QPS proxy with a load-test result at the intended context/output
distribution and SLO. Quality also needs an evaluation; choosing INT4 does
not by itself calculate a quality loss.

For a small decision fixture, require 120 requests/s with 25% capacity headroom,
p99 no greater than 100 ms, quality loss no greater than 1 percentage point,
and daily cost no greater than USD 250. Define headroom relative to demand:
capacity must be at least $1.25\cdot120=150$ requests/s. This is a planning
assumption, not a universal serving target.

<!-- example: inference-practice-feasibility -->
```python
from math import ceil

# Assigned fixtures, not benchmark or pricing data.
candidates = [
    dict(name="A", rate=80, p99=90, loss=0.1, memory=70, devices=2),
    dict(name="B", rate=140, p99=120, loss=0.1, memory=90, devices=1),
    dict(name="C", rate=75, p99=80, loss=1.4, memory=65, devices=1),
    dict(name="D", rate=70, p99=95, loss=0.4, memory=65, devices=1),
]
for row in candidates:
    replicas = ceil(150 / row["rate"])
    daily_cost = replicas * row["devices"] * 3 * 24
    failed = [name for name, ok in [
        ("memory", row["memory"] <= 80),
        ("p99", row["p99"] <= 100),
        ("quality", row["loss"] <= 1),
        ("cost", daily_cost <= 250),
    ] if not ok]
    print(row["name"], replicas, daily_cost, failed or ["pass"])
```

| Candidate | Replicas | Cost/day, assigned USD | Failed constraints |
|---|---:|---:|---|
| A | 2 | 288 | cost |
| B | 2 | 144 | memory, p99 |
| C | 2 | 144 | quality |
| D | 3 | 216 | none |

The cheapest candidates are infeasible. D passes the assigned constraints
despite using more replicas than A. If no row passes, report that no feasible
configuration was found; do not select a winner by allowing a large throughput
score to compensate for an SLO or quality failure. The source lab's scheduling
score allows such a trade-off, so it is not used as this acceptance rule.

The table also assumes each replica has sufficient capacity at the listed
p99. It does not predict p99 after changing the arrival rate. Measure that
curve. In the slide's queue example, eight servers at 5 requests/s provide
40 requests/s, so 100 requests/s gives $\rho=2.5$, not 0.25. Queueing conclusions
drawn from the latter number do not apply.

### E. Include time-to-ready in a scale-out experiment

A replica becomes useful only after resource allocation, weight loading and
warmup complete. A successful process start does not establish that requests
can be served. Record the time of a successful warm inference, together with
the distribution of cold starts, rather than using a fixed model-size lookup.
The Harvard cold-start slide separates these stages; its minute counts are
illustrative and are not reused as a measurement here.

Suppose demand rises to 120 requests/s, existing capacity is 100 requests/s,
and the added capacity takes an assigned 300 seconds to become ready. With
no rejection, expiry or change in service rate, the fluid backlog grows by
$20\cdot300=6000$ requests. When a second 100 requests/s replica is ready,
the 200 requests/s total capacity drains the backlog at 80 requests/s,
taking another 75 seconds. The queue does not disappear at the instant the
new replica becomes ready.

<!-- example: inference-practice-cold-start -->
```python
demand, old_capacity, added_capacity = 120, 100, 100
ready_after = 300
backlog = max(0, demand - old_capacity) * ready_after
drain_rate = old_capacity + added_capacity - demand
assert drain_rate > 0
print("queued requests", backlog)
print("seconds to drain after ready", backlog / drain_rate)
```

This is a deterministic fluid calculation, not a p99 estimator. With a bounded
queue, rejections or expirations replace part of the backlog. With priorities,
the waiting-time distributions differ between classes. A load-test report must
count all offered requests, including rejected, cancelled and expired ones;
calculating latency only for completed requests can conceal overload.

### Connect the calculations to the EDLS engine

Keep the original EDLS seminar and homework linked at the top as the neural
implementation path. The scheduler above is a CPU reference state machine;
it does not execute a model or supply a GPU latency estimate.

For the implementation, retain a request ID, token history, KV ownership,
arrival time, token timestamps and termination reason. A scheduler decision
must respect both the token budget and the free-block budget. Prefill supplies
the first output token; an answer with $N$ output tokens then needs $N-1$
ordinary decode passes. Check greedy generation against an uncached reference
at batch one before sharing an iteration between requests.

The correctness tests must cover EOS, cancellation, context limits, a request
with one output token, admission at an iteration boundary, padding/attention
masks and independent cache ownership. After a request is removed, no later
token may be emitted for it. Repeat equivalent requests in different batch
orders to detect accidental exchange of histories. Shared-prefix tests must
also verify that one user's completion does not invalidate another user's KV.

For the performance run, record model and tokenizer revisions, weight/KV formats,
engine commit, hardware and parallelism, prompt/output lengths, arrival process,
warmup and timed interval. Compare a fixed set of offered requests. Report
TTFT, per-request TPOT where defined, ITL distributions, throughput and SLO
goodput alongside peak allocated KV and failures. Keep the arithmetic fixtures
above separate from the measured report.

Original source and license: Vijay Janapa Reddi and Harvard ML Systems Book
contributors, pinned commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The source excerpts and their worked adaptation retain attribution here;
the original lab remains available as
[[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_10_inference|the complete imported source]].
<!-- END REVIEW: inference practice 2026-10-26 -->
