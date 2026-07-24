---
title: Data pipeline, padding и packing
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Data pipeline, padding и packing

GPU простаивает, если batch не готов. Data pipeline должен доставлять следующий
batch быстрее, чем модель обрабатывает текущий, и не тратить compute на
бессмысленный padding.

## Что хранить

> **Адаптация, не дословная цитата:** EDLS week 2, PDF p. 26,
> slide “Bottlenecks in data loading”. Pipeline разделяется на два связанных
> вопроса: что читать и как доставлять прочитанное к модели.

Raw files удобны для просмотра, но множество мелких файлов создаёт metadata и
network overhead. Для structured data подходят Arrow/Protobuf/msgpack или
шардированные containers. Детерминированную тяжёлую preprocessing выполняют
один раз. Для языка часто выгодно хранить уже токенизированные integer ids —
обязательно вместе с версией tokenizer и правилами EOS/BOS.

## Как загружать

Pipeline:

```text
storage → read shards → deserialize/decode → augment/tokenize
        → collate → pinned host batch → async H2D → model
```

Параллельные workers помогают, пока не упираются в storage bandwidth, CPU,
RAM или duplicated worker state. Prefetch перекрывает стадии; bounded queue
предотвращает бесконтрольный рост памяти. Для изображений decoder и тяжёлые
augmentations могут быть bottleneck; EDLS указывает Pillow-SIMD,
jpeg-turbo/nvJPEG, DALI и GPU augmentations как варианты, а не универсальный
рецепт.

## Dynamic padding

Если длины $l_i$, а batch padded до $L_{\max}$, доля полезных tokens:

$$\eta_{\text{pad}}=\frac{\sum_i l_i}{B L_{\max}},
\qquad
waste=1-\eta_{\text{pad}}.
$$

Для длин `[128, 140, 160, 1024]` полезность
$1452/(4\cdot1024)\approx35{,}4\%$: почти две трети token-level compute
потрачены на padding. Поэтому samples хранят без padding, а `collate_fn`
дополняет только до maximum текущего batch.

## Bucketing

Группировка близких длин уменьшает $L_{\max}$. Полная сортировка ухудшает
randomness, поэтому обычно shuffle выполняют между buckets и внутри них либо
сортируют ограниченное окно. Batch sampler может ограничивать не число samples,
а число tokens: $\sum_i l_i\le T_{\text{budget}}$.

## Packing

Packing помещает несколько коротких документов в один fixed-length sequence:

```text
[doc A][EOS][doc B][EOS][doc C][PAD]
```

Нужно явно определить границы:

- attention mask не должна пропускать информацию между независимыми examples,
  если objective этого не допускает;
- labels на boundary/EOS формируются согласно training objective;
- position ids либо продолжаются, либо сбрасываются согласованно с kernel;
- loss нормализуется по non-padding tokens, а не по числу packed sequences.

Для causal LM иногда допускают continuous token stream: тогда переход через EOS
является частью objective. Это и «изоляция документов mask» — разные режимы.

## Storage и DataLoader как система

Shards амортизируют открытие мелких файлов; Arrow/Parquet дают column projection;
`mmap` удобен для локального random read, но page faults и network filesystem
делают latency неровной. Compression уменьшает storage/network bytes ценой CPU
decode. Manifest хранит checksum, schema, records/tokens, preprocessing version
и tokenizer commit; tokenizer также фиксирует normalization, vocabulary,
special-token ids и BOS/EOS policy.

В `DataLoader` каждый worker имеет replica dataset и prefetch queue. Настройки
`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory` и batch
size меняют единый RAM/CPU/I/O budget. Примерный скрытый запас равен
`workers × prefetch × batch_bytes`. Seed выводят из global seed, epoch, rank и
worker id; distributed sampler получает `set_epoch(epoch)`. Иначе повторяемый
запуск может увидеть другой порядок shards и augmentations.

Variable-token sampler ограничивает $\sum_i l_i\le T_{budget}$, а не число
samples. Loss нормируют по valid tokens всего accumulation window. Это
стабилизирует compute, но число samples/step становится переменным.

## Packing без leakage

Обычная causal mask не даёт смотреть в будущее, но документ B всё ещё видит A.
Для независимых examples нужна block-diagonal causal mask или segment-aware
kernel, согласованные `position_ids` и labels `-100` на запрещённых переходах.
Synthetic-тест: изменение tokens A не должно менять logits B. Continuous-stream
objective, напротив, сознательно разрешает переход через EOS — режимы нельзя
смешивать.

## Каркас pipeline

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate(rows):
    ids = [row["input_ids"] for row in rows]
    x = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    mask = x.ne(pad_id)
    return {"input_ids": x, "attention_mask": mask,
            "labels": x.masked_fill(~mask, -100)}

loader = DataLoader(
    dataset, batch_sampler=token_budget_sampler, collate_fn=collate,
    num_workers=8, prefetch_factor=2, persistent_workers=True,
    pin_memory=True,
)
for cpu_batch in loader:
    batch = {k: v.cuda(non_blocking=True) for k, v in cpu_batch.items()}
    loss = model(**batch).loss
```

`pad_id` не следует молча приравнивать к EOS. Bounded queues создают
backpressure вместо роста RAM. Для поиска bottleneck логируют `next(loader)`,
queue depth, read/decode/collate/H2D, page faults, bytes/s и useful tokens/s;
затем сравнивают с synthetic tensors и по одному отключают decode,
augmentation, compression и remote read.

```text
длины [128,140,160,1024], Lmax=1024
128  ████░░░░░░░░░░░░░░░░░░░░░░░░
140  ████░░░░░░░░░░░░░░░░░░░░░░░░
160  █████░░░░░░░░░░░░░░░░░░░░░░░
1024 ████████████████████████████████
useful=1452, slots=4096, efficiency=35.4%
```

## Проверка throughput

Измеряют samples/s **и** non-padding tokens/s, padding ratio, queue wait,
CPU utilization, read bandwidth и H2D overlap. Рост batches/s может скрывать
меньше полезных tokens или изменение length distribution.

## Источники

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) — “Bottlenecks in data loading”, “Storage formats”, “Minimizing preprocessing time”, “Optimal sequence processing”; title locators used because incremental slides repeat in the PDF.
- [EDLS week 2 dynamic-padding homework](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/homework/task2)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] →
