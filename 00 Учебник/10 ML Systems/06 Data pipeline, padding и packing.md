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

> Two components: what to read and how to read.

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

## Проверка throughput

Измеряют samples/s **и** non-padding tokens/s, padding ratio, queue wait,
CPU utilization, read bandwidth и H2D overlap. Рост batches/s может скрывать
меньше полезных tokens или изменение length distribution.

## Источники

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf)
- [EDLS week 2 dynamic-padding homework](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/homework/task2)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] →
