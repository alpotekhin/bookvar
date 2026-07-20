---
title: Эффективный attention
type: research-line
status: active
started: 2019
last_updated: 2026-07-20
last_verified: 2026-07-20
key_concepts: [FlashAttention, MQA, GQA, MLA, KV-Cache]
key_models: [LLaMA 2, Mistral, DeepSeek-V2, DeepSeek-V3]
primary_sources:
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2405.04434
---

# Эффективный attention

## Тезис

«Эффективность attention» обозначает минимум три независимые оптимизации:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention-tiling.png]]

*Иллюстрация блочного вычисления attention: промежуточные блоки scores и
softmax вычисляются в SRAM, а накопленный выход пересчитывается при изменении
нормирующего знаменателя. Это оптимизация движения данных, а не изменение
множества связей. Источник изображения: Tri Dao,
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://crfm.stanford.edu/2023/07/17/flash2.html);
исходный алгоритм и доказательство exactness —
[FlashAttention, §3 и Algorithm 1](https://arxiv.org/abs/2205.14135).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-35-36/cs336-mha-gqa-mqa-mla.png]]

*Сопоставление MHA, GQA, MQA и MLA из Stanford CS336 показывает другую ось:
что именно сохраняется в KV-cache при декодировании. Колонки сравнивают
представления, а не задают хронологию: MQA опубликован раньше GQA, а MLA
использует отдельную латентную параметризацию. Источник: Stanford CS336,
[Lecture 10: Inference](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py),
рисунок `mla-schema.png`, основанный на Figure 3 отчёта
[DeepSeek-V2](https://arxiv.org/abs/2405.04434).*

- kernel сохраняет математический результат, но считает его эффективнее;
- KV-компрессия меняет представление ключей и значений;
- sparse/linear pattern меняет множество доступных взаимодействий.

## Сравнение

| Подход | Экономит | Меняет модель | Типичная цена |
|---|---|---:|---|
| [[02 Areas/ML & DL/01 Справочник/Inference/FlashAttention|FlashAttention]] | HBM I/O, activations | нет | сложный kernel |
| MQA | KV-cache | да | возможная потеря качества |
| [[02 Areas/ML & DL/01 Справочник/Attention/MHA, MQA и GQA|GQA]] | KV-cache | да | компромисс MHA/MQA |
| [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]] | KV-cache | да | сложные проекции и реализация |
| sliding window | FLOPs/память на длинном входе | да | ограниченная дальняя связь |

## Основная ошибка сравнения

Training throughput, prefill latency, decode latency и peak memory — разные
метрики. FlashAttention особенно важен для больших матриц prefill/training;
KV-cache compression — для memory-bound autoregressive decoding.

## Открытые вопросы

- Какие compression schemes лучше сохраняют качество после fine-tuning?
- Когда structured sparsity превосходит dense optimized kernels на реальном GPU?
- Можно ли выбирать attention pattern динамически для каждого токена?
- Как честно сравнивать kernels на разных длинах, batch size и hardware?

## Связанные страницы

[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA и GQA]] ·
[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA]] ·
[[02 Areas/ML & DL/Papers/Flash Attention|FlashAttention paper]] ·
[[02 Areas/ML & DL/Papers/DeepSeek-V2|DeepSeek-V2]]
