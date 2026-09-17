---
title: Эффективный attention
type: research-line
status: active
started: 2019
last_updated: 2026-08-06
last_verified: 2026-08-06
key_concepts: [FlashAttention, MQA, GQA, MLA, KV-Cache]
key_models: [LLaMA 2, Mistral, DeepSeek-V2, DeepSeek-V3]
primary_sources:
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2405.04434
---

# Эффективный attention

## Три источника стоимости

На обучении и prefill строятся большие матрицы по всей последовательности;
стоимость растёт квадратично, а activations занимают память. На decode новый
токен имеет один query, но читает K/V всего префикса; операция часто ограничена
bandwidth, а KV-cache растёт с batch и контекстом. Поэтому можно не менять
математику, сжимать KV или менять набор связей — это разные исследования.

[Longformer (2020)](https://arxiv.org/abs/2004.05150) разреживает матрицу связей
и тем самым меняет функцию. [Multi-Query Attention
(2019)](https://arxiv.org/abs/1911.02150) сохраняет отдельные query-heads, но
делит одну пару K/V, уменьшая cache и чтение на decode. [Grouped-Query Attention
(2023)](https://arxiv.org/abs/2305.13245) предлагает промежуточное число групп и
способ конвертировать MHA-checkpoint коротким uptraining.

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
- sparse attention оставляет только выбранные связи между позициями;
- linear attention меняет способ вычисления весов и агрегирования истории.
  Например, kernelized attention может учитывать все предыдущие позиции,
  не сохраняя отдельную пару K/V для каждой из них. Линейная стоимость здесь
  не означает, что часть связей просто удалили.

Например, в [Linear Transformers, §3.3](https://arxiv.org/html/2006.16236#S3.SS3)
вклад всей причинной истории записывается через накопленные суммы
$S_t=\sum_{j\le t}\phi(k_j)v_j^\top$ и $z_t=\sum_{j\le t}\phi(k_j)$.
Выход равен $\phi(q_t)^\top S_t/[\phi(q_t)^\top z_t]$ при ненулевом знаменателе.
Здесь $q,k,v$ — запросы, ключи и значения, $\phi$ — выбранное признаковое
отображение ядра. Все прошлые позиции входят в суммы, но сходство
$\phi(q)^\top\phi(k)$ заменяет обычное softmax-взвешивание. Это линейная по
длине агрегация при фиксированной размерности признаков, не вычисление
исходного softmax с удалёнными рёбрами. У FlashAttention, напротив,
математическая функция сохраняется.

## FlashAttention: exact attention как I/O-задача

[FlashAttention (2022)](https://arxiv.org/abs/2205.14135) исходит из иерархии
памяти GPU. Наивная реализация материализует scores в HBM и несколько раз их
читает. Tiling переносит блоки Q, K и V в SRAM, а online softmax накапливает
нормированный результат без полной матрицы. Экономия получена из меньшего
движения данных, а не из аппроксимации.

[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) сократил
не-матричные операции и лучше разделил работу между thread blocks и warps.
Выигрыш зависит от causal mask, head dimension, длины и GPU; асимптотика
остаётся квадратичной, хотя предел памяти отодвигается.

## Латентное сжатие KV

[DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) предложил MLA: вместо
отдельных K/V heads кешируется низкоразмерное латентное представление, из
которого компоненты восстанавливаются проекциями. Это не просто новая
группировка heads. Преимущество в bytes/token нужно проверять вместе с ценой
проекций, RoPE-компонентом и поддержкой конкретного inference engine.

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

## Состояние доказательств

FlashAttention — наиболее строго установленный результат: алгоритм exact, а
ускорение воспроизводится при подходящих shapes. Для MQA/GQA хорошо подтверждено
уменьшение KV-cache; влияние на качество зависит от групп и обучения. MLA
показал сильный end-to-end рецепт в DeepSeek, но контролируемых сравнений при
одинаковых данных меньше. Sparse и linear attention имеют лучшую асимптотику,
однако optimized dense kernels нередко выигрывают на практических длинах.
Поэтому FLOPs нужно дополнять measured latency, memory и quality на одинаковом
hardware и workload.

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
