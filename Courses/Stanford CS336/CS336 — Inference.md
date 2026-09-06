---
title: "CS336 — Inference"
type: source-note
status: legacy
source_only: true
course: "Stanford CS336"
last_updated: 2026-09-06
canonical_target: "[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/_index|Inference и оптимизация]]"
robots: noindex
search_exclude: true
---

# CS336 — Inference

> [!note] Историческая карточка источника
> Эта страница больше не является отдельной учебной главой. Материал Lecture 10
> перенесён в канонические главы ниже, где формулы, допущения и экспериментальные
> результаты проверены по pinned Stanford CS336 Spring 2026 и первичным работам.

**Курс:** [[Stanford CS336/_index|Stanford CS336]]

**Лекция:** [Stanford CS336 Spring 2026 — Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py)

**Pinned commit:** `8b59b50730766695c2ffedd1a79c50cd09b9eb91`

## Куда перенесён материал

1. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Prefill, decode, tensor arithmetic и roofline]].
2. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA и GQA]].
3. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA и сжатие KV-cache]].
4. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache и PagedAttention]].
5. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Continuous batching, chunked prefill и prefix caching]].
6. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines и границы runtime/kernel]].
7. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Квантизация]].
8. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank|Pruning и distillation]].
9. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Спекулятивное декодирование]].
10. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, goodput и SLO]].

## Почему старый конспект не сохранён как учебный текст

В прежней версии рядом с механизмами стояли числа без воспроизводимого
контекста: `10–100×` lifetime cost, `1–5%` peak utilization, универсальные
`2–4×` speedups, INT4 degradation `<1%` и утверждение о запуске 70B на
«consumer GPU». Там также были неверное число KV-голов Llama 2 70B, слишком
сильная формулировка об «идентичном» результате speculative sampling и линейная
схема, в которой architecture, allocator, scheduler, kernel и sampler выглядели
последовательными стадиями одного стека.

Эти утверждения не используются как доказательства. Там, где конкретные числа
нужны, канонические главы приводят model, dtype, batch, context, hardware,
runtime и первичный источник. Страница сохранена только как указатель, чтобы не
ломать старые Obsidian-ссылки и provenance курса.

## Первичные источники

- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180).
- Yu et al., [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf).
- Leviathan et al., [Speculative Decoding](https://arxiv.org/abs/2211.17192).
- Ainslie et al., [Grouped-Query Attention](https://aclanthology.org/2023.emnlp-main.298/).
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978).
- Muralidharan et al., [Minitron](https://arxiv.org/abs/2407.14679).
