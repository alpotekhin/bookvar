---
title: Измерить KV-cache и quantization
type: practice
status: canonical
last_updated: 2026-08-06
---

# Измерить KV-cache и quantization

Цель работы — разделить три разных источника экономии памяти: представление
весов, число KV-heads и формат самого KV-cache. Итогом должна быть модель
памяти, которая предсказывает измерение, а не таблица случайных показаний GPU.

## Подготовка

Выберите одну модель, доступную одновременно в исходной или FP16/BF16 версии и
как минимум в двух форматах llama.cpp. Зафиксируйте model revision, tokenizer,
commit llama.cpp, compiler flags, CPU/GPU и backend.

Теория:

- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]];
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention]];
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей]].

## Часть 1. Предсказать память KV-cache

Для batch size $B$ и длины $T$:

$$
M_{KV}=2BLTn_{kv}d_hs,
$$

где $L$ — число слоёв, $n_{kv}$ — число KV-heads, $d_h$ — размер головы, $s$ —
байт на элемент. Посчитайте теоретическую память для MHA, GQA и MQA при
одинаковых $L,T,d_h$. Затем сделайте sweep $T$ и $B$, измеряя cache отдельно от
весов и runtime buffers.

Постройте график «предсказанные байты — измеренные байты». Остаток объясните
через block allocation, alignment, metadata и зарезервированную память.

## Часть 2. Сравнить форматы весов

Для FP16/BF16 baseline и двух quantized formats измерьте:

- размер файла и resident RAM/VRAM после загрузки;
- prompt processing, tokens/s и TTFT;
- generation tokens/s и TPOT;
- peak memory для одинаковых prompt/generated lengths;
- perplexity или фиксированный task eval;
- время загрузки и размер рабочего набора.

Не меняйте одновременно batch, context, threads, GPU offload и sampling.
Сделайте warmup, сохраните все повторы и сообщайте median/p95.

## Часть 3. Проверить причинное объяснение

Удвойте context length. KV-составляющая должна вырасти линейно, тогда как
память весов останется почти постоянной. Затем смените только формат весов:
размер модели изменится, но FP16 KV-cache не обязан измениться. Если backend
поддерживает quantized KV, включите его отдельным экспериментом.

## Что сдать

- конфигурацию модели с $L,n_{kv},d_h$ и dtype;
- расчётную таблицу и график predicted/measured KV memory;
- полный benchmark protocol и raw samples;
- таблицу speed–memory–quality для форматов;
- profiler или runtime evidence, подтверждающий используемый backend;
- объяснение расхождений и минимум один regression check.

Работа считается законченной, когда формула предсказывает наклон измеренной
кривой, а вывод о квантизации учитывает одновременно память, скорость и
качество.
