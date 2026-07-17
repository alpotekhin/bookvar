---
title: "XGrammar: Flexible and Efficient Structured Generation Engine For Large Language Models"
url: https://arxiv.org/abs/2411.15100
authors: [Anonymous (MLSys under review)]
year: 2024
date_reviewed: 2026-04-10
type: source-note
status: legacy
category: paper
tags:
  - structured-generation
  - constrained-decoding
  - inference
  - LLM
  - JSON
  - grammar
Organization: Anonymous (интегрировано в vLLM, MLC-LLM)
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]]"
---

# XGrammar: Flexible and Efficient Structured Generation Engine For Large Language Models

**URL:** https://arxiv.org/abs/2411.15100
**Published:** ноябрь 2024 (under review at MLSys)
**Интегрирован в:** vLLM (дефолтный бэкенд), MLC-LLM

## TL;DR

XGrammar — движок structured generation на основе context-free grammar (CFG). Ключевая идея: делит словарь токенов на **context-independent** (можно прекомпьютить) и **context-dependent** (проверяются в runtime). Даёт **до 100x ускорение** per-token latency и **80x ускорение end-to-end** (Llama-3.1 на H100) по сравнению с существующими решениями.

## Problem

Структурированная генерация (JSON, SQL, function calls) нужна агентам. Стандартный подход — constrained decoding: на каждом шаге маскировать логиты невалидных токенов.

Проблема наивной реализации CFG:
1. **Размер словаря** — нужно проверить каждый токен (до 128k в Llama 3.1) на каждом шаге
2. **Stack state** — CFG требует стека для отслеживания рекурсивных правил; невозможно прекомпьютить все комбинации
3. **Misaligned boundaries** — токен (`"true"`) может пересекать границы грамматических элементов, вызывая рекурсию или pop стека в runtime

## Method

### Ключевая идея: разделение токенов

```
Vocabulary
├── Context-independent tokens  ← проверяются только по локальному состоянию автомата
│   └── прекомпьютятся и кэшируются ДО генерации
└── Context-dependent tokens    ← требуют полного состояния стека
    └── проверяются в runtime (их меньшинство)
```

Большинство токенов — context-independent. XGrammar строит **adaptive token mask cache**: для каждой позиции в автомате хранит предвычисленную маску для context-independent токенов.

### Архитектура

**1. Byte-level Pushdown Automaton (PDA)**
CFG компилируется в PDA на уровне байт (не символов). Это решает проблему misaligned boundaries — токены обрабатываются как последовательности байт.

**2. Adaptive Token Mask Cache**
- При препроцессинге: для каждого состояния автомата прекомпьютится маска context-independent токенов
- В runtime: берётся cached partial mask + доопределяется для context-dependent токенов
- Оптимизированный формат хранения под каждую позицию автомата

**3. Context Expansion**
Алгоритмы расширяют "локальный контекст" каждого правила грамматики → больше токенов становятся context-independent → меньше работы в runtime.

**4. Persistent Execution Stack**
Стек с поддержкой быстрого branching и rollback:
- Ускоряет проверку context-dependent токенов
- Ускоряет препроцессинг кэша
- Copy-on-write семантика для ветвлений

**5. Co-design с LLM inference engine**
Вычисления грамматики перекрываются с GPU computation → near-zero overhead в end-to-end serving.

### Pipeline

```
LLM output tokens →
  PDA parsing → stack state →
    stack top → index в adaptive token mask cache →
      cached partial mask (context-independent) +
      runtime check (context-dependent) →
        complete token mask →
          logits masking → sampling
```

## Key Results

| Метрика | XGrammar vs SOTA |
|---------|-----------------|
| Per-token latency (CFG) | **до 100x ускорение** |
| End-to-end LLM serving (Llama-3.1, H100) | **до 80x ускорение** |
| Overhead при structured generation | **near-zero** |

Сравнение с Outlines и lm-format-enforcer: XGrammar значительно быстрее, особенно на больших словарях (128k токенов Llama 3.1).

## My Notes

- XGrammar стал дефолтным движком structured outputs в vLLM — это де-факто стандарт в продакшне
- Ключевой инсайт: большинство токенов можно проверить статически (context-independent). Раньше все проверяли динамически — отсюда и overhead
- Byte-level PDA vs character-level — важно для мультиязычных токенов и случаев когда токен пересекает границу JSON-элемента (`"tru` + `e"`)
- Конкуренты: Outlines (FSM + regex), lm-format-enforcer (token tree), Guidance (interleaved generation). XGrammar выигрывает по скорости за счёт препроцессинга
- Связано с обсуждением constrained decoding: automata-based подход (arXiv 2305.13971) — XGrammar это production-ready реализация тех же идей
- Практически: если деплоишь вLLM и нужен JSON output — XGrammar уже включён, настраивается через `guided_json` параметр
