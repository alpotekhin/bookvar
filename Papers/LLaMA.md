---
title: "LLaMA: Open and Efficient Foundation Language Models"
url: https://arxiv.org/abs/2302.13971
authors: [Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample]
year: 2023
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - LLaMA
  - arch
Date: 2023-01-02
Organization: Meta
Status: Done
Sub-item:
  - "[[LLaMA 2]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]]"
raw: "[[02 Areas/ML & DL/raw/papers/llama/paper.txt]]"
---

# LLaMA: Open and Efficient Foundation Language Models

**Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard et al. (Meta AI)
**Published:** 2023 (arXiv:2302.13971v1, Feb 2023)
**URL:** https://arxiv.org/abs/2302.13971

## TL;DR

LLaMA — серия открытых foundation моделей от Meta (7B–65B параметров), обученных исключительно на **публично доступных данных**. Ключевой результат: **LLaMA-13B превосходит GPT-3 (175B) на большинстве бенчмарков**, несмотря на то, что в 10× меньше. LLaMA-65B конкурентоспособна с Chinchilla-70B и PaLM-540B. Главный инсайт: оптимальная стратегия — обучать маленькие модели **на большем количестве токенов**, чем рекомендуют Chinchilla scaling laws, чтобы минимизировать inference budget, а не training compute.

## Problem

Доминирующая парадигма после Chinchilla (Hoffmann et al., 2022): для заданного compute budget оптимально иметь меньшую модель, обученную на большем датасете. Однако эта оптимизация направлена на **training compute**, игнорируя **inference budget**.

Проблема: при масштабном деплое модели стоимость inference доминирует над стоимостью обучения. Если обучить небольшую модель ещё дольше (больше токенов), получаем модель, дешёвую при inference — даже если на её обучение ушло больше compute, чем "оптимально" по Chinchilla.

Авторы находят, что **производительность 7B модели продолжает расти даже после 1 триллиона токенов** — Chinchilla рекомендует только 200B для модели такого размера.

Дополнительно: все сильные модели того времени (PaLM, GPT-3, Chinchilla) используют проприетарные данные. LLaMA показывает, что можно достичь SOTA на публично доступных данных, делая работу совместимой с open-source.

## Method

### Architecture

[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] Transformer с тремя ключевыми модификациями относительно оригинального Transformer:

**1. Pre-normalization (из GPT-3):** Нормализация входа каждого трансформерного sub-слоя вместо нормализации выхода. Используется **RMSNorm** (Zhang & Sennrich, 2019) вместо LayerNorm. Улучшает стабильность обучения.

**2. SwiGLU activation (из PaLM):** Замена ReLU на **SwiGLU** (Shazeer, 2020). Используется размер hidden dimension `2/3 × 4d` вместо `4d`.

**3. Rotary Positional Embeddings — RoPE (из GPT-NeoX):** Удалены абсолютные positional embeddings. Вместо них добавляются **RoPE** (Su et al., 2021) на каждом слое сети.

### Model Sizes (Table 2)

| Параметры | Dimension | Heads | Layers | LR | Tokens |
|-----------|-----------|-------|--------|----|--------|
| 6.7B | 4096 | 32 | 32 | 3.0e-4 | 1.0T |
| 13.0B | 5120 | 40 | 40 | 3.0e-4 | 1.0T |
| 32.5B | 6656 | 52 | 60 | 1.5e-4 | 1.4T |
| 65.2B | 8192 | 64 | 80 | 1.5e-4 | 1.4T |

### Pre-training Data (Table 1)

| Датасет | Доля | Токены | Эпоки |
|---------|------|--------|-------|
| CommonCrawl (CCNet pipeline) | 67.0% | ~938B | 1.10 |
| C4 | 15.0% | ~210B | 1.06 |
| Github (Apache/BSD/MIT) | 4.5% | ~63B | 0.64 |
| Wikipedia (20 языков, 2022) | 4.5% | ~63B | 2.45 |
| Books (Gutenberg + Books3) | 4.5% | ~63B | 2.23 |
| ArXiv | 2.5% | ~35B | 1.06 |
| StackExchange (28 сайтов) | 2.0% | ~28B | 1.03 |

Всего после токенизации: **~1.4T токенов**. Только публично доступные данные.

**Токенизатор:** Byte-Pair Encoding (BPE) через SentencePiece. Числа разбиваются на отдельные цифры; неизвестные UTF-8 символы декомпозируются в байты.

### Optimizer

AdamW (β₁=0.9, β₂=0.95), cosine learning rate schedule (финальный LR = 10% от максимального), weight decay 0.1, gradient clipping 1.0, 2,000 warmup steps.

### Efficient Implementation

- **Efficient causal multi-head attention** (xformers library): не сохраняются attention weights, не вычисляются masked key/query scores → снижение memory usage и runtime
- **Activation checkpointing:** сохраняются только дорогостоящие активации (выходы linear слоев), остальные пересчитываются при backward
- **Model + sequence parallelism** через GPU
- **LLaMA-65B:** ~380 tokens/sec/GPU на 2048 A100 (80GB). Обучение на 1.4T токенах ≈ 21 день.

## Key Results

### Common Sense Reasoning (Table 3, zero-shot)

| Модель | Размер | BoolQ | HellaSwag | WinoGrande | ARC-e | ARC-c |
|--------|--------|-------|-----------|------------|-------|-------|
| GPT-3 | 175B | 60.5 | 78.9 | 70.2 | 68.8 | 51.4 |
| Chinchilla | 70B | 83.7 | 80.8 | 74.9 | - | - |
| PaLM | 540B | 88.0 | 83.4 | 81.1 | 76.6 | 53.0 |
| **LLaMA** | **13B** | **78.1** | **79.2** | **73.0** | **74.8** | **52.7** |
| **LLaMA** | **65B** | **85.3** | **84.2** | **77.0** | **78.9** | **56.0** |

**LLaMA-13B превосходит GPT-3 (175B)** на большинстве бенчмарков, будучи в 10× меньше.

### Closed-book QA

**NaturalQuestions** (Table 4): LLaMA-65B 64-shot — 39.9% (PaLM-540B: 39.6%) — паритет с гораздо большей моделью.

**TriviaQA** (Table 5): LLaMA-65B 64-shot — **73.0%** (Chinchilla 70B: 64.6%) — превосходит Chinchilla.

### Reading Comprehension (Table 6, zero-shot)

RACE-high: LLaMA-65B — **51.6%** (GPT-3: 45.5%, PaLM-540B: 49.1%).

### Math & Code

**GSM8k** (math): LLaMA-65B — **50.9%** без majority voting, 69.7% с maj1@100. LLaMA-65B outperforms Minerva-62B (52.4%), несмотря на то что не дообучался на математических данных.

**HumanEval** (Python): LLaMA-65B pass@1 — **23.7%** (PaLM-540B: 26.2%). LLaMA-13B превосходит LaMDA-137B (14.0%).

### MMLU (Table 9, 5-shot)

| Модель | Humanities | STEM | Social | Other | Average |
|--------|-----------|------|--------|-------|---------|
| GPT-3 175B | 40.8 | 36.7 | 50.4 | 48.8 | 43.9 |
| Chinchilla 70B | 63.6 | 54.9 | 79.3 | 73.9 | 67.5 |
| PaLM 540B | 77.0 | 55.6 | 81.0 | 69.6 | 69.3 |
| **LLaMA 65B** | **61.8** | **51.7** | **72.9** | **67.4** | **63.4** |

LLaMA-65B отстаёт от Chinchilla-70B и PaLM-540B на ~4-6%. Авторы объясняют это меньшим объёмом книг/академических статей в данных.

### Instruction Finetuning (Table 10)

LLaMA-I (65B, instruction-tuned): **68.9%** на MMLU — превосходит Flan-PaLM-cont-62B (66.1%).

## Limitations & Safety

- MMLU отставание от Chinchilla: вероятно из-за меньшего объёма книг (177GB vs ~2TB)
- **Toxicity растёт с размером модели** (RealToxicityPrompts, Table 11): LLaMA-65B basic=0.128 vs 7B=0.106
- Gender bias: 70.6% верных назначений в WinoGender (GPT-3: 62.6%)
- Модели released только для исследований; не instruction-tuned → могут генерировать вредоносный контент

## My notes

- LLaMA — один из самых важных open releases в истории LLM. Дал толчок всей экосистеме: Alpaca, Vicuna, Mistral и т.д. всё из этого корня.
- Ключевой сдвиг парадигмы: оптимизировать под **inference budget**, а не training compute. Это фундаментально другой вывод из scaling laws.
- Pre-normalization (RMSNorm) + SwiGLU + RoPE стали **стандартным стеком** для большинства последующих open-source LLM (LLaMA 2, Mistral, Falcon...). Эти три модификации — де-факто архитектурный шаблон современных decoder-only LLM.
- Публичные данные = открытая наука. Важный принципиальный вклад помимо технического.
