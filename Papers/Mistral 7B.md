---
title: "Mistral 7B"
url: https://arxiv.org/abs/2310.06825
authors: "Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, et al."
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/mistral-7b/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
---

# Mistral 7B

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. (Mistral AI)
**Published:** 2023
**URL:** https://arxiv.org/abs/2310.06825

## TL;DR

Mistral 7B -- 7-миллиардная language model, которая превосходит Llama 2 13B на всех бенчмарках и Llama 1 34B на reasoning, math, code. Ключевые архитектурные решения: Grouped-Query Attention (GQA) для быстрого inference и Sliding Window Attention (SWA) с окном 4096 для эффективной обработки длинных последовательностей. Демонстрирует, что LM может сжимать знания значительно эффективнее, чем считалось ранее. Apache 2.0 лицензия.

## Problem

Гонка за performance в NLP часто требует увеличения размера моделей, что повышает стоимость inference и снижает доступность для реальных приложений. Вопрос: можно ли достичь high performance с компактной моделью при правильном архитектурном дизайне? Авторы утверждают, что задача трёхмерная (capabilities, training cost, inference cost), а не двумерная (capabilities, training cost) как в scaling laws.

## Method

### Архитектура (Table 1)

| Parameter | Value |
|-----------|-------|
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 |
| window_size | 4096 |
| context_len | 8192 |
| vocab_size | 32000 |

Базируется на transformer architecture с несколькими модификациями:

### Sliding Window Attention (SWA)

Каждый токен может attend к максимум W=4096 предыдущим токенам в одном слое. Благодаря стеку слоёв, информация распространяется по цепочке: после k attention layers один токен имеет доступ к токенам на расстоянии до W*k. На последнем (32-м) слое с W=4096 -- теоретический attention span ~131K токенов. Модификации FlashAttention и xFormers дают 2x speedup по сравнению с vanilla attention на длине 16K.

### Rolling Buffer Cache

Фиксированный размер KV cache = W. Ключи и значения для позиции i хранятся в позиции i mod W. При последовательности 32K токенов -- 8x уменьшение memory usage cache без потери качества.

### Pre-fill and Chunking

Промпт разбивается на чанки размером W=4096 для pre-fill KV cache. Каждый чанк: attend к cache (sliding window) + attend к себе (causal mask). Контролирует пиковый расход памяти.

### Grouped-Query Attention (GQA)

8 KV heads на 32 query heads (n_kv_heads=8, n_heads=32, ratio 1:4). Ускоряет inference, снижает memory при decoding, позволяет больший batch size.

### Instruction Finetuning (Mistral 7B -- Instruct)

Простой fine-tuning на публичных instruction datasets с Hugging Face, без proprietary data или tricks.

### Guardrails (Section 5)

System prompt для safety: "Always assist with care, respect, and truth. Respond with utmost utility yet securely..."
- С system prompt: 100% decline rate на 175 unsafe промптах
- Без потери MT-Bench: 6.58 (Mistral system prompt) vs 6.84 (no prompt) vs 6.38 (Llama 2 system prompt)

Self-reflection для content moderation: precision 99.4%, recall 95.6% на классификации промптов как acceptable/harmful.

## Key Results

### Основные бенчмарки (Table 2)

| Model | MMLU | HellaSwag | HumanEval | MBPP | MATH | GSM8K |
|-------|------|-----------|-----------|------|------|-------|
| LLaMA 2 7B | 44.4% | 77.1% | 11.6% | 26.1% | 3.9% | 16.0% |
| LLaMA 2 13B | 55.6% | 80.7% | 18.9% | 35.4% | 6.0% | 34.3% |
| Code-Llama 7B | 36.9% | 62.9% | 31.1% | 52.5% | 5.2% | 20.8% |
| **Mistral 7B** | **60.1%** | **81.3%** | **30.5%** | **47.5%** | **13.1%** | **52.2%** |

Mistral 7B превосходит Llama 2 13B на всех метриках. Приближается к code performance Code-Llama 7B (HumanEval 30.5% vs 31.1%) без потери на non-code бенчмарках.

### Equivalent Model Size (Figure 5)

На reasoning, comprehension, STEM (MMLU): Mistral 7B эквивалентен Llama 2 модели >3x его размера. На knowledge benchmarks compression rate 1.9x (ограничение по количеству параметров для хранения знаний).

### Chat Model (Table 3)

| Model | Chatbot Arena ELO | MT Bench |
|-------|-------------------|----------|
| WizardLM 13B v1.2 | 1047 | 7.2 |
| **Mistral 7B Instruct** | **1031** | **6.84** |
| Llama 2 13B Chat | 1012 | 6.65 |
| Llama 2 7B Chat | 985 | 6.27 |

Mistral 7B Instruct превосходит все 7B модели на MT-Bench и сравним с 13B chat моделями. Human evaluation: Mistral 7B preferred 5020 раз vs Llama 2 13B 4143 раза.

## My notes

