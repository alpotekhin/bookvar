---
title: "Mixtral of Experts"
url: https://arxiv.org/abs/2401.04088
authors: "Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, et al."
year: 2024
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/mixtral-of-experts/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
---

# Mixtral of Experts

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, et al. (Mistral AI)
**Published:** 2024
**URL:** https://arxiv.org/abs/2401.04088

## TL;DR

Mixtral 8x7B -- Sparse Mixture of Experts (SMoE) language model, где каждый FFN-блок заменён на 8 экспертов, из которых router выбирает 2 для каждого токена. Итого 47B параметров, но только 13B active per token. При 5x меньшем inference compute превосходит или равен Llama 2 70B и GPT-3.5 на большинстве бенчмарков. Mixtral 8x7B -- Instruct (SFT + DPO) превосходит GPT-3.5 Turbo, Claude-2.1, Gemini Pro на human benchmarks. Apache 2.0.

## Problem

Масштабирование LLM через увеличение dense параметров повышает и training и inference cost пропорционально. Sparse Mixture of Experts позволяет увеличить model capacity (total parameters) при фиксированном compute per token (active parameters). Вопрос: можно ли с SMoE на 13B active params достичь качества 70B dense model?

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
| context_len | 32768 |
| vocab_size | 32000 |
| num_experts | 8 |
| top_k_experts | 2 |

Базируется на Mistral 7B с заменой FFN блоков на MoE layers и расширением context length до 32K.

### Sparse Mixture of Experts (Section 2.1)

Для входного токена x:
```
y = sum_{i=0}^{n-1} Softmax(Top2(x * Wg))_i * SwiGLU_i(x)
```

- **Router:** линейный слой Wg проецирует x в n-мерный вектор логитов
- **TopK:** выбираются 2 эксперта с наибольшими логитами, остальные обнуляются
- **Softmax:** нормализация весов только по выбранным экспертам
- **Expert function:** SwiGLU (такой же как FFN в vanilla Mistral 7B)

Каждый эксперт -- полноценный FFN блок (hidden_dim=14336), итого 8 * FFN_params + router_params на каждом из 32 слоёв.

**Sparse vs Active parameters:**
- Total (sparse): 47B (все 8 экспертов + shared attention)
- Active per token: 13B (2 эксперта + shared attention)

### Efficient Inference

- **Megablocks:** CUDA kernels для MoE как sparse matrix multiplications, handle variable number of tokens per expert
- **Expert Parallelism:** распределение экспертов по GPU, токены маршрутизируются к соответствующему GPU
- Memory cost пропорционален sparse count (47B) -- всё ещё меньше Llama 2 70B

### Instruction Fine-tuning (Section 4)

SFT на instruction dataset + DPO на paired feedback dataset. MT-Bench: 8.30 -- лучшая open-weights модель на декабрь 2023.

## Key Results

### Основные бенчмарки (Table 2)

| Model | Active Params | MMLU | HumanEval | MBPP | MATH | GSM8K |
|-------|--------------|------|-----------|------|------|-------|
| LLaMA 2 7B | 7B | 44.4% | 11.6% | 26.1% | 3.9% | 16.0% |
| LLaMA 2 13B | 13B | 55.6% | 18.9% | 35.4% | 6.0% | 34.3% |
| LLaMA 2 70B | 70B | 69.9% | 29.3% | 49.8% | 13.8% | 69.6% |
| Mistral 7B | 7B | 62.5% | 26.2% | 50.2% | 12.7% | 50.0% |
| **Mixtral 8x7B** | **13B** | **70.6%** | **40.2%** | **60.7%** | **28.4%** | **74.4%** |

Mixtral превосходит Llama 2 70B на почти всех бенчмарках при 5x меньшем active compute. Особенно сильно на code (HumanEval 40.2% vs 29.3%) и math (MATH 28.4% vs 13.8%, GSM8K 74.4% vs 69.6%).

### vs GPT-3.5 (Table 3)

| Benchmark | LLaMA 2 70B | GPT-3.5 | Mixtral 8x7B |
|-----------|-------------|---------|-------------|
| MMLU | 69.9% | 70.0% | 70.6% |
| MBPP | 49.8% | 52.2% | 60.7% |
| GSM-8K | 53.6% | 57.1% | 58.4% |
| MT Bench (Instruct) | 6.86 | 8.32 | 8.30 |

### Multilingual (Table 4)

Значительно превосходит Llama 2 70B на French, German, Spanish, Italian (Arc-c, HellaSwag, MMLU).

### Long Range (Figure 4)

100% accuracy на Passkey Retrieval при любой длине и позиции в контексте 32K. Perplexity монотонно снижается с ростом context length.

### Bias (Figure 5)

BBQ accuracy: 56.0% vs Llama 2 70B 51.5%. BOLD sentiment: более позитивный с меньшим std.

### Chatbot Arena (Figure 6)

Mixtral 8x7B Instruct v0.1: Elo 1121, превосходит Claude-2.1 (1117), GPT-3.5-Turbo best (1117), Gemini Pro (1111), Llama-2-70b-chat (1077).

### Routing Analysis (Section 5, Figure 7)

Эксперты НЕ специализируются по доменам (ArXiv, biology, philosophy имеют похожие распределения). Роутинг скорее синтаксический: одинаковые токены (self, Question, индентация) направляются к одним экспертам. Высокая temporal locality: в средних/последних слоях ~60-67% consecutive tokens идут к тому же эксперту (vs 46% при random). Это можно использовать для caching оптимизации.

## My notes
