---
title: "Gemma: Open Models Based on Gemini Research and Technology"
url: https://arxiv.org/abs/2403.08295
authors: "Gemma Team, Google DeepMind"
year: 2024
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/gemma/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
---

# Gemma: Open Models Based on Gemini Research and Technology

**Authors:** Gemma Team, Google DeepMind
**Published:** 2024 (arXiv:2403.08295v4, Apr 2024)
**URL:** https://arxiv.org/abs/2403.08295

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig1.png]]
*Performance Gemma 7B по категориям задач в сравнении с LLaMA 2 и Mistral.*

## TL;DR

Gemma -- семейство open-weight моделей (2B и 7B) от Google DeepMind, построенных на основе технологий Gemini. Gemma 7B превосходит сопоставимые open-source модели (LLaMA 2 7B, Mistral 7B) на 11 из 18 text-based бенчмарков, особенно в math (GSM8K 46.4 vs Mistral 35.4) и coding (HumanEval 32.3 vs Mistral 26.2). Выпущены pretrained и instruction-tuned (SFT + RLHF) checkpoints.

## Problem

На момент выпуска open-source LLM (LLaMA 2, Mistral, Falcon) уступали закрытым моделям (GPT-4, Gemini) по качеству и safety. Google стремится предоставить высококачественные open-weight модели с:
- Компактными размерами для deployment (2B -- on-device/CPU, 7B -- GPU/TPU)
- Strong performance на стандартных бенчмарках
- Детальной проработкой safety и responsible deployment

## Method

### Архитектура

Transformer decoder, context length 8192 токенов.

| Parameter | 2B | 7B |
|-----------|----|----|
| d_model | 2048 | 3072 |
| Layers | 18 | 28 |
| FF hidden dims | 32768 | 49152 |
| Num heads | 8 | 16 |
| Num KV heads | 1 (MQA) | 16 (MHA) |
| Head size | 256 | 256 |
| Vocab size | 256128 | 256128 |

Ключевые архитектурные решения:
- **Multi-Query Attention (MQA)** для 2B (num_kv_heads=1), **Multi-Head Attention** для 7B
- **RoPE** positional embeddings (в каждом слое)
- **GeGLU** activations вместо ReLU
- **RMSNorm** для стабилизации обучения
- Shared input/output embeddings для уменьшения размера модели
- Большой vocabulary (256k) от Gemini -- рассчитан на многоязычность

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig2.png]]
*Сравнение rates memorization между Gemma и PaLM моделями.*

### Обучение

- **Данные**: 2B обучена на 3T токенов, 7B -- на 6T токенов. Primarily English: web documents, mathematics, code.
- **Инфраструктура**: TPUv5e. 7B -- 4096 TPUv5e (16 pods), 2B -- 512 TPUv5e (2 pods).
- **Tokenizer**: subset SentencePiece из Gemini (splits digits, byte-level for unknown tokens).
- **Carbon footprint**: ~131 tCO2eq.

### Instruction Tuning

Двухэтапный процесс:
1. **SFT**: на синтетических и human-generated prompt-response парах (English-only). Выбор данных по LM-based side-by-side evaluations.
2. **RLHF**: reward model по Bradley-Terry, policy оптимизируется novel RL алгоритмом. Автоматический LM-rater для контроля reward hacking.

Специальные control tokens для форматирования: `<start_of_turn>`, `<end_of_turn>`, `user`, `model`.

### Filtering и Safety

- Фильтрация pre-training данных: heuristics + model-based classifiers для удаления harmful/low-quality content.
- Фильтрация personal information через Google Cloud Sensitive Data Protection.
- Evaluation sets удалены из pre-training data.
- Filtering SFT данных: personal info, unsafe outputs, self-identification errors, duplicates.

## Key Results

### Academic benchmarks (Table 6)

| Benchmark | LLaMA-2 7B | Mistral 7B | Gemma 2B | Gemma 7B |
|-----------|-----------|-----------|---------|---------|
| MMLU (5-shot) | 45.3 | 62.5 | 42.3 | **64.3** |
| HellaSwag | 77.2 | 81.0 | 71.4 | **81.2** |
| HumanEval (pass@1) | 12.8 | 26.2 | 22.0 | **32.3** |
| MBPP (3-shot) | 20.8 | 40.2 | 29.2 | **44.4** |
| GSM8K (maj@1) | 14.6 | 35.4 | 17.7 | **46.4** |
| MATH (4-shot) | 2.5 | 12.7 | 11.8 | **24.3** |
| BBH | 32.6 | 56.1 | 35.2 | **55.1** |
| Average | 46.9 | 54.5 | 45.0 | **56.9** |

Gemma 7B превосходит **LLaMA 2 13B** по Average (56.9 vs 52.4).

Особенно сильное преимущество в **math** (+10 GSM8K vs Mistral) и **coding** (+6 HumanEval vs Mistral).

### HuggingFace H6 benchmark (Table 7)

Gemma 7B: 63.8 average vs Mistral 7B: 61.0. Наибольшее преимущество на GSM8K (50.9 vs 37.8).

### Human preference (Gemma 1.1 IT vs Mistral v0.2 Instruct)

- **Safety**: 63.5% win rate (7B), 60.1% (2B)
- **Instruction Following**: 61.2% win rate (7B), 45% (2B)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig3.png]]
*Rates memorization personal data по data source.*

### Memorization

- Exact memorization rates сопоставимы с PaLM.
- **Нет** случаев memorization sensitive data.
- ~50% больше данных approximate-memorized vs exact.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig4.png]]
*Exact vs approximate memorization по data source.*

### Safety benchmarks (Table 8)

Gemma 7B IT превосходит Mistral 7B на CrowS-Pairs (49.67 vs 32.76), BBQ Ambig (86.06 vs 97.53 -- Mistral лучше), TruthfulQA (45.34 vs 48.54 -- Mistral лучше).

## My notes

