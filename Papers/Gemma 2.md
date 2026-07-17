---
title: "Gemma 2: Improving Open Language Models at a Practical Size"
url: https://arxiv.org/abs/2408.00118
authors: "Gemma Team, Google DeepMind"
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/gemma-2/source]]"
concepts: [Knowledge Distillation, Grouped Query Attention, Sliding Window Attention, Logit Soft-Capping, Model Merging]
---
# Gemma 2: Improving Open Language Models at a Practical Size

## TL;DR

Gemma 2 -- семейство open-weight моделей (2B, 9B, 27B) от Google DeepMind. Ключевая идея: **knowledge distillation вместо next-token prediction** для маленьких моделей (2B, 9B). Плюс architectural tweaks: interleaving local-global attention, logit soft-capping, deeper networks. Результат -- модели, конкурентные с моделями в 2-3x больше.

## Problem

Small-scale модели улучшаются в основном за счёт увеличения training length, но gains логарифмически уменьшаются с размером dataset. Нужен способ улучшить маленькие модели без бесконечного наращивания токенов -- через более богатый training signal.

## Method

### Архитектура
- **Local-Global Attention interleaving**: чередование sliding window attention (4096 tokens) и global attention (8192 tokens) в каждом слое. Более эффективно, чем all-global.
- **Logit soft-capping**: `logits = soft_cap * tanh(logits / soft_cap)` -- ограничивает logits в [-soft_cap, +soft_cap]. 50.0 для attention layers, 30.0 для final layer. Стабилизирует training.
- **Post-norm + Pre-norm**: RMSNorm на входе И выходе каждого sub-layer. Double normalization для стабильности.
- **GQA** с num_groups=2 -- ускорение inference при сохранении quality.
- **Deeper networks**: 9B модель глубже, чем шире -- ablations показывают small но consistent improvement.

### Training
- 27B: 13T tokens from scratch, 9B: 8T tokens with distillation, 2B: 2T tokens with distillation
- **Knowledge Distillation**: минимизация cross-entropy между teacher (large model) и student distributions: `min -P_T(x|x_c) * log P_S(x|x_c)`
- Distillation позволяет тренировать на 50x больше compute-optimal количества токенов, симулируя "тренинг за пределами доступных данных"
- TPUv4/v5e/v5p infrastructure

### Post-training
- SFT на synthetic + human data (teacher generates responses)
- RLHF с reward model (order of magnitude larger than policy)
- **Model merging**: averaging моделей с разными hyperparameters
- Data filtering для safety, dedup, self-identification

| Model | MMLU | GSM8K | ARC-c | HellaSwag |
|---|---|---|---|---|
| Gemma 2 27B | 75.2 | 74.0 | 71.4 | 86.4 |
| Qwen1.5 32B | 74.3 | 61.1 | 63.6 | 85.0 |
| LLaMA-3 70B | 79.2 | 76.9 | 68.8 | 88.0 |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma-2/figure1-memorization.png]]

*Figure 1 из отчёта Gemma 2: сравнение memorization rates между семействами моделей и источниками данных.*

## Key Results

- **Gemma 2 27B**: Elo 1218 на Chatbot Arena, выше Llama 3 70B (Elo 1206) -- при 2.5x меньшем размере
- **Gemma 2 9B**: Elo 1187, сопоставимо с GPT-4-0314 (Elo 1186)
- **Gemma 2 2.6B**: Elo 1126, выше GPT-3.5-Turbo-0613 (Elo 1116)
- Distillation даёт +7.4% averaged benchmarks для 2B (60.3→67.7)
- Carbon footprint: 1247.61 tCO2eq (carbon neutral datacenters)

## My notes

- **Distillation > longer training** для маленьких моделей -- это ключевой takeaway. Gain от distillation сохраняется при увеличении model size (ablation с 200M→1B).
- Logit soft-capping -- интересный trick, пришёл из RL literature. Предотвращает extreme logit values, которые дестабилизируют attention.
- Local-global attention alternation -- pragmatic compromise: не всем слоям нужен full context. Sliding window в 4096 достаточно для local patterns.
- Model merging в post-training -- простая но effective техника. Averaging моделей с разными hyperparams работает как implicit ensemble.
- 256K vocabulary (от Gemini) -- overhead для English-only, но designed для multilingual. Trade-off: больше embedding parameters.
- Paper довольно краткий (15 pages) для техрепорта -- мало деталей про data pipeline, нет training dynamics. Google стиль: результат есть, details skipped.
