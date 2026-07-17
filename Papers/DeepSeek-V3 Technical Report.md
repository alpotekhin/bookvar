---
title: "DeepSeek-V3 Technical Report"
url: https://arxiv.org/abs/2412.19437
authors: "DeepSeek-AI"
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/deepseek-v3/source]]"
concepts: [Mixture of Experts, Multi-Head Latent Attention, FP8 Training, Multi-Token Prediction, Load Balancing, Reinforcement Learning, Knowledge Distillation]
---
# DeepSeek-V3 Technical Report

## TL;DR

DeepSeek-V3 -- MoE-модель на 671B параметров (37B активных на токен), обученная всего за **$5.576M** на 14.8T токенов. Использует Multi-head Latent Attention (MLA) для эффективного inference, auxiliary-loss-free load balancing, multi-token prediction и FP8 training. Сопоставима по качеству с GPT-4o и Claude-3.5-Sonnet при радикально меньших затратах.

## Problem

Открытые LLM отставали от closed-source моделей (GPT-4o, Claude) по качеству, а их обучение требовало колоссальных вычислительных ресурсов. Нужна архитектура, способная масштабироваться эффективно -- с минимальным бюджетом, стабильным тренингом и высоким quality.

## Method

### Архитектура
- **Multi-Head Latent Attention (MLA)**: low-rank сжатие K/V в латентное пространство, кэшируется только сжатый вектор + decoupled RoPE ключ. Результат -- радикальное уменьшение KV-cache при сохранении качества стандартного MHA.
- **DeepSeekMoE**: fine-grained experts (256 routed + 1 shared), Top-8 routing. В отличие от GShard, эксперты более мелкие, что даёт лучшую специализацию.
- **Auxiliary-Loss-Free Load Balancing**: вместо auxiliary loss (который деградирует quality) используется bias term в gating, обновляемый на основе текущего дисбаланса. Нет потери качества от балансировки.
- **Multi-Token Prediction (MTP)**: предсказание нескольких следующих токенов одновременно через дополнительные MTP-модули. При inference может использоваться для speculative decoding (1.8x ускорение acceptance rate).

### Инфраструктура
- **DualPipe**: pipeline parallelism с overlap вычислений и коммуникаций, минимизация bubble time.
- **FP8 mixed precision**: tile-wise quantization (1x128 для активаций, 128x128 для весов), высокоточная аккумуляция на CUDA cores каждые 128 элементов. Первая успешная валидация FP8 на модели такого масштаба.
- **2048 H800 GPU** кластер, InfiniBand + NVLink, 20 SMs на коммуникацию.

### Обучение
- 14.8T токенов pre-training, два этапа context extension (32K → 128K)
- SFT: 1.5M примеров, включая reasoning data дистиллированный из DeepSeek-R1
- RL: GRPO с rule-based rewards для math/code + reward model для общих задач

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-v3-figure1-hq.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-v3-figure2-architecture-hq.png]]

## Key Results

| Benchmark | DeepSeek-V3 | GPT-4o | Claude-3.5-Sonnet |
|---|---|---|---|
| MMLU-Pro (EM) | 75.9 | 72.6 | 78.0 |
| GPQA-Diamond | 59.1 | 49.9 | 65.0 |
| MATH-500 | 90.2 | 74.6 | 78.3 |
| AIME 2024 | 39.2 | 9.3 | 16.0 |
| Codeforces (Percentile) | 51.6 | 23.6 | 20.3 |
| SWE-bench Verified | 42.0 | 38.8 | 50.8 |

- **Training cost**: 2.788M H800 GPU hours = **$5.576M** (только official training)
- Pre-training: 2664K GPU hours, context extension: 119K, post-training: 5K
- Zero irrecoverable loss spikes за весь pre-training
- FP8 training: relative loss error < 0.25% vs BF16 baseline

## My notes

- Это переломный момент для индустрии -- модель уровня GPT-4o за $5.5M training cost. Ключ не в архитектурных инновациях (MLA/MoE были в V2), а в инженерной оптимизации: FP8, DualPipe, communication overlap.
- Auxiliary-loss-free balancing -- элегантное решение давней проблемы MoE. Простой bias term вместо auxiliary loss, который всегда деградировал quality.
- MTP даёт двойной выигрыш: лучше бенчмарки на train-time + speculative decoding на inference. Но acceptance rate 85-90% при top-1, что указывает на ceiling.
- Distillation из R1 в V3 post-training -- интересный паттерн: reasoning модель "учит" general модель думать, сохраняя при этом compact output style.
- Стабильность тренинга (zero rollbacks на 14.8T токенов) -- это нетривиально для MoE такого масштаба и говорит о высокой зрелости инфраструктуры.
