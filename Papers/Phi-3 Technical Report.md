---
title: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"
url: https://arxiv.org/abs/2404.14219
authors: "Microsoft"
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/phi-3/source]]"
concepts: [Data Quality, Synthetic Data, Small Language Models, Mixture of Experts, Blocksparse Attention, Knowledge Distillation]
---
# Phi-3 Technical Report

## TL;DR

Phi-3 -- семейство small language models от Microsoft: phi-3-mini (3.8B), phi-3-small (7B), phi-3-medium (14B), phi-3.5-MoE (16x3.8B, 6.6B active). Главный тезис: **data quality > model size**. За счёт heavily filtered web data + synthetic LLM-generated data, 3.8B модель достигает уровня Mixtral 8x7B и GPT-3.5. Phi-3-mini работает на iPhone с 12+ tokens/sec.

## Problem

Scaling laws предполагают "fixed" data source. Но LLM сами могут генерировать и фильтровать данные, что меняет trade-off между model size и data quality. Можно ли получить GPT-3.5 level quality в модели, помещающейся на телефоне?

## Method

### Архитектура
- **Phi-3-mini** (3.8B): Llama-2 compatible architecture, 32 heads, 32 layers, dim 3072, vocab 32064. Context: 4K default, 128K via LongRope.
- **Phi-3-small** (7B): tiktoken tokenizer (100K vocab), 32 heads/layers, dim 4096. GEGLU activation, muP для hyperparameter transfer. **Blocksparse attention**: разные sparsity patterns по heads, alternating dense/blocksparse layers. Triton kernel для training, vLLM для inference.
- **Phi-3-medium** (14B): 40 heads/layers, dim 5120, same data, 4.8T tokens.
- **Phi-3.5-MoE** (16x3.8B): top-2 routing, 6.6B active params, 42B total. SparseMixer для router training.

### Training Data ("Data Optimal Regime")
- **Phase 1**: heavily filtered web data -- "educational level" filtering. Убраны factual trivia (sports scores), оставлены reasoning-heavy pages.
- **Phase 2**: even more filtered web (subset of Phase 1) + synthetic LLM-generated data для logical reasoning и niche skills.
- Ключевая идея: для маленьких моделей capacity ограничена, поэтому данные должны быть оптимизированы под reasoning, а не general knowledge.
- 3.3T tokens для mini, 4.8T для small/medium.

### Post-training
- SFT: curated data across math, coding, reasoning, conversation, safety
- DPO: chat format, reasoning, RAI
- English-only SFT, multilingual mid-training для phi-3.5 versions

### On-device
- 4-bit quantization: ~1.8GB memory
- iPhone 14 (A16 Bionic): 12+ tokens/sec, fully offline

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/phi-3/fig1.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/phi-3/fig3.png]]

## Key Results

| Benchmark | Phi-3-mini (3.8B) | Phi-3-medium (14B) | Mixtral 8x7B | GPT-3.5 |
|---|---|---|---|---|
| MMLU (5-shot) | 68.8 | 78.0 | 70.5 | 71.4 |
| GSM-8K (8-shot CoT) | 82.5 | 91.0 | 64.7 | 78.1 |
| MATH (0-shot CoT) | 41.3 | 53.1 | 11.1 | 45.3 |
| HumanEval (0-shot) | 58.5 | 62.2 | 37.8 | 62.2 |
| BigBench-Hard (3-shot) | 71.7 | 81.4 | 69.7 | 68.3 |
| MT-Bench | 8.38 | 8.91 | -- | 8.35 |

- Phi-3.5-MoE: competitive с Gemini-1.5-Flash, >90% of GPT-4o-mini performance
- Phi-3.5-MoE MMLU multilingual: 69.9 average
- RepoQA: phi-3.5-MoE 85% average, выше Llama-3.1-8B (71%)

## My notes

- **"Data Optimal Regime"** -- это главный contribution. Не compute optimal (Chinchilla), не over-train (Llama), а подбор данных под capacity модели. Для маленьких моделей reasoning > facts.
- Phi-3-mini = Mixtral quality at 12x fewer parameters. Но есть caveat: data filtering через LLM -- это indirect использование large model compute, не отражённое в training cost.
- Blocksparse attention в phi-3-small -- интересное решение для KV cache reduction. Разные heads видят разные parts of context, в сумме все tokens covered.
- muP (Maximal Update Parametrization) для hyperparameter transfer -- позволяет тюнить на proxy model и переносить на target. Экономит compute на hyperparameter search.
- On-device inference (12 tok/s на iPhone) -- practical milestone. 4-bit quantization с minimal quality loss делает 3.8B модели deployment-ready.
- Phi-3.5-MoE -- неожиданно strong: 6.6B active params конкурирует с 8B dense models. SparseMixer для router -- claimed improvement over standard top-k routing.
- Scaling law plot (Fig 3) показывает: phi-series consistently ниже Llama-2 trend line на MMLU error vs model size. Data quality -- multiplier для effective model size.
