---
title: "Gemini: A Family of Highly Capable Multimodal Models"
url: https://arxiv.org/abs/2312.11805
authors: "Gemini Team, Google"
year: 2023
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/gemini/source]]"
concepts: [Multimodal, Transformer, TPU, Multi-Query Attention, Distillation, Safety, Chain of Thought]
---
# Gemini: A Family of Highly Capable Multimodal Models

## TL;DR

Gemini -- семейство native multimodal моделей от Google (Ultra, Pro, Nano), обученных **jointly** на text, image, audio и video. Ultra -- первая модель, превзошедшая human-expert performance на MMLU (90.04%). SOTA на 30 из 32 бенчмарков. Nano (1.8B, 3.25B) -- on-device через distillation + 4-bit quantization. AlphaCode 2 (на базе Gemini) -- top-15% на Codeforces.

## Problem

Существующие multimodal модели были либо compositional (отдельные encoders + adapter), либо joint но уступающие text-only specialist models. Вопрос: можно ли получить модель, которая одновременно SOTA в text, image, audio И video -- при joint training?

## Method

### Архитектура
- Transformer decoder с improvements для stable large-scale training
- **Multi-query attention** для efficient inference
- 32K context length
- Native multimodal: text, image, audio, video inputs interleaved
- Image encoding: inspired by Flamingo, CoCa, PaLI. Variable input resolution.
- Video: sequence of frames в context window
- Audio: 16kHz features от Universal Speech Model (USM)
- Discrete image tokens на выходе (может генерировать images)

### Размеры
| Model | Description |
|---|---|
| Ultra | Maximum capability, complex reasoning + multimodal |
| Pro | Performance-optimized, good cost/latency |
| Nano-1 (1.8B) | On-device, low memory |
| Nano-2 (3.25B) | On-device, high memory |

- Nano: distillation из больших Gemini + 4-bit quantization

### Training Infrastructure
- TPUv4 и TPUv5e, multi-datacenter training
- SuperPods: 4096 chips с optical switch, dynamic 3D torus topologies
- Model parallelism within superpods, data parallelism across superpods
- **Goodput optimization**: redundant in-memory copies вместо periodic checkpointing → recovery time dramatically reduced, goodput 85%→97%
- **Silent Data Corruption (SDC)** detection: deterministic replay для isolation, proactive SDC scanners
- Single controller programming (Jax + Pathways)

### Pre-training
- Multimodal + multilingual dataset: web, books, code, image, audio, video
- SentencePiece tokenizer trained на full corpus
- Quality filters: heuristic + model-based, safety filtering
- Staged training: altering mixture composition towards end
- Data decontamination: removed evaluation data from training

### Post-training
- Two variants: Gemini Apps (chat-focused) и Gemini API (developer-focused)
- MMLU: CoT@32 с uncertainty-based fallback to greedy -- 90.04%

## Key Results

| Benchmark | Gemini Ultra | GPT-4 | PaLM 2-L |
|---|---|---|---|
| MMLU (CoT@32) | 90.04% | 87.29% | 78.4% |
| GSM8K (Maj1@32) | 94.4% | 92.0% | 80.0% |
| MATH (4-shot) | 53.2% | 52.9% | 34.4% |
| BIG-Bench-Hard (3-shot) | 83.6% | 83.1% | 77.7% |
| HumanEval (0-shot) | 74.4% | 67.0% | -- |
| Natural2Code (0-shot) | 74.9% | 73.9% | -- |
| MMMU | 62.4% | -- | -- |

- Первая модель с >90% на MMLU (human expert = 89.8%)
- SOTA на 30/32 бенчмарков
- SOTA на всех 9 image, 6 video, 5 speech benchmarks
- AlphaCode 2: top-15% на Codeforces (vs top-50% для AlphaCode 1)
- HellaSwag caveat: +100 fine-tuning steps на related data → 96%, указывает на sensitivity к pretraining composition

## My notes

- **Native multimodal** -- правильный путь. Joint training text+image+audio+video даёт модель, которая reasoning across modalities, а не просто routing between encoders.
- MMLU 90% through CoT@32 + uncertainty fallback -- clever evaluation strategy, но inflates numbers vs standard few-shot. Важно сравнивать fair.
- Infrastructure innovations (goodput 97%, SDC detection) -- underappreciated contributions. На таких масштабах reliability > raw performance.
- Variable input resolution -- важно для practical image understanding. Больше compute на tasks, требующих fine-grained understanding.
- Nano через distillation -- эффективный путь к on-device. 1.8B/3.25B с 4-bit -- realistic для mobile.
- Paper осторожен с деталями: нет architecture specifics (model dimensions, layers), нет training data size, нет cost. Google стиль -- результаты есть, reproducibility не в приоритете.
- HellaSwag observation -- важное: +100 steps на related data даёт +4% accuracy. Это ставит под вопрос все benchmark comparisons -- pretraining data composition matters больше, чем model capability.
