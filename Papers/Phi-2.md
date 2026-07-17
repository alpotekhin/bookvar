---
title: "Exponentially Faster Language Modeling (UltraFastBERT)"
url: https://arxiv.org/abs/2311.10770
authors: "Peter Belcak, Roger Wattenhofer"
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/phi-2/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]"
---

# Exponentially Faster Language Modeling (UltraFastBERT)

**Authors:** Peter Belcak, Roger Wattenhofer (ETH Zurich)
**Published:** 2023 (arXiv:2311.10770v2, Nov 2023)
**URL:** https://arxiv.org/abs/2311.10770

## TL;DR

UltraFastBERT -- вариант BERT, в котором feedforward networks заменены на **fast feedforward networks (FFF)**, организующие нейроны в сбалансированное бинарное дерево. При inference используется только **0.3% нейронов** (12 из 4095), при этом downstream performance на GLUE сопоставим с BERT-base. Достигнут CPU speedup 78x по сравнению с dense FFN через **conditional matrix multiplication (CMM)**.

## Problem

Feedforward-слои содержат большинство параметров LLM, но не все нейроны нужны для каждого входа. Стандартный FFN выполняет dense matrix multiplication (DMM), задействуя все нейроны за O(n). Вопрос: можно ли использовать экспоненциально меньше нейронов (O(log n)) без потери качества?

## Method

### Fast Feedforward Networks (FFF)

Нейроны организованы в **balanced binary tree** глубины D+1. Inference -- проход по одной ветке дерева:
1. На каждом уровне вычисляется dot product входа с весами текущего узла.
2. По знаку результата выбирается левый или правый потомок.
3. Каждый посещённый узел вносит вклад в output через свои output weights.

Для дерева из 2^D - 1 нейронов inference проходит только D нейронов: **O(log n)** вместо O(n).

### Conditional Matrix Multiplication (CMM)

Ключевая операция -- CMM: строки входа перемножаются с колонками весов по одной за раз, и выбор следующей колонки зависит от результата предыдущей операции. Это **не branching** на уровне инструкций CPU -- это просто изменение offset указателя.

### Модификации FFF для UltraFastBERT

1. Единообразная обработка leaf и non-leaf узлов (одинаковая GeLU activation, output weights у всех узлов, без output biases).
2. Leaf size = 1.
3. Несколько FFF деревьев параллельно -- выходы суммируются. Модель обозначается UltraFastBERT-KxD (K деревьев глубины D+1).

### Обучение

На базе crammedBERT (Geiping & Goldstein, 2023): 1 день на одной A6000 GPU. Без dropout, 1-cycle triangular LR schedule. Финальная модель UltraFastBERT-1x11-long обучена 2x дольше.

## Key Results

### GLUE downstream (Table 1)

| Model | Neurons used | RTE | MRPC | STSB | SST-2 | MNLI | QNLI | QQP | Avg (w/o CoLA) |
|-------|-------------|-----|------|------|-------|------|------|-----|--------|
| crammedBERT-3072 | 100% | 58.8 | 87.6 | 85.2 | 91.9 | 82.8 | 90.4 | 89.0 | 83.6 |
| UltraFastBERT-1x11-long | **0.3%** | 60.7 | 87.5 | 86.4 | 89.9 | 81.3 | 89.7 | 87.6 | 83.0 |
| BERT-base | 100% | 66.4 | 88.9 | 85.8 | 93.5 | 83.4 | 90.5 | 71.2 | 83.0 |

- Без CoLA сохраняется >= 98.6% performance при любой глубине FFF.
- CoLA -- единственная задача с заметной деградацией (требует больше finetuning эпох: 15 vs 5).

### Inference speedup (Table 2)

| | CPU Level 2 (fair) | GPU Naive CUDA (fair) |
|---|---|---|
| UltraFastBERT-1x11 vs BERT-base-4095 | **255x** | **118x** |
| UltraFastBERT-1x11 vs BERT-base-3072 (Level 3) | **78x** | **3.15x** (vs native fused) |

- Теоретический предел: 341x (= 4095/12).
- CPU выигрывает больше, т.к. sequential execution напрямую бенефитит от меньшего числа операций.
- GPU speedup ограничен отсутствием native CMM kernel в CUDA/cuBLAS.

### Градиент деградации

По мере углубления дерева (от 3072x0 до 1x11) performance падает плавно -- большая часть падения приходится на CoLA. Все остальные задачи стабильны.

## My notes

