---
title: "Revisiting Few-sample BERT Fine-tuning"
url: https://arxiv.org/abs/2006.05987
authors: [Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, Yoav Artzi]
year: 2021
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - BERT
  - Review
Date: 2021-11-03
Organization: Stanford University / ASAPP / Cornell
Parent item:
  - "[[BERT]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Few-shot Fine-tuning|Few-shot Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
raw: "[[02 Areas/ML & DL/raw/papers/revisiting-few-sample-bert-fine-tuning/paper.txt]]"
---

# Revisiting Few-sample BERT Fine-tuning

**Authors:** Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, Yoav Artzi (Stanford / ASAPP / Penn State / Cornell)
**Published:** ICLR 2021 (arXiv:2006.05987v3, Mar 2021)
**URL:** https://arxiv.org/abs/2006.05987

## TL;DR

Систематическое исследование нестабильности BERT fine-tuning в few-sample сценариях. Выявлены три источника нестабильности: (1) **смещённый BERTAdam** (без bias correction), (2) **плохая инициализация верхних слоёв** для downstream задач, (3) **слишком малое число итераций** (3 эпохи по умолчанию). Три соответствующих исправления — debiased Adam, слоёвая ре-инициализация, длинное обучение — устраняют нестабильность и делают многократные random restarts ненужными.

## Problem

BERT fine-tuning на малых датасетах (<10k примеров) нестабильна: одинаковые обучения с разными random seeds дают кардинально разные результаты, иногда деградируя до случайного угадывания. На RTE, например, **48% random trials** достигают accuracy < 55% (близко к случайному). Практики вынуждены запускать много random trials и выбирать лучший checkpoint по валидации.

Авторы идентифицируют три причины и предлагают конкретные исправления.

## Method

### Experimental Setup

- BERT_Large (uncased, 24 layers), batch 32, dropout 0.1, peak LR 2×10⁻⁵
- 20 random seeds для каждого эксперимента
- Оценка на 8 GLUE датасетах (RTE, MRPC, STS-B, CoLA + 4 dowsampled)
- Основной фокус: RTE, MRPC, STS-B, CoLA (< 10k train examples)

### Issue 1: BERTAdam — Missing Bias Correction (Section 4)

**Проблема:** Devlin et al. (2019) реализовали Adam **без bias correction** (строки 9-10 алгоритма Adam). Эта ошибка распространилась в HuggingFace Transformers, AllenNLP, GluonNLP и большинство community библиотек.

**Механизм:** Adam инициализирует моменты mt и vt нулями. В начале обучения они сильно смещены к нулю. Bias correction исправляет это через:
```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
```
Без correction — **overestimation learning rate** в начале обучения. При fine-tuning малых датасетов (< 1k итераций) этот bias значим на протяжении всего обучения (Figure 1).

**Исправление:** использовать стандартный debiased Adam (PyTorch `torch.optim.AdamW`).

**Результат (Figure 2):** Bias correction значительно снижает variance. На RTE: 48% degenerate runs → редкие outliers. C debiased Adam достаточно **5-10 random trials** вместо 50+.

### Issue 2: Top Layers Overspecialized for Pre-training (Section 5)

**Проблема:** Верхние Transformer блоки BERT специализируются под MLM/NSP pre-training objective и плохо служат инициализацией для downstream задач. Probing literature (Tenney et al., 2019) показала: промежуточные слои более переносимы.

**Исправление: Re-initialization (Re-init):**
Re-инициализируются pooler layer и top L BERT Transformer блоков стандартной BERT инициализацией N(0, 0.02²). L подбирается по валидационной производительности.

**Результат (Figure 5, Table 1):** Re-init стабильно улучшает среднее performance:
- RTE: 69.5 → **72.6%** (+3.1%)
- MRPC: 90.8 → **91.4 F1** (+0.6)
- CoLA: 63.0 → **64.2 MCC** (+1.2)

Важно: уже re-init только pooler дает улучшение; дальнейшие слои помогают больше до определённого L, затем performance плато и деградирует.

Re-init → faster convergence (Figure 6), уменьшение L2 distance изменений для верхних блоков (Figure 7) → меньше "работы" при fine-tuning.

### Issue 3: Fixed 3 Epochs is Suboptimal (Section 6)

**Проблема:** Devlin et al. (2019) рекомендовали 3 эпохи для GLUE. Эта рекомендация принята всем сообществом. Но 3 эпохи соответствуют **всего 96 шагам** для 1k-sample датасета — модель не успевает сойтись.

**Исправление:** Тюнить число итераций как гиперпараметр. Тестировалось {200, 400, 800, 1600, 3200} шагов.

**Результат (Figure 8):** Дольше обучение улучшает и performance, и стабильность:
- 1k MRPC: 80.5±3.3 (3 эпохи) → **86.0±1.2** (longer with Re-init)
- 1k MNLI: 52.2±4.2 (3 эпохи) → 67.5±1.1 (longer standard) / **68.8±0.5** (longer Re-init)

Разные датасеты требуют разного числа итераций — нет универсального решения. Рекомендация: тюнить на validation.

## Comparing Existing Methods (Section 7, Table 2)

При использовании debiased Adam (вместо BERTAdam), эффект большинства ранее предложенных методов для стабилизации **значительно снижается**:

| Метод | RTE | MRPC | STS-B | CoLA |
|-------|-----|------|-------|------|
| Standard (baseline) | 69.5±2.5 | 90.8±1.3 | 89.0±0.6 | 63.0±1.5 |
| Int. Task (MNLI pretrain) | **81.8±1.7** | 91.8±1.0 | 89.2±0.3 | 63.9±1.8 |
| LLRD | 69.7±3.2 | 91.3±1.1 | 89.2±0.4 | 63.0±2.5 |
| Mixout | 71.3±1.4 | 90.4±1.4 | 89.2±0.4 | 61.6±1.7 |
| Pre-trained WD | 69.6±2.1 | 90.8±1.3 | 89.0±0.5 | 63.4±1.5 |
| **Re-init** | **72.6±1.6** | **91.4±0.8** | **89.4±0.2** | **64.2±1.6** |
| **Longer** | 72.3±1.9 | 91.0±1.3 | 89.6±0.3 | 62.4±1.7 |

Intermediate task transfer (через MNLI) по-прежнему значительно помогает на RTE (+12.3%). Re-init стабильно лучше всех других методов. Mixout, LLRD, Pre-trained WD — не выше baseline после debiasing.

## My notes

- Работа — **критически важная практическая инструкция** для fine-tuning BERT. Всё, что вы читали в туториалах "3 эпохи с BERTAdam" — вероятно, неоптимально.
- **BERTAdam был багом**, распространившимся в весь open-source NLP экосистем. HuggingFace исправил это в commit ec07cf5a (Jul 11, 2019), но многие старые туториалы и репозитории всё ещё используют biased версию.
- Re-init top layers — интуитивный и эффективный трюк. Верхние слои BERT "помнят" MLM task, а не синтаксис/семантику. Для конкретной downstream задачи лучше дать им чистый старт.
- LLRD (Layer-wise LR Decay) и Mixout оказались неэффективными при правильном Adam — это хороший пример того, как неправильный baseline маскирует реальную картину.
- "Tuning training iterations" — самый практичный совет. 3 эпохи = 96 шагов для 1k датасета совершенно недостаточно. Тюнить итерации обязательно.
